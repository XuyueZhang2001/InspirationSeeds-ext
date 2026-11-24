# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch",
#     "diffusers",
#     "scikit-learn",
#     "tqdm",
#     "transformers",
#     "huggingface-hub",
# ]
# ///
"""
Decompose an image into K concept clusters of CLIP-SAE features and generate one
input image per cluster with Kandinsky 2.2.

For each (image, K, top-k, cleaning fraction):
  1. e = CLIP image embedding; SAE-encode e and keep the top-k features.
  2. Cluster the decoded feature directions into K groups (k-means).
  3. Cleaning: for every feature, compute the distance to the nearest *other*
     cluster center; within each cluster, remove the given fraction of features
     with the smallest such distance.
  4. Cluster contribution d_i = W_dec s_i (model.decoder has no bias),
     residual r = e - sum_i d_i, natural proportion m_i = ||d_i|| / sum_j ||d_j||.
  5. One-vs-rest direction v_i = d_i - mean_{j != i} d_j,
     input embedding e_i = e + lambda * v_i   (or lambda * v_i / ||v_i|| with
     --direction-norm unit), decoded with Kandinsky into input_i.jpeg.

For K = 2 with lambda = 1 and no normalization, input_1.jpeg / input_0.jpeg
correspond to step_1.jpeg / step_-1.jpeg of the original script.

Filtering metrics are only recorded in meta.json; nothing is discarded.
"""

import argparse
import importlib.util
import json
import os
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.utils import load_image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

DEVICE = "cuda"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# {cluster_label: {feature_id: {"feature_id": int, "feature_value": float}}}
ClustersDict = Dict[int, Dict[int, Dict[str, float]]]


# --------------------------------------------------------------------------------------
# Clustering and cleaning
# --------------------------------------------------------------------------------------
def cluster_clip_embeddings(
    clip_embeds_dict: Dict[int, np.ndarray],
    k: int = 2,
    random_state: int | None = 0,
) -> Tuple[Dict[int, List[int]], np.ndarray, np.ndarray]:
    """K-means on the decoded feature directions.

    Returns (clusters, cluster_centers, labels): clusters maps label -> feature ids,
    labels is aligned with the insertion order of clip_embeds_dict.
    """
    keys = list(clip_embeds_dict)
    X = np.stack([clip_embeds_dict[key] for key in keys])

    # n_init="auto" (a single init) keeps K=2 identical to the original script;
    # use several restarts for K>2, where k-means is more sensitive to initialization.
    n_init = "auto" if k == 2 else 10
    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X)

    clusters: Dict[int, List[int]] = defaultdict(list)
    for key, label in zip(keys, labels, strict=True):
        clusters[int(label)].append(key)
    return dict(clusters), km.cluster_centers_, labels


def clean_clusters_by_distance(
    clusters_dict: ClustersDict,
    cluster_centers: np.ndarray,
    clip_embeds_dict: Dict[int, np.ndarray],
    removal_fraction: float = 0.5,
) -> Tuple[ClustersDict, ClustersDict]:
    """K-way cleaning.

    For each feature, compute its distance to the nearest *other* cluster center.
    Within each cluster, remove the `removal_fraction` of features with the smallest
    such distance (those lying closest to another cluster). For K=2 this is identical
    to the original two-cluster version.
    """
    labels = sorted(clusters_dict)
    clean_clusters: ClustersDict = {}
    dropped_features: ClustersDict = {}

    for label in labels:
        current = clusters_dict[label]
        other_centers = np.stack([cluster_centers[o] for o in labels if o != label])

        scored = []
        for feat_key, feat_data in current.items():
            emb = clip_embeds_dict[int(feat_data["feature_id"])]
            d_nearest_other = float(np.linalg.norm(other_centers - emb, axis=1).min())
            scored.append((d_nearest_other, feat_key, feat_data))
        scored.sort(key=lambda x: x[0])  # closest to another cluster first

        num_to_remove = int(len(scored) * removal_fraction)
        dropped_features[label] = {k: v for _, k, v in scored[:num_to_remove]}
        clean_clusters[label] = {k: v for _, k, v in scored[num_to_remove:]}

    return clean_clusters, dropped_features


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    n_labels = len(set(labels.tolist()))
    if n_labels < 2 or len(X) <= n_labels:
        return None
    return float(silhouette_score(X, labels))


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def embed_image(item, prior):
    return prior.interpolate([item], [1]).image_embeds[0]


@torch.inference_mode()
def generate_one(image_embeds, decoder, negative_emb):
    generator = torch.Generator("cuda")
    generator.manual_seed(1)
    images = decoder(
        image_embeds=image_embeds.unsqueeze(0),
        negative_image_embeds=negative_emb,
        height=512,
        width=512,
        num_inference_steps=18,
        num_images_per_prompt=1,
        generator=generator,
    ).images
    return images[0]


@torch.inference_mode()
def load_sae():
    from huggingface_hub import snapshot_download

    model_file_path = snapshot_download(repo_id="gytdau/clip-sae-128")
    model_py_path = os.path.join(model_file_path, "model.py")
    weights_path = os.path.join(model_file_path, "sparse_autoencoder_128.pth")

    spec = importlib.util.spec_from_file_location("model_module", model_py_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    SparseAutoencoder = getattr(model_module, "SparseAutoencoder")

    embedding_dim = 1280
    hidden_dim = embedding_dim * 128
    model = SparseAutoencoder(embedding_dim, hidden_dim)

    state_dict = torch.load(weights_path, map_location=torch.device("cuda"))
    model.half()
    model.load_state_dict(state_dict)
    model.eval()
    model.to("cuda")
    return model


@torch.inference_mode()
def load_kandinsky():
    image_encoder = (
        CLIPVisionModelWithProjection.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", subfolder="image_encoder"
        )
        .half()
        .to(DEVICE)
    )
    unet = (
        UNet2DConditionModel.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", subfolder="unet")
        .half()
        .to(DEVICE)
    )
    prior = KandinskyV22PriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior", image_encoder=image_encoder, torch_dtype=torch.float16
    ).to(DEVICE)
    decoder = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder", unet=unet, torch_dtype=torch.float16
    ).to(DEVICE)
    zero_embed = prior.get_zero_embed()
    return prior, decoder, zero_embed


# --------------------------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def feature_directions(sae_model, top_k_idxs: List[int], sae_features) -> Dict[int, np.ndarray]:
    """Decode each active feature as a one-hot vector (same as the original script).

    decode() adds a constant bias to every vector; this does not change the distances
    between features, so clustering and cleaning are unaffected.
    """
    embeds = {}
    for idx in top_k_idxs:
        one_hot = torch.zeros_like(sae_features)
        one_hot[idx] = 1
        embeds[idx] = sae_model.decode(one_hot).cpu().detach().float().numpy()
    return embeds


@torch.inference_mode()
def run(
    output_dir: Path,
    base_image_embed: torch.Tensor,
    base_sae_features: torch.Tensor,
    decoder,
    negative_emb,
    model,
    n_clusters: int,
    top_k: int,
    clean_frac: float,
    lam: float = 1.0,
    direction_norm: str = "none",
    save_steps: bool = False,
    image_path: str = "",
):
    output_dir.mkdir(exist_ok=True, parents=True)
    e = base_image_embed
    sae_feats = base_sae_features

    # 1. top-k active features
    top_k_idxs = [int(i) for i in reversed(sae_feats.argsort()[-top_k:].tolist())]

    # 2. cluster decoded feature directions
    embeds = feature_directions(model, top_k_idxs, sae_feats)
    clusters, centers, labels = cluster_clip_embeddings(embeds, k=n_clusters)
    clusters_json: ClustersDict = {
        c: {i: {"feature_id": i, "feature_value": sae_feats[i].item()} for i in ids} for c, ids in clusters.items()
    }
    sizes_before = {c: len(v) for c, v in clusters_json.items()}
    X_all = np.stack([embeds[i] for i in embeds])
    sil_before = safe_silhouette(X_all, labels)

    # 3. cleaning
    dropped: ClustersDict = {c: {} for c in clusters_json}
    if clean_frac > 0:
        clusters_json, dropped = clean_clusters_by_distance(clusters_json, centers, embeds, clean_frac)
    sizes_after = {c: len(v) for c, v in clusters_json.items()}
    kept_ids = [(c, fid) for c in sorted(clusters_json) for fid in clusters_json[c]]
    sil_after = safe_silhouette(
        np.stack([embeds[fid] for _, fid in kept_ids]) if kept_ids else np.zeros((0, 1)),
        np.array([c for c, _ in kept_ids]),
    )

    # 4. cluster contributions, residual, natural proportions
    cluster_labels = sorted(clusters_json)
    d = {}
    for c in cluster_labels:
        s = torch.zeros_like(sae_feats)
        for m in clusters_json[c].values():
            s[m["feature_id"]] = m["feature_value"]
        d[c] = model.decoder(s).float()  # W_dec s, no bias
    e32 = e.float()
    r = e32 - sum(d.values())
    d_norms = {c: d[c].norm().item() for c in cluster_labels}
    total = sum(d_norms.values())
    m_nat = {c: (d_norms[c] / total if total > 0 else 0.0) for c in cluster_labels}

    # 5. one-vs-rest directions and input embeddings
    K = len(cluster_labels)
    v, e_in = {}, {}
    for c in cluster_labels:
        others = [d[o] for o in cluster_labels if o != c]
        v[c] = d[c] - sum(others) / (K - 1)
        step = v[c] / v[c].norm().clamp_min(1e-8) if direction_norm == "unit" else v[c]
        e_in[c] = e32 + lam * step

    # 6. generate images
    generate_one(e, decoder, negative_emb).save(output_dir / "recon.jpeg")
    for c in tqdm(cluster_labels, desc=f"K={K} topk={top_k} frac={clean_frac}", leave=False):
        generate_one(e_in[c].to(e.dtype), decoder, negative_emb).save(output_dir / f"input_{c}.jpeg")
        if save_steps:
            half = e32 + 0.5 * (e_in[c] - e32)
            generate_one(half.to(e.dtype), decoder, negative_emb).save(output_dir / f"input_{c}_half.jpeg")

    # 7. save embeddings and metadata
    torch.save(
        {
            "e": e32.cpu(),
            "d": torch.stack([d[c] for c in cluster_labels]).cpu(),
            "r": r.cpu(),
            "v": torch.stack([v[c] for c in cluster_labels]).cpu(),
            "e_inputs": torch.stack([e_in[c] for c in cluster_labels]).cpu(),
            "cluster_labels": cluster_labels,
        },
        output_dir / "embeds.pt",
    )

    def feats(cl: Dict[int, Dict[str, float]]):
        return [{"feature_id": int(f["feature_id"]), "feature_value": float(f["feature_value"])} for f in cl.values()]

    meta = {
        "image": str(image_path),
        "K": K,
        "top_k": top_k,
        "clean_frac": clean_frac,
        "lambda": lam,
        "direction_norm": direction_norm,
        "clusters": {
            str(c): {
                "kept": feats(clusters_json[c]),
                "dropped": feats(dropped.get(c, {})),
                "size_before_cleaning": sizes_before.get(c, 0),
                "size_after_cleaning": sizes_after[c],
                "m": m_nat[c],
                "norm_d": d_norms[c],
                "norm_v": v[c].norm().item(),
            }
            for c in cluster_labels
        },
        "metrics": {
            "norm_e": e32.norm().item(),
            "norm_r": r.norm().item(),
            "r_ratio": (r.norm() / e32.norm()).item(),
            "min_cluster_size_after_cleaning": min(sizes_after.values()),
            "silhouette_before_cleaning": sil_before,
            "silhouette_after_cleaning": sil_after,
        },
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",")]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",")]


def parse_top_k(value: str):
    return "auto" if value.strip().lower() == "auto" else parse_int_list(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decompose images into K clusters of SAE features and generate one input image per cluster"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="Path to one input image")
    src.add_argument("--image-dir", type=str, help="Folder of input images (processed in sorted order)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output root (default: outputs)")
    parser.add_argument("--k", type=parse_int_list, default=[2], help="Comma-separated numbers of clusters, e.g. 2,3,4")
    parser.add_argument(
        "--top-k",
        type=parse_top_k,
        default="auto",
        help="'auto' (16*K, i.e. 32/48/64 for K=2/3/4) or comma-separated values, e.g. 32,64",
    )
    parser.add_argument(
        "--clean-clusters-frac",
        type=parse_float_list,
        default=[0.5],
        help="Comma-separated cleaning fractions (default: 0.5; 0 = no cleaning)",
    )
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0, help="Step size along v_i (default: 1)")
    parser.add_argument(
        "--direction-norm",
        choices=["none", "unit"],
        default="none",
        help="none: e + lambda*v_i (same scale as the paper); unit: e + lambda*v_i/||v_i||",
    )
    parser.add_argument("--save-steps", action="store_true", help="Also save the half step (lambda/2) for each cluster")
    parser.add_argument("--overwrite", action="store_true", help="Recompute folders that already contain meta.json")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_paths = sorted(p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise ValueError("No input images found.")

    configs = []
    for K in args.k:
        if K < 2:
            raise ValueError("K must be at least 2.")
        top_ks = [16 * K] if args.top_k == "auto" else args.top_k
        for top_k, frac in product(top_ks, args.clean_clusters_frac):
            if top_k < K:
                raise ValueError(f"top-k ({top_k}) must be at least K ({K}).")
            configs.append((K, top_k, frac))
    print(f"{len(image_paths)} image(s) x {len(configs)} config(s): {configs}")

    prior, decoder, negative_emb = load_kandinsky()
    model = load_sae()

    for image_path in tqdm(image_paths, desc="images"):
        base_image = load_image(str(image_path))
        e = embed_image(item=base_image, prior=prior)
        sae_feats = model.encode(e)

        for K, top_k, frac in configs:
            out = Path(args.output_dir) / f"{image_path.stem}_K{K}_topk{top_k}_frac{frac}_lam{args.lam}"
            if (out / "meta.json").exists() and not args.overwrite:
                print(f"Skipping (exists): {out}")
                continue
            run(
                output_dir=out,
                base_image_embed=e,
                base_sae_features=sae_feats,
                decoder=decoder,
                negative_emb=negative_emb,
                model=model,
                n_clusters=K,
                top_k=top_k,
                clean_frac=frac,
                lam=args.lam,
                direction_norm=args.direction_norm,
                save_steps=args.save_steps,
                image_path=str(image_path),
            )


if __name__ == "__main__":
    main()
