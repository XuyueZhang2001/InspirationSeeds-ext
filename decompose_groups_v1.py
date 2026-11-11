"""
Adaptive grouping of CLIP-SAE features and group-conditioned reference images.

Stage 1 of the revised decomposition (reuses the existing CLIP, SAE and Kandinsky models;
no new training). Adapted from ConceptExpress (pre-clustering -> threshold -> constrained
post-clustering with FINCH), applied to SAE decoder directions instead of attention maps.

For each source image:
  1. e = CLIP(image), a = SAE_enc(e); keep the top-K features by activation.
  2. d_j = Dec(1_j) - Dec(0) = j-th column of the SAE decoder (no bias);
     u_j = d_j / ||d_j||, distance D_jk = 1 - u_j . u_k
  3. FINCH pre-clustering on {u_j} -> hierarchy of partitions.
  4. Threshold: take the level whose cluster count N' is the smallest one above N_max;
     delta = largest intra-cluster distance (max pairwise member distance) at that level.
     If no usable level exists (level 1 already <= N_max, or only singleton clusters),
     the count is marked unstable; --delta-fallback is used if given, otherwise delta is
     estimated from FINCH level 1 and post-clustering starts from that level.
  5. Post-clustering from that level: FINCH merges, but
       - no edge between clusters whose centroid distance exceeds delta, and
       - a merged group is accepted only if its spread (mean pairwise member distance)
         is <= chain_factor * delta (prevents chaining); otherwise only mutual-nearest
         pairs that satisfy it are merged. The mean is used instead of the max because
         the max pairwise distance grows with group size and would block valid merges.
     Stops when nothing merges; the remaining groups are the N factors.
  6. Group checks (recorded only): coherence, salience, redundancy.
  7. Conditions per group i (a^(i) keeps only the group's activations):
       e_group_i  = Dec(a^(i)) = W_dec a^(i) + b          (t = 1)
       e_anchor_i = (1 - t) e + t e_group_i               (other t)
     Each condition is decoded by Kandinsky with the same seed.

Usage:
  python decompose_groups.py --image path/to/img.jpeg --output-dir groups_out
  python decompose_groups.py --image-dir hf_samples/sources --output-dir hf_groups
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import load_image
from PIL import Image, ImageDraw, ImageFont
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

from decompose import embed_image, load_kandinsky, load_sae

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# --------------------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------------------
def normalize_rows(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def cos_dist(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """1 - cosine similarity between rows of U and V (rows assumed normalized)."""
    return 1.0 - U @ V.T


def diameter(U: np.ndarray, members) -> float:
    """Max pairwise cosine distance among the given points (0 for a single point)."""
    if len(members) < 2:
        return 0.0
    P = U[list(members)]
    return float(cos_dist(P, P).max())


def spread(U: np.ndarray, members) -> float:
    """Mean pairwise cosine distance among the given points (0 for a single point)."""
    if len(members) < 2:
        return 0.0
    P = U[list(members)]
    Dm = cos_dist(P, P)
    return float(Dm[np.triu_indices(len(members), 1)].mean())


def centroids(U: np.ndarray, clusters) -> np.ndarray:
    return normalize_rows(np.stack([U[list(c)].mean(0) for c in clusters]))


# --------------------------------------------------------------------------------------
# FINCH
# --------------------------------------------------------------------------------------
def first_neighbor_components(C: np.ndarray, delta: float | None = None):
    """
    One FINCH step on points C (normalized rows).
    Edges i -- kappa_i (first neighbor); kappa_i == kappa_j is covered transitively.
    With delta, edges longer than delta are dropped.
    Returns (n_components, labels, kappa, dist_to_kappa).
    """
    n = len(C)
    if n == 1:
        return 1, np.zeros(1, dtype=int), np.zeros(1, dtype=int), np.zeros(1)
    D = cos_dist(C, C)
    np.fill_diagonal(D, np.inf)
    kappa = D.argmin(1)
    dk = D[np.arange(n), kappa]
    keep = np.ones(n, dtype=bool) if delta is None else dk <= delta
    rows, cols = np.arange(n)[keep], kappa[keep]
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(A, directed=False)
    return n_comp, labels, kappa, dk


def finch_hierarchy(U: np.ndarray):
    """Unconstrained FINCH. Returns a list of partitions (level 1, 2, ...), each a list of member lists."""
    levels = []
    clusters = [[i] for i in range(len(U))]
    while len(clusters) > 1:
        C = centroids(U, clusters)
        n_comp, labels, _, _ = first_neighbor_components(C)
        if n_comp == len(clusters):
            break
        merged = [[] for _ in range(n_comp)]
        for ci, lab in enumerate(labels):
            merged[lab].extend(clusters[ci])
        clusters = [sorted(m) for m in merged]
        levels.append(clusters)
    return levels


def max_intra_distance(U: np.ndarray, clusters) -> float:
    return max(diameter(U, c) for c in clusters)


def post_cluster(U: np.ndarray, clusters, delta: float, chain_factor: float):
    """Constrained FINCH merging starting from `clusters`. Returns (groups, n_rounds)."""
    clusters = [list(c) for c in clusters]
    limit = chain_factor * delta
    rounds = 0
    while len(clusters) > 1:
        C = centroids(U, clusters)
        n_comp, labels, kappa, dk = first_neighbor_components(C, delta=delta)
        if n_comp == len(clusters):
            break
        new_clusters, used = [], set()
        for comp in range(n_comp):
            idx = [i for i in range(len(clusters)) if labels[i] == comp]
            if len(idx) == 1:
                continue
            members = sorted(m for i in idx for m in clusters[i])
            if spread(U, members) <= limit:
                new_clusters.append(members)
                used.update(idx)
            else:
                # chaining: only merge mutual nearest pairs that stay compact
                for i in idx:
                    j = int(kappa[i])
                    if i < j and int(kappa[j]) == i and i not in used and j not in used and dk[i] <= delta:
                        pair = sorted(clusters[i] + clusters[j])
                        if spread(U, pair) <= limit:
                            new_clusters.append(pair)
                            used.update([i, j])
        if not used:
            break
        new_clusters += [clusters[i] for i in range(len(clusters)) if i not in used]
        clusters = new_clusters
        rounds += 1
    return clusters, rounds


def distance_stats(U: np.ndarray):
    D = cos_dist(U, U)
    n = len(U)
    off = D[np.triu_indices(n, 1)] if n > 1 else np.zeros(1)
    Dn = D + np.diag(np.full(n, np.inf))
    nn = Dn.min(1) if n > 1 else np.zeros(1)
    return {
        "pairwise_mean": float(off.mean()),
        "pairwise_median": float(np.median(off)),
        "pairwise_p10": float(np.percentile(off, 10)),
        "pairwise_p90": float(np.percentile(off, 90)),
        "pairwise_min": float(off.min()),
        "nearest_neighbor_median": float(np.median(nn)),
    }


def adaptive_groups(U: np.ndarray, n_max: int, delta_fallback: float | None, chain_factor: float,
                    delta_mode: str = "quantile", delta_quantile: float = 0.1, delta_cap: float | None = 0.9):
    """
    Full stage: FINCH hierarchy -> delta -> constrained post-clustering.

    delta_mode = "max"      : ConceptExpress-style; delta = largest intra-cluster distance at the
                              FINCH level just above n_max; post-clustering starts from that level.
    delta_mode = "quantile" : delta = q-quantile of all pairwise distances among the selected
                              features (only pairs clearly closer than a typical pair may merge);
                              post-clustering starts from single features. Used because SAE
                              decoder directions are close to orthogonal, so the max intra-cluster
                              distance is about as large as inter-cluster distances.
                              delta is capped at delta_cap: random directions in 1280-d have cosine
                              distance about 1 +- 0.03, so without a cap the closest *random* pairs
                              would also be merged.
    """
    levels = finch_hierarchy(U)
    info = {
        "finch_level_sizes": [len(lv) for lv in levels],
        "distance_stats": distance_stats(U),
        "delta_mode": delta_mode,
        "chosen_level": None,
        "n_prime": None,
        "delta": None,
        "delta_source": None,
        "count_unstable": False,
        "post_rounds": 0,
    }

    if delta_mode == "quantile":
        n = len(U)
        off = cos_dist(U, U)[np.triu_indices(n, 1)]
        if delta_fallback is not None:
            delta, source = delta_fallback, "fallback"
        else:
            delta, source = float(np.quantile(off, delta_quantile)), f"quantile {delta_quantile}"
            if delta_cap is not None and delta > delta_cap:
                delta, source = delta_cap, f"quantile {delta_quantile} capped at {delta_cap}"
        info.update({"delta": delta, "delta_source": source})
        groups, rounds = post_cluster(U, [[i] for i in range(n)], delta, chain_factor)
        info["post_rounds"] = rounds
        return groups, levels, info

    above = [(li, lv) for li, lv in enumerate(levels) if len(lv) > n_max]
    chosen = min(above, key=lambda x: len(x[1])) if above else None
    if chosen is not None and any(len(c) > 1 for c in chosen[1]):
        li, start = chosen
        delta = max_intra_distance(U, start)
        info.update({"chosen_level": li + 1, "n_prime": len(start), "delta": delta, "delta_source": "estimated"})
    else:
        # No level above n_max with non-singleton clusters (typically: FINCH level 1 already
        # has <= n_max clusters). The count is marked unstable.
        info["count_unstable"] = True
        if delta_fallback is not None:
            start = chosen[1] if chosen is not None else (levels[0] if levels else [[i] for i in range(len(U))])
            delta = delta_fallback
            info.update({"delta": delta, "delta_source": "fallback"})
        elif levels and any(len(c) > 1 for c in levels[0]):
            # Estimate delta from the finest FINCH level instead, and post-cluster from it.
            start = levels[0]
            delta = max_intra_distance(U, start)
            info.update({"delta": delta, "delta_source": "estimated from FINCH level 1 (unstable)"})
        else:
            info["delta_source"] = "none (all singletons)"
            return [[i] for i in range(len(U))], levels, info
        info.update({"chosen_level": None if chosen is None else chosen[0] + 1, "n_prime": len(start)})

    groups, rounds = post_cluster(U, start, delta, chain_factor)
    info["post_rounds"] = rounds
    return groups, levels, info


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def generate(image_embeds, decoder, negative_emb, seed: int, steps: int, size: int):
    generator = torch.Generator("cuda").manual_seed(seed)
    return decoder(
        image_embeds=image_embeds.unsqueeze(0),
        negative_image_embeds=negative_emb,
        height=size,
        width=size,
        num_inference_steps=steps,
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]


def make_overview(source, recon, rows, out_path: Path, tile=224):
    """rows: list of (row_label, [(image, caption), ...])."""
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    n_cols = 2 + max((len(r[1]) for r in rows), default=0)
    label_w, gap, cap_h, head_h = 70, 6, 34, 4
    W = label_w + n_cols * (tile + gap) + gap
    H = head_h + len(rows) * (tile + cap_h + gap) + gap
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (row_label, items) in enumerate(rows):
        y = head_h + r * (tile + cap_h + gap) + gap
        draw.text((6, y + tile // 2 - 7), row_label, fill="black", font=font)
        cells = [(source, "source"), (recon, "recon")] + items
        for c, (img, cap) in enumerate(cells):
            x = label_w + c * (tile + gap) + gap
            canvas.paste(img.convert("RGB").resize((tile, tile)), (x, y))
            draw.multiline_text((x + 2, y + tile + 2), cap, fill="black", font=font, spacing=1)
    canvas.save(out_path, quality=92)


# --------------------------------------------------------------------------------------
# Per image
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def process_image(image_path: Path, args, models):
    prior, decoder, negative_emb, sae = models
    out = Path(args.output_dir) / f"{image_path.stem}_topk{args.top_k}_nmax{args.n_max}"
    if (out / "groups.json").exists() and not args.overwrite:
        print(f"[skip] {out}")
        return None
    out.mkdir(parents=True, exist_ok=True)

    source = load_image(str(image_path)).convert("RGB")
    e_half = embed_image(item=source, prior=prior)
    e = e_half.float()
    acts = sae.encode(e_half)

    # 1. top-K features
    top = [int(i) for i in reversed(acts.argsort()[-args.top_k:].tolist())]
    a_top = acts[top].float()

    # 2. decoder directions (no bias)
    W = sae.decoder.weight  # (1280, n_features)
    D = W[:, top].T.float()  # (K, 1280), d_j
    d_norm = D.norm(dim=1)
    U = normalize_rows(D.cpu().numpy().astype(np.float64))

    # 3-5. adaptive grouping
    groups, levels, ginfo = adaptive_groups(U, args.n_max, args.delta_fallback, args.chain_factor,
                                            args.delta_mode, args.delta_quantile,
                                            None if args.delta_cap <= 0 else args.delta_cap)

    # 6. group checks
    contrib_w = (a_top * d_norm).cpu().numpy()  # a_j ||d_j||
    total_w = float(contrib_w.sum())
    stats = []
    for g in groups:
        P = U[g]
        if len(g) > 1:
            S = P @ P.T
            coherence = float(S[np.triu_indices(len(g), 1)].mean())
        else:
            coherence = None
        stats.append({
            "members": g,
            "salience": float(contrib_w[g].sum() / total_w) if total_w > 0 else 0.0,
            "coherence": coherence,
            "diameter": diameter(U, g),
            "spread": spread(U, g),
        })
    order = sorted(range(len(groups)), key=lambda i: -stats[i]["salience"])
    groups = [groups[i] for i in order]
    stats = [stats[i] for i in order]
    N = len(groups)
    Cg = centroids(U, groups) if N > 0 else np.zeros((0, U.shape[1]))
    centroid_sim = (Cg @ Cg.T).tolist()

    # which groups get reference images
    generate_ids = [i for i in range(N) if stats[i]["salience"] >= args.min_salience][: args.n_max]
    decomposable = N >= 2

    # 7. conditions
    b = sae.decode(torch.zeros_like(acts)).float()
    e_group, rows = [], []
    for i, g in enumerate(groups):
        s = torch.zeros_like(acts)
        idx = torch.tensor([top[j] for j in g], device=acts.device)
        s[idx] = acts[idx]
        eg = sae.decode(s).float()
        e_group.append(eg)
        Wa = eg - b
        stats[i].update({
            "feature_ids": [top[j] for j in g],
            "activations": [float(a_top[j]) for j in g],
            "norm_Wa_over_bias": float(Wa.norm() / b.norm().clamp_min(1e-8)),
            "generated": i in generate_ids and decomposable,
        })
    eg_stack = torch.stack(e_group) if e_group else torch.zeros(0, e.shape[0], device=e.device)
    eg_n = torch.nn.functional.normalize(eg_stack, dim=1)
    group_embed_sim = (eg_n @ eg_n.T).tolist() if N else []

    cond_info = {}
    anchors = {}
    for t in args.t:
        anchors[t] = []
        for i in range(N):
            ea = (1 - t) * e + t * e_group[i]
            anchors[t].append(ea)
            cond_info.setdefault(f"t{t}", []).append({
                "group": i,
                "norm_over_e": float(ea.norm() / e.norm()),
                "cos_to_e": float(torch.nn.functional.cosine_similarity(ea, e, dim=0)),
            })

    # generation
    recon = generate(e_half, decoder, negative_emb, args.seed, args.steps, args.size)
    recon.save(out / "recon.jpeg")
    overview_rows = []
    if decomposable:
        for t in args.t:
            items = []
            for i in generate_ids:
                img = generate(anchors[t][i].to(e_half.dtype), decoder, negative_emb, args.seed, args.steps, args.size)
                img.save(out / f"ref_{i}_t{t}.jpeg")
                items.append((img, f"g{i} n={len(groups[i])}\nsal={stats[i]['salience']:.2f}"))
            overview_rows.append((f"t={t}", items))
    else:
        overview_rows.append(("N<2", []))
    make_overview(source, recon, overview_rows, out / "overview.jpeg")

    torch.save({
        "e": e.cpu(),
        "top_features": top,
        "groups": groups,
        "d_group": torch.stack([(D[g] * a_top[g, None]).sum(0) for g in groups]).cpu() if N else None,
        "e_group": eg_stack.cpu(),
        "bias": b.cpu(),
        "e_anchor": {f"t{t}": torch.stack(anchors[t]).cpu() for t in args.t} if N else {},
    }, out / "embeds.pt")

    meta = {
        "image": str(image_path),
        "top_k": args.top_k,
        "n_max": args.n_max,
        "delta_quantile": args.delta_quantile if args.delta_mode == "quantile" else None,
        "chain_factor": args.chain_factor,
        "min_salience": args.min_salience,
        "t": args.t,
        "seed": args.seed,
        "steps": args.steps,
        "size": args.size,
        "top_features": [{"feature_id": f, "activation": float(a)} for f, a in zip(top, a_top.tolist())],
        "finch_levels": [[[top[j] for j in c] for c in lv] for lv in levels],
        **ginfo,
        "N": N,
        "n_multi_feature_groups": sum(len(g) > 1 for g in groups),
        "decomposable": decomposable,
        "n_generated": len(generate_ids) if decomposable else 0,
        "groups": stats,
        "group_centroid_cosine": centroid_sim,
        "group_embedding_cosine": group_embed_sim,
        "conditions": cond_info,
        "norm_e": float(e.norm()),
        "norm_bias": float(b.norm()),
    }
    with open(out / "groups.json", "w") as f:
        json.dump(meta, f, indent=2)

    flag = " [count unstable]" if ginfo["count_unstable"] else ""
    delta_txt = "none" if ginfo["delta"] is None else f"{ginfo['delta']:.3f}"
    ds = ginfo["distance_stats"]
    print(f"{image_path.name}: pairwise dist median {ds['pairwise_median']:.3f} (p10 {ds['pairwise_p10']:.3f}), "
          f"FINCH levels {ginfo['finch_level_sizes']}, delta={delta_txt} [{ginfo['delta_source']}], "
          f"N={N} (multi-feature: {sum(len(g) > 1 for g in groups)}){flag}"
          + ("" if decomposable else " [not decomposable]"))
    return {"image": str(image_path), "folder": str(out), "N": N,
            "n_multi_feature_groups": sum(len(g) > 1 for g in groups), "n_prime": ginfo["n_prime"],
            "delta": ginfo["delta"], "count_unstable": ginfo["count_unstable"], "decomposable": decomposable,
            "saliences": [s["salience"] for s in stats]}


# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Adaptive SAE feature grouping and group-conditioned reference images")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="One source image")
    src.add_argument("--image-dir", help="Folder of source images")
    p.add_argument("--output-dir", default="groups_out")
    p.add_argument("--top-k", type=int, default=32, help="Number of top-activated SAE features (K)")
    p.add_argument("--n-max", type=int, default=6, help="Upper limit on the number of groups (N_max)")
    p.add_argument("--delta-mode", choices=["quantile", "max"], default="quantile",
                   help="quantile (default): delta = quantile of pairwise distances, post-cluster from single "
                        "features; max: ConceptExpress-style max intra-cluster distance")
    p.add_argument("--delta-quantile", type=float, default=0.1,
                   help="Quantile of pairwise distances used as delta in quantile mode")
    p.add_argument("--delta-cap", type=float, default=0.9,
                   help="Quantile mode: upper limit on delta (cosine distance; 0.9 = cosine similarity >= 0.1). "
                        "0 disables the cap")
    p.add_argument("--delta-fallback", type=float, default=None,
                   help="Fixed delta (quantile mode: overrides the quantile; max mode: used when the count is unstable)")
    p.add_argument("--chain-factor", type=float, default=1.0,
                   help="A merged group must have mean pairwise distance <= chain_factor * delta")
    p.add_argument("--min-salience", type=float, default=0.0,
                   help="Groups with a smaller contribution share get no reference image (0 = all)")
    p.add_argument("--t", type=lambda v: [float(x) for x in v.split(",")], default=[1.0, 0.75, 0.5],
                   help="Anchoring strengths; t=1 is the group-only condition")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=18, help="Kandinsky inference steps (decompose.py uses 18)")
    p.add_argument("--size", type=int, default=512, help="Kandinsky output size (decompose.py uses 512)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.image:
        paths = [Path(args.image)]
    else:
        paths = sorted(q for q in Path(args.image_dir).iterdir() if q.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise ValueError("No input images found.")

    prior, decoder, negative_emb = load_kandinsky()
    sae = load_sae()
    models = (prior, decoder, negative_emb, sae)

    summary = []
    for path in tqdm(paths, desc="images"):
        res = process_image(path, args, models)
        if res is not None:
            summary.append(res)

    if summary:
        out_root = Path(args.output_dir)
        with open(out_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        Ns = [s["N"] for s in summary]
        print(f"\n{len(summary)} image(s). N distribution: "
              + ", ".join(f"N={n}: {Ns.count(n)}" for n in sorted(set(Ns))))
        Ms = [s["n_multi_feature_groups"] for s in summary]
        print("multi-feature groups per image: " + ", ".join(f"{m}: {Ms.count(m)}" for m in sorted(set(Ms))))
        print(f"count unstable: {sum(s['count_unstable'] for s in summary)}, "
              f"not decomposable (N<2): {sum(not s['decomposable'] for s in summary)}")
        print(f"summary: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
