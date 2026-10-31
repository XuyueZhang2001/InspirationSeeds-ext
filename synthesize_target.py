"""
Synthesize a weighted target image from decompose.py output folders.

    python synthesize_target.py --folder <decompose output folder> --weights high,low,low,medium
    python synthesize_target.py --root hf_decomposed --k 4 --weights high,low,low,medium

Each folder must contain embeds.pt, meta.json, recon.jpeg and input_i.jpeg written by
decompose.py. input_i.jpeg corresponds to cluster i, and the i-th weight applies to input_i.

Steps:
  1. weights -> normalized proportions w   (levels: low=1, medium=3, high=9, or numbers)
     w is the label of the sample (proportions in embedding space).
  2. c_i = w_i / m_i, clipped to c_max      (m_i = natural proportion from meta.json)
  3. e_target = residual_scale * r + sum_i c_i d_i   (optionally rescaled to ||e||)
  4. generate the target with Kandinsky
  5. measure (default: --measure inputs):
       Re-encode recon.jpeg, every input_i.jpeg and the target with CLIP
       (and DINOv2 with --dino). All of them went through the same Kandinsky round trip.
       s_i = cos(f(target) - f(recon), f(input_i) - f(recon))
       i.e. how much the target moved from the reconstruction toward input_i.
     Rank check: for every pair with w_i > w_j (ratio >= --rank-min-ratio), require
       s_i > s_j. The sample passes if every such pair is ordered correctly.
     The older measurements (--measure projection / features) are kept for comparison;
     only they support --calibrate.
"""

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import nnls

from decompose import embed_image, load_kandinsky, load_sae

LEVEL_CENTERS = {"low": 1.0, "medium": 3.0, "high": 9.0}
LEVEL_RANGE = (0.8, 1.25)  # each level's interval = center * [0.8, 1.25], sampled log-uniformly


# --------------------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------------------
def parse_weights(text: str):
    """Return ("none", None), ("levels", [...]) or ("numeric", [...])."""
    text = text.strip().lower()
    if text == "none":
        return "none", None
    items = [t.strip() for t in text.split(",") if t.strip()]
    if all(t in LEVEL_CENTERS for t in items):
        return "levels", items
    try:
        values = [float(t) for t in items]
    except ValueError:
        raise ValueError(f"--weights must be 'none', levels from {list(LEVEL_CENTERS)}, or numbers; got '{text}'")
    if any(v < 0 for v in values) or sum(values) <= 0:
        raise ValueError("Numeric weights must be non-negative and not all zero.")
    return "numeric", values


def level_scores(levels, sample: bool, rng: random.Random):
    scores = []
    for lv in levels:
        center = LEVEL_CENTERS[lv]
        if sample:
            lo, hi = math.log(center * LEVEL_RANGE[0]), math.log(center * LEVEL_RANGE[1])
            scores.append(math.exp(rng.uniform(lo, hi)))
        else:
            scores.append(center)
    return scores


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def generate(image_embeds, decoder, negative_emb, seed: int):
    generator = torch.Generator("cuda").manual_seed(seed)
    return decoder(
        image_embeds=image_embeds.unsqueeze(0),
        negative_image_embeds=negative_emb,
        height=512,
        width=512,
        num_inference_steps=18,
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]


# --------------------------------------------------------------------------------------
# Feature extractors for the "inputs" measurement
# --------------------------------------------------------------------------------------
class ClipFeatures:
    name = "clip"

    def __init__(self, prior):
        self.prior = prior

    @torch.inference_mode()
    def __call__(self, image):
        return embed_image(item=image, prior=self.prior).float()


class DinoFeatures:
    name = "dino"

    def __init__(self, model_name: str = "facebook/dinov2-base"):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to("cuda").eval()

    @torch.inference_mode()
    def __call__(self, image):
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to("cuda")
        out = self.model(**inputs)
        return out.pooler_output[0].float()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm().item(), b.norm().item()
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float((a @ b) / (na * nb))


def input_scores(extractor, recon_img, input_imgs, target_img, cache):
    """s_i = cos(f(target) - f(recon), f(input_i) - f(recon)); also raw cos(f(target), f(input_i))."""
    if extractor.name not in cache:
        f_recon = extractor(recon_img)
        f_inputs = [extractor(im) for im in input_imgs]
        cache[extractor.name] = (f_recon, f_inputs)
    f_recon, f_inputs = cache[extractor.name]
    f_t = extractor(target_img)
    delta_t = f_t - f_recon
    s = [cos(delta_t, f_i - f_recon) for f_i in f_inputs]
    raw = [cos(f_t, f_i) for f_i in f_inputs]
    moved = float(delta_t.norm() / f_recon.norm().clamp_min(1e-8))
    return {"s": s, "raw_cos": raw, "relative_shift_from_recon": moved}


def rank_check(w, s, min_ratio: float):
    """Every pair with w_i / w_j >= min_ratio must have s_i > s_j."""
    pairs, correct = [], 0
    for i in range(len(w)):
        for j in range(len(w)):
            if i != j and w[j] > 0 and w[i] / w[j] >= min_ratio:
                ok = s[i] > s[j]
                pairs.append({"stronger": i, "weaker": j, "ok": ok})
                correct += ok
    n = len(pairs)
    return {
        "n_pairs": n,
        "n_correct": correct,
        "agreement": correct / n if n else None,
        "passed": (correct == n) if n else None,  # None: no ordered pairs (e.g. all equal)
        "pairs": pairs,
    }


# --------------------------------------------------------------------------------------
# Older measurements (kept for comparison / calibration)
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def measure_features(e_hat, sae, cluster_feature_ids):
    acts = sae.encode(e_hat.to(torch.float16))
    norms = []
    for ids in cluster_feature_ids:
        s = torch.zeros_like(acts)
        if ids:
            idx = torch.tensor(ids, device=acts.device)
            s[idx] = acts[idx]
        norms.append(sae.decoder(s).float().norm().item())
    total = sum(norms)
    return [n / total if total > 0 else 0.0 for n in norms], {"norm_d_hat": norms}


def measure_projection(e_hat, r, d):
    A = torch.cat([r[None], d], dim=0).T.double().cpu().numpy()  # (1280, K+1)
    b = e_hat.double().cpu().numpy()
    coef, resid = nnls(A, b)
    alpha = coef[1:]
    contrib = alpha * d.norm(dim=1).double().cpu().numpy()
    total = contrib.sum()
    w_hat = (contrib / total).tolist() if total > 0 else [0.0] * len(alpha)
    return w_hat, {"alpha": alpha.tolist(), "a0": float(coef[0]),
                   "relative_residual": float(resid / max(np.linalg.norm(b), 1e-12))}


@torch.inference_mode()
def measure_embedding(image, prior, sae, feature_ids, r, d, method):
    e_hat = embed_image(item=image, prior=prior).float()
    if method == "projection":
        return measure_projection(e_hat, r, d)
    return measure_features(e_hat, sae, feature_ids)


# --------------------------------------------------------------------------------------
# Overview image
# --------------------------------------------------------------------------------------
def make_overview(input_paths, labels, target_img, target_label, tile=256, text_h=60):
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    tiles = [Image.open(p).convert("RGB") for p in input_paths] + [target_img.convert("RGB")]
    texts = labels + [target_label]
    gap = 12
    width = len(tiles) * tile + (len(tiles) - 1) * gap + gap * 2
    canvas = Image.new("RGB", (width, tile + text_h), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for k, (img, txt) in enumerate(zip(tiles, texts)):
        if k == len(tiles) - 1:
            x += gap * 2
        canvas.paste(img.resize((tile, tile)), (x, 0))
        draw.multiline_text((x + 4, tile + 4), txt, fill="black", font=font, spacing=2)
        x += tile + gap
    return canvas


# --------------------------------------------------------------------------------------
# Per-folder processing
# --------------------------------------------------------------------------------------
def fmt(xs, nd=3):
    return "[" + ", ".join(f"{x:.{nd}f}" for x in xs) + "]"


def find_folders(root: Path):
    """All decompose.py output folders (containing embeds.pt and meta.json) under root, root included."""
    found = []
    for emb in sorted(root.rglob("embeds.pt")):
        f = emb.parent
        if "targets" in f.relative_to(root).parts:
            continue
        if (f / "meta.json").exists():
            found.append(f)
    return found


def process_folder(folder: Path, args, models):
    prior, decoder, negative_emb, sae, extractors = models
    E = torch.load(folder / "embeds.pt")
    meta = json.load(open(folder / "meta.json"))
    labels = list(E["cluster_labels"])
    K = len(labels)

    mode, values = parse_weights(args.weights)
    if args.k is not None and args.k != K:
        print(f"[skip] {folder}: K = {K}, --k = {args.k}")
        return None
    if mode != "none" and len(values) != K:
        print(f"[skip] {folder}: K = {K}, but {len(values)} weights given")
        return None

    m = [meta["clusters"][str(c)]["m"] for c in labels]
    feature_ids = [[f["feature_id"] for f in meta["clusters"][str(c)]["kept"]] for c in labels]
    input_paths = [folder / f"input_{c}.jpeg" for c in labels]
    recon_path = folder / "recon.jpeg"
    if args.measure == "inputs" and not (recon_path.exists() and all(p.exists() for p in input_paths)):
        print(f"[skip] {folder}: recon.jpeg or input_i.jpeg missing (needed for --measure inputs)")
        return None

    # weights -> w, c
    rng = random.Random(args.score_seed)
    if mode == "levels":
        scores = level_scores(values, args.sample_scores, rng)
    elif mode == "numeric":
        scores = values
    else:
        scores = None
    if scores is not None:
        total = sum(scores)
        w = [s / total for s in scores]
        c = [w_i / m_i if m_i > 0 else float("inf") for w_i, m_i in zip(w, m)]
    else:
        w = list(m)
        c = [1.0] * K

    # output folder
    if args.tag:
        tag = args.tag
    elif mode == "levels":
        tag = "-".join(values) + (f"_s{args.score_seed}" if args.sample_scores else "")
    elif mode == "numeric":
        tag = "w" + "-".join(f"{x:.2f}" for x in w)
    else:
        tag = "none"
    out = folder / "targets" / tag
    if (out / "target_meta.json").exists() and not args.overwrite:
        print(f"[skip] {out} already exists (use --overwrite)")
        return None
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {folder}")
    print(f"K = {K}, mode = {mode}, measure = {args.measure}")
    print(f"m = {fmt(m)}")
    print(f"w = {fmt(w)}")

    # c_max
    warnings = []
    c_requested = list(c)
    over = [i for i, ci in enumerate(c) if ci > args.c_max]
    if over:
        warnings.append(
            f"{folder.name}: requested weights need c above c_max={args.c_max} for "
            + ", ".join(f"input_{labels[i]} (c={c[i]:.2f})" for i in over)
            + ("" if args.allow_exceed else f"; clipped to {args.c_max}")
        )
    if not args.allow_exceed:
        c = [min(ci, args.c_max) for ci in c]

    e = E["e"].cuda().float()
    d = E["d"].cuda().float()
    r = E["r"].cuda().float()

    def synthesize(coefs):
        c_t = torch.tensor(coefs, device="cuda", dtype=torch.float32)
        e_t = args.residual_scale * r + (c_t[:, None] * d).sum(0)
        if args.rescale_norm:
            e_t = e_t * (e.norm() / e_t.norm().clamp_min(1e-8))
        return e_t

    # ---- generation (+ optional calibration with embedding-space measurements)
    history = []
    calibrate = args.calibrate and args.measure in ("projection", "features") and mode != "none"
    n_runs = 1 + args.max_iters if calibrate else 1
    stop_reason = "single_run"
    for it in range(n_runs):
        e_t = synthesize(c)
        img = generate(e_t.to(torch.float16), decoder, negative_emb, args.seed)
        img.save(out / f"target_iter{it}.jpeg")
        entry = {"iteration": it, "c": c, "norm_e_target_over_e": (e_t.norm() / e.norm()).item()}
        if args.measure in ("projection", "features"):
            w_hat, info = measure_embedding(img, prior, sae, feature_ids, r, d, args.measure)
            err = max(abs(a - b) for a, b in zip(w_hat, w))
            entry.update({"w_hat": w_hat, "measurement": info, "max_abs_error": err})
            print(f"iter {it}: c = {fmt(c, 2)}  w_hat = {fmt(w_hat)}  max|w_hat - w| = {err:.3f}")
        else:
            entry["max_abs_error"] = 0.0
            print(f"iter {it}: c = {fmt(c, 2)}")
        history.append(entry)

        if not calibrate:
            break
        if entry["max_abs_error"] <= args.tolerance:
            stop_reason = "within_tolerance"
            break
        if it > 0 and entry["max_abs_error"] >= min(h["max_abs_error"] for h in history[:-1]) - 1e-3:
            stop_reason = "no_improvement"
            break
        new_c = [ci * (wi / max(wh, args.w_hat_floor)) ** args.damping
                 for ci, wi, wh in zip(c, w, entry["w_hat"])]
        if not args.allow_exceed:
            new_c = [min(max(ci, 0.0), args.c_max) for ci in new_c]
        if all(abs(a - b) < 1e-6 for a, b in zip(new_c, c)):
            stop_reason = "pinned_at_c_max"
            break
        c = new_c
        stop_reason = "max_iters"

    best = min(history, key=lambda h: h["max_abs_error"])
    shutil.copy(out / f"target_iter{best['iteration']}.jpeg", out / "target.jpeg")
    target_img = Image.open(out / "target.jpeg").convert("RGB")

    # ---- input-similarity measurement and rank check
    input_measure = {}
    rank = {}
    if args.measure == "inputs":
        recon_img = Image.open(recon_path).convert("RGB")
        input_imgs = [Image.open(p).convert("RGB") for p in input_paths]
        cache = {}
        for ex in extractors:
            res = input_scores(ex, recon_img, input_imgs, target_img, cache)
            input_measure[ex.name] = res
            rank[ex.name] = rank_check(w, res["s"], args.rank_min_ratio) if mode != "none" else None
            msg = f"{ex.name}: s = {fmt(res['s'])}  shift from recon = {res['relative_shift_from_recon']:.3f}"
            if rank[ex.name] is not None:
                rk = rank[ex.name]
                msg += (f"  rank {rk['n_correct']}/{rk['n_pairs']}"
                        + ("" if rk["passed"] is None else (" PASS" if rk["passed"] else " FAIL")))
            print(msg)

    # overall rank result: all extractors that have ordered pairs must pass
    rank_results = [v["passed"] for v in rank.values() if v is not None and v["passed"] is not None]
    rank_passed = all(rank_results) if rank_results else None

    # ---- overview
    if mode == "levels":
        in_labels = [f"input_{c_}: {lv}\nw={wi:.2f}" for c_, lv, wi in zip(labels, values, w)]
    else:
        in_labels = [f"input_{c_}\nw={wi:.2f}" for c_, wi in zip(labels, w)]
    if args.measure == "inputs":
        lines = [f"{name} s=" + ", ".join(f"{x:.2f}" for x in res["s"]) for name, res in input_measure.items()]
        status = "n/a" if rank_passed is None else ("PASS" if rank_passed else "FAIL")
        tgt_label = "target\n" + "\n".join(lines) + f"\nrank: {status}"
    else:
        tgt_label = "target  w_hat=\n" + ", ".join(f"{x:.2f}" for x in best["w_hat"])
    make_overview(input_paths, in_labels, target_img, tgt_label,
                  text_h=44 + 16 * max(0, len(input_measure) + 1 - 2)).save(out / "overview.jpeg")

    result = {
        "folder": str(folder),
        "K": K,
        "mode": mode,
        "levels": values if mode == "levels" else None,
        "scores": scores,
        "sample_scores": args.sample_scores,
        "score_seed": args.score_seed if args.sample_scores else None,
        "w": w,  # label
        "m": m,
        "c_requested": c_requested,
        "c_used": best["c"],
        "c_max": args.c_max,
        "allow_exceed": args.allow_exceed,
        "rescale_norm": args.rescale_norm,
        "residual_scale": args.residual_scale,
        "seed": args.seed,
        "measure": args.measure,
        "input_measure": input_measure,
        "rank_min_ratio": args.rank_min_ratio,
        "rank_check": rank,
        "rank_passed": rank_passed,
        "calibrate": calibrate,
        "stop_reason": stop_reason,
        "iterations": history,
        "best_iteration": best["iteration"],
        "c_max_warnings": warnings,
    }
    if args.measure != "inputs":
        result.update({"best_w_hat": best["w_hat"], "best_max_abs_error": best["max_abs_error"],
                       "passed_tolerance": best["max_abs_error"] <= args.tolerance})
    with open(out / "target_meta.json", "w") as f:
        json.dump(result, f, indent=2)

    status = "n/a" if rank_passed is None else ("PASS" if rank_passed else "FAIL")
    print(f"saved to {out}: rank check {status}")
    return {
        "folder": str(folder),
        "target": str(out / "target.jpeg"),
        "K": K,
        "w": w,
        "m": m,
        "c_used": best["c"],
        "s": {k: v["s"] for k, v in input_measure.items()},
        "rank_agreement": {k: (v["agreement"] if v else None) for k, v in rank.items()},
        "rank_passed": rank_passed,
        "c_max_exceeded": bool(over),
        "warnings": warnings,
    }


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Synthesize weighted targets from decompose.py output folders")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--folder", help="One decompose.py output folder")
    src.add_argument("--root", help="Process every decompose.py output folder found under this directory")
    p.add_argument("--weights", required=True,
                   help="'none', levels (e.g. high,low,low,medium) or numbers (e.g. 0.6,0.2,0.1,0.1), in input_0.. order")
    p.add_argument("--k", type=int, default=None, help="Only process folders with this K")
    p.add_argument("--sample-scores", action="store_true", help="Sample level scores within their intervals")
    p.add_argument("--score-seed", type=int, default=0, help="Seed for --sample-scores")
    p.add_argument("--c-max", type=float, default=3.0)
    p.add_argument("--allow-exceed", action="store_true", help="Do not clip c to c_max")
    p.add_argument("--rescale-norm", action="store_true", help="Rescale the target embedding to ||e||")
    p.add_argument("--residual-scale", type=float, default=1.0, help="Coefficient on the residual r (default 1)")
    p.add_argument("--seed", type=int, default=1, help="Kandinsky generation seed (decompose.py uses 1)")
    p.add_argument("--tag", type=str, default=None, help="Output subfolder name (default: derived from weights)")
    p.add_argument("--overwrite", action="store_true", help="Recompute targets that already exist")
    # measurement
    p.add_argument("--measure", choices=["inputs", "projection", "features"], default="inputs",
                   help="inputs (default): similarity to the input images; projection/features: older w_hat")
    p.add_argument("--dino", action="store_true", help="Also measure with DINOv2 features (with --measure inputs)")
    p.add_argument("--rank-min-ratio", type=float, default=1.5,
                   help="Only pairs with w_i / w_j >= this ratio are rank-checked")
    # optional calibration (projection / features only)
    p.add_argument("--calibrate", action="store_true", help="Iterative calibration (projection/features only)")
    p.add_argument("--tolerance", type=float, default=0.1)
    p.add_argument("--max-iters", type=int, default=2)
    p.add_argument("--damping", type=float, default=0.5)
    p.add_argument("--w-hat-floor", type=float, default=0.02)
    args = p.parse_args()
    parse_weights(args.weights)  # validate before loading models
    if args.calibrate and args.measure == "inputs":
        print("Note: --calibrate is ignored with --measure inputs (no absolute proportions to calibrate against).")

    if args.folder:
        folders = [Path(args.folder)]
        if not (folders[0] / "embeds.pt").exists():
            raise ValueError(f"{folders[0]} has no embeds.pt; use --root for a directory of folders")
    else:
        folders = find_folders(Path(args.root))
        print(f"Found {len(folders)} decompose output folder(s) under {args.root}")
    if not folders:
        raise ValueError("No decompose.py output folders found.")

    prior, decoder, negative_emb = load_kandinsky()
    sae = load_sae()
    extractors = []
    if args.measure == "inputs":
        extractors.append(ClipFeatures(prior))
        if args.dino:
            extractors.append(DinoFeatures())
    models = (prior, decoder, negative_emb, sae, extractors)

    summaries = []
    for folder in folders:
        res = process_folder(folder, args, models)
        if res is not None:
            summaries.append(res)

    if summaries:
        checked = [s for s in summaries if s["rank_passed"] is not None]
        if checked:
            n_pass = sum(s["rank_passed"] for s in checked)
            print(f"\nRank check: {n_pass}/{len(checked)} passed "
                  f"({len(summaries) - len(checked)} without ordered pairs)")
    if args.root and summaries:
        tag = args.tag or args.weights.replace(",", "-")
        summary_path = Path(args.root) / f"targets_summary_{tag}.json"
        with open(summary_path, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"Summary: {summary_path}")

    warnings = [msg for s in summaries for msg in s["warnings"]]
    if warnings:
        print("\n" + "!" * 70)
        print(f"WARNING: c_max reached in {sum(s['c_max_exceeded'] for s in summaries)} folder(s)")
        for msg in warnings:
            print("  - " + msg)
        print("!" * 70)


if __name__ == "__main__":
    main()
