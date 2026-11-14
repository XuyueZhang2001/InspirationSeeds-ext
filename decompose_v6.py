"""
decompose_v6.py -- adaptive SAE feature grouping with coverage expansion, input-based
merging and weakening of the shared part.
Last modified: 2026-11-12

What changed in v6 (compared with decompose_v5.py)
-------------------------------------------------
Diagnostics with v5 (--debug-images) showed that (1) the main subject that appears in
every input mostly lives in the shared remainder R, not in the group contributions, and
(2) the SAE decoder bias alone decodes to a "mean image" of a brown furry animal, which is
where the unrelated animals came from whenever too much was removed from an embedding.
So the shared part is now weakened as a whole (bias included) instead of being removed
piece by piece, and the subject is kept.

* Shared-part scaling (new, --alpha-r, default 0.7). Every input is
      e_i' = alpha_r * S + (1 + gamma) * D_i,   rescaled to ||e|| (--renorm auto/on/off),
  where S = e - sum_main D_k - beta_minor D_minor - beta_amb D_amb is the part shared by
  all inputs (identical to v5's R when beta = 1). Scaling S as a whole keeps the ratios
  between subject, bias and SAE error, so the subject is kept but weakened and the inputs
  do not drift toward the bias. alpha_r = 1 reproduces v5. With --renorm auto, inputs are
  rescaled to ||e|| whenever alpha_r != 1.
  (Also tried and dropped: per-image automatic alpha, partial removal of the direction
  common to all groups (rho; increased artifacts, since the projection adds negative
  amounts of other groups' features), and subtracting only the unassigned features while
  keeping the bias (moves inputs toward the bias).)
* Merging defaults changed: --merge-mode absolute, --merge-threshold 0.88. Merging
  decisions are made on the v5 construction (alpha_r = 1) so that they do not depend on
  alpha_r.
* groups.json: "shared_weakening" section (alpha_r, renorm, cos(e, bias), split of R into
  unassigned features / bias / SAE error with norms and cosines to e) and cos(e_i, bias)
  per input.

Version history
---------------
* decompose_original.py : Inspiration Seeds. k-means into 2 clusters on decoded SAE
  features, cleaning near the other cluster, inputs e +/- lambda (d_1 - d_0).
* decompose.py           : K clusters (k-means), one-vs-rest direction
  v_i = d_i - mean_{j != i} d_j, inputs e + lambda v_i; saves embeddings and metadata.
* decompose_groups.py v1 : adaptive grouping with FINCH and the ConceptExpress threshold
  (max intra-cluster distance), conditions e_group = Dec(a^(i)) and
  e_anchor = (1 - t) e + t e_group.
* decompose_groups.py v2 : threshold = quantile of pairwise distances, capped at 0.9,
  post-clustering from single features (SAE decoder directions are nearly orthogonal,
  so the max-based threshold merged everything into one group).
* decompose_groups.py v3 : main / minor groups, cleaning of main groups as in the
  original paper, "remove other groups" inputs with beta_minor, beta_amb and gamma;
  t-based conditions removed; steps 50 / size 768; tuned defaults.
* decompose_v4.py        : coverage expansion (other active features assigned to main
  groups by decoder-direction similarity, --assign-threshold).
* decompose_v5.py        : input-based merging of groups; --debug-images diagnostics.
* decompose_v6.py        : this file (weakening of the shared part).

Method (per source image)
-------------------------
  1. e = CLIP(image), a = SAE_enc(e); keep the top-K features by activation.
  2. d_j = j-th column of the SAE decoder (no bias); u_j = d_j / ||d_j||;
     distance D_jk = 1 - u_j . u_k.
  3. Grouping (--delta-mode quantile): delta = quantile of pairwise distances among the
     top-K features, capped at --delta-cap; constrained FINCH merging from single
     features (centroid distance <= delta, mean pairwise distance of a group <= delta).
  4. Roles: main = >= 2 features and salience >= --min-salience (at most --n-max, by
     salience); minor = all other groups.
     salience = sum_j a_j ||d_j|| over the group / sum over the top-K features.
  5. Cleaning: in every main group, remove the --clean-frac fraction of features closest
     to the nearest other main group centroid (ambiguous features).
  6. Coverage expansion: assign the remaining active features to main groups (see above).
  7. Inputs: with D_k = W_dec a^(k) (no bias),
       e_i = e - sum_{k != i, main} D_k - beta_minor D_minor - beta_amb D_amb + gamma D_i.
  8. Merging: merge main groups whose inputs are not separated, recompute.
  9. Weaken the shared part: e_i' = alpha_r S + (1 + gamma) D_i, rescaled to ||e||.
 10. Decode every e_i' with Kandinsky 2.2, same seed for every input.

Usage:
  python decompose_v6.py --image path/to/img.jpeg --output-dir v6_out
  python decompose_v6.py --image-dir hf_samples/sources --output-dir hf_v6
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


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------
def clean_main_groups(U: np.ndarray, main_groups, mode: str, frac: float, margin_thr: float):
    """
    Remove ambiguous features from main groups.
    d_own(j)   = distance to its own group centroid
    d_other(j) = distance to the nearest OTHER main group centroid
    fraction mode: drop int(len * frac) features with the smallest d_other (original paper);
    margin mode  : drop features with 1 - d_own / d_other < margin_thr.
    At least one feature is always kept. Returns (cleaned groups, removed ids, per-feature info).
    """
    C = centroids(U, main_groups)
    cleaned, removed, feat_info = [], [], {}
    for gi, g in enumerate(main_groups):
        others = [k for k in range(len(main_groups)) if k != gi]
        rows = []
        for j in g:
            d_own = float(1 - U[j] @ C[gi])
            d_other = float(min(1 - U[j] @ C[k] for k in others)) if others else float("inf")
            margin = 1 - d_own / d_other if d_other > 0 and np.isfinite(d_other) else 1.0
            rows.append((j, d_own, d_other, margin))
        if mode == "fraction":
            order = sorted(rows, key=lambda x: x[2])  # closest to another group first
            n_drop = min(int(len(g) * frac), len(g) - 1)
            drop = {r[0] for r in order[:n_drop]}
        else:
            drop = {r[0] for r in rows if r[3] < margin_thr}
            if len(drop) >= len(g):  # keep the clearest feature
                drop.discard(max(rows, key=lambda x: x[3])[0])
        for j, d_own, d_other, margin in rows:
            feat_info[j] = {"d_own": d_own, "d_other": d_other if np.isfinite(d_other) else None,
                            "margin": margin, "removed": j in drop}
        cleaned.append([j for j in g if j not in drop])
        removed += sorted(drop)
    return cleaned, removed, feat_info


# --------------------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------------------
def make_overview(cells, out_path: Path, tile=256):
    """cells: list of (image, caption) shown in one row."""
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    gap, cap_h = 6, 34
    W = len(cells) * (tile + gap) + gap
    H = tile + cap_h + 2 * gap
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    for c, (img, cap) in enumerate(cells):
        x = gap + c * (tile + gap)
        canvas.paste(img.convert("RGB").resize((tile, tile)), (x, gap))
        draw.multiline_text((x + 2, gap + tile + 2), cap, fill="black", font=font, spacing=1)
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
    Dfeat = sae.decoder.weight[:, top].T.float()  # (K, 1280)
    d_norm = Dfeat.norm(dim=1)
    U = normalize_rows(Dfeat.cpu().numpy().astype(np.float64))

    # 3. grouping
    groups, levels, ginfo = adaptive_groups(U, args.n_max, args.delta_fallback, args.chain_factor,
                                            args.delta_mode, args.delta_quantile,
                                            None if args.delta_cap <= 0 else args.delta_cap)

    # 4. roles
    w = (a_top * d_norm).cpu().numpy()  # a_j ||d_j||
    total_w = float(w.sum())
    sal = lambda g: float(w[list(g)].sum() / total_w) if total_w > 0 and len(g) else 0.0
    groups = sorted(groups, key=lambda g: -sal(g))
    eligible = [g for g in groups if len(g) >= 2 and sal(g) >= args.min_salience]
    main = eligible[: args.n_max]
    main_ids = {id(g) for g in main}
    minor = [g for g in groups if id(g) not in main_ids]

    # 5. cleaning
    if len(main) >= 2 and args.clean_frac > 0 or (len(main) >= 2 and args.clean_mode == "margin"):
        main_clean, ambiguous, feat_info = clean_main_groups(U, main, args.clean_mode, args.clean_frac,
                                                             args.clean_margin)
    else:
        main_clean, ambiguous, feat_info = [list(g) for g in main], [], {}
    decomposable = len(main_clean) >= 2

    # 6. contributions and conditions
    def contribution(ids):
        if not ids:
            return torch.zeros_like(e)
        idx = torch.tensor(list(ids), device=Dfeat.device)
        return (Dfeat[idx] * a_top[idx, None]).sum(0)

    D_core = [contribution(g) for g in main_clean]
    minor_ids = [j for g in minor for j in g]
    D_minor = contribution(minor_ids)
    D_amb = contribution(ambiguous)

    # 6. coverage expansion: assign the other active features to main groups
    active = torch.nonzero(acts > 0).squeeze(1)
    top_t = torch.tensor(top, device=acts.device)
    rest = active[~torch.isin(active, top_t)]
    assigned = [[] for _ in main_clean]
    D_assigned = [torch.zeros_like(e) for _ in main_clean]
    assign_sims = [[] for _ in main_clean]
    if main_clean and len(rest) > 0 and args.assign_threshold < 1:
        C = torch.tensor(centroids(U, main_clean), device=acts.device, dtype=torch.float32)  # (M, 1280)
        R_dirs = sae.decoder.weight[:, rest].T.float()                                        # (n_rest, 1280)
        sims = torch.nn.functional.normalize(R_dirs, dim=1) @ C.T                             # (n_rest, M)
        best_sim, best = sims.max(1)
        keep = best_sim >= args.assign_threshold
        a_rest = acts[rest].float()
        for gi in range(len(main_clean)):
            m = keep & (best == gi)
            if m.any():
                assigned[gi] = rest[m].tolist()
                assign_sims[gi] = best_sim[m].tolist()
                D_assigned[gi] = (R_dirs[m] * a_rest[m, None]).sum(0)
        R_feat = (R_dirs[~keep] * a_rest[~keep, None]).sum(0) if (~keep).any() else torch.zeros_like(e)
    elif len(rest) > 0:
        R_feat = (sae.decoder.weight[:, rest].T.float() * acts[rest].float()[:, None]).sum(0)
    else:
        R_feat = torch.zeros_like(e)
    D_top = contribution(list(range(len(top))))
    R = e - D_top - sum(D_assigned, torch.zeros_like(e))
    n_main_before_merge = len(main_clean)

    def build_inputs(Dm):
        res = []
        for i in range(len(Dm)):
            others = sum((Dm[k] for k in range(len(Dm)) if k != i), torch.zeros_like(e))
            res.append(e - others - args.beta_minor * D_minor - args.beta_amb * D_amb + args.gamma * Dm[i])
        return res

    # 7. merge groups whose inputs are not separated
    merge_log = []
    while args.merge_mode != "off" and len(main_clean) > max(args.n_min, 1):
        Dm = [D_core[i] + D_assigned[i] for i in range(len(main_clean))]
        E_in = build_inputs(Dm)
        En = torch.nn.functional.normalize(torch.stack(E_in), dim=1)
        S = (En @ En.T).cpu().numpy()
        c2e = [float(torch.nn.functional.cosine_similarity(x, e, dim=0)) for x in E_in]
        cands = []
        for i in range(len(E_in)):
            for j in range(i + 1, len(E_in)):
                if args.merge_mode == "relative":
                    hit = S[i, j] >= min(c2e[i], c2e[j]) - args.merge_margin
                else:
                    hit = S[i, j] >= args.merge_threshold
                if hit:
                    cands.append((float(S[i, j]), i, j))
        if not cands:
            break
        sim_ij, i, j = max(cands)
        merge_log.append({
            "merged": [i, j],
            "input_cosine": sim_ij,
            "cos_to_e": [c2e[i], c2e[j]],
            "sizes": [len(main_clean[i]), len(main_clean[j])],
            "n_groups_before": len(main_clean),
        })
        main[i] = main[i] + main[j]
        main_clean[i] = main_clean[i] + main_clean[j]
        assigned[i] = assigned[i] + assigned[j]
        assign_sims[i] = assign_sims[i] + assign_sims[j]
        D_core[i] = D_core[i] + D_core[j]
        D_assigned[i] = D_assigned[i] + D_assigned[j]
        for lst in (main, main_clean, assigned, assign_sims, D_core, D_assigned):
            del lst[j]
        # keep groups ordered by salience
        order = sorted(range(len(main)), key=lambda k: -sal(main[k]))
        main, main_clean, assigned, assign_sims, D_core, D_assigned = (
            [lst[k] for k in order] for lst in (main, main_clean, assigned, assign_sims, D_core, D_assigned))
    decomposable = len(main_clean) >= 2
    D_main = [D_core[i] + D_assigned[i] for i in range(len(main_clean))]

    # 9. weaken the shared part (v6)
    b_vec = sae.decode(torch.zeros_like(acts)).float()          # SAE decoder bias ("mean image")
    full_rec = sae.decode(acts).float()                         # SAE reconstruction of e (with bias)
    err_vec = e - full_rec                                      # SAE reconstruction error
    cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x, y, dim=0))
    # shared part: identical for every input before weakening
    S_shared = e - sum(D_main, torch.zeros_like(e)) - args.beta_minor * D_minor - args.beta_amb * D_amb
    own = [(1 + args.gamma) * D for D in D_main]
    renorm = (args.renorm == "on") or (args.renorm == "auto" and args.alpha_r != 1)
    alpha_used = args.alpha_r
    cos_e_bias = cos(e, b_vec)
    e_inputs_final = []
    for o in own:
        x = alpha_used * S_shared + o
        if renorm:
            x = x * (e.norm() / x.norm().clamp_min(1e-8))
        e_inputs_final.append(x)

    e_inputs, cond = [], []
    for i in range(len(main_clean)):
        others = sum((D_main[k] for k in range(len(main_clean)) if k != i), torch.zeros_like(e))
        ei = e_inputs_final[i]
        e_inputs.append(ei)
        cond.append({
            "group": i,
            "norm_over_e": float(ei.norm() / e.norm()),
            "cos_to_e": float(torch.nn.functional.cosine_similarity(ei, e, dim=0)),
            "norm_removed_over_e": float(others.norm() / e.norm()),
            "norm_D_core_over_e": float(D_core[i].norm() / e.norm()),
            "norm_D_expanded_over_e": float(D_main[i].norm() / e.norm()),
            "n_assigned": len(assigned[i]),
            "cos_to_bias": cos(ei, b_vec),
        })
    if e_inputs:
        En = torch.nn.functional.normalize(torch.stack(e_inputs), dim=1)
        input_cos = (En @ En.T).tolist()
    else:
        input_cos = []

    # generation
    recon = generate(e_half, decoder, negative_emb, args.seed, args.steps, args.size)
    recon.save(out / "recon.jpeg")
    cells = [(source, "source"), (recon, "recon")]
    if decomposable:
        for i, ei in enumerate(e_inputs):
            img = generate(ei.to(e_half.dtype), decoder, negative_emb, args.seed, args.steps, args.size)
            img.save(out / f"input_{i}.jpeg")
            cells.append((img, f"g{i} n={len(main_clean[i])}/{len(main[i])} +{len(assigned[i])}\nsal={sal(main[i]):.2f}"))
    make_overview(cells, out / "overview.jpeg")

    # diagnostic images
    debug = None
    if args.debug_images:
        R_in = e - sum(D_main, torch.zeros_like(e)) - args.beta_minor * D_minor - args.beta_amb * D_amb
        img_R = generate(R_in.to(e_half.dtype), decoder, negative_emb, args.seed, args.steps, args.size)
        img_R.save(out / "debug_R.jpeg")
        img_b = generate(b_vec.to(e_half.dtype), decoder, negative_emb, args.seed, args.steps, args.size)
        img_b.save(out / "debug_bias.jpeg")
        dcells = [(source, "source"), (img_R, "R only\n(shared part)"), (img_b, "SAE bias only")]
        debug = {
            "norm_R_input_over_e": float(R_in.norm() / e.norm()),
            "cos_R_input_e": float(torch.nn.functional.cosine_similarity(R_in, e, dim=0)),
            "cos_bias_e": float(torch.nn.functional.cosine_similarity(b_vec, e, dim=0)),
            "groups": [],
        }
        if decomposable:
            Dn = torch.nn.functional.normalize(torch.stack(D_main), dim=1)
            c_dir = torch.nn.functional.normalize(Dn.mean(0), dim=0)
            for i, ei in enumerate(e_inputs):
                s_i = (D_main[i] @ c_dir) * c_dir
                e_u = ei - (1 + args.gamma) * s_i
                img_u = generate(e_u.to(e_half.dtype), decoder, negative_emb, args.seed, args.steps, args.size)
                img_u.save(out / f"input_{i}_unique.jpeg")
                cos_dc = float(torch.nn.functional.cosine_similarity(D_main[i], c_dir, dim=0))
                share = float(s_i.norm() / D_main[i].norm().clamp_min(1e-8))
                dcells.append((img_u, f"g{i} unique\ncos(D,c)={cos_dc:.2f}"))
                debug["groups"].append({"group": i, "cos_D_c": cos_dc, "shared_norm_over_D": share,
                                        "cos_unique_input_e": float(torch.nn.functional.cosine_similarity(e_u, e, dim=0))})
            if len(e_inputs) > 1:
                Du = torch.stack([D_main[i] - (D_main[i] @ c_dir) * c_dir for i in range(len(D_main))])
                Dun = torch.nn.functional.normalize(Du, dim=1)
                debug["unique_contribution_cosine"] = (Dun @ Dun.T).tolist()
                debug["contribution_cosine"] = (Dn @ Dn.T).tolist()
        make_overview(dcells, out / "debug_overview.jpeg")

    # save
    torch.save({
        "e": e.cpu(),
        "top_features": top,
        "main_groups": [[top[j] for j in g] for g in main_clean],
        "D_main": torch.stack(D_main).cpu() if D_main else None,
        "D_core": torch.stack(D_core).cpu() if D_core else None,
        "assigned_features": assigned,
        "R": R.cpu(),
        "D_minor": D_minor.cpu(),
        "D_amb": D_amb.cpu(),
        "e_inputs": torch.stack(e_inputs).cpu() if e_inputs else None,
    }, out / "embeds.pt")

    def describe(g, g_clean=None):
        g_clean = g if g_clean is None else g_clean
        P = U[g_clean]
        coh = float((P @ P.T)[np.triu_indices(len(g_clean), 1)].mean()) if len(g_clean) > 1 else None
        return {
            "feature_ids": [top[j] for j in g_clean],
            "activations": [float(a_top[j]) for j in g_clean],
            "size_before_cleaning": len(g),
            "size_after_cleaning": len(g_clean),
            "salience_before_cleaning": sal(g),
            "salience_after_cleaning": sal(g_clean),
            "coherence": coh,
            "spread": spread(U, g_clean),
        }

    meta = {
        "image": str(image_path),
        "top_k": args.top_k,
        "n_max": args.n_max,
        "min_salience": args.min_salience,
        "delta_quantile": args.delta_quantile if args.delta_mode == "quantile" else None,
        "delta_cap": args.delta_cap,
        "chain_factor": args.chain_factor,
        "clean_mode": args.clean_mode,
        "clean_frac": args.clean_frac,
        "clean_margin": args.clean_margin if args.clean_mode == "margin" else None,
        "gamma": args.gamma,
        "assign_threshold": args.assign_threshold,
        "beta_minor": args.beta_minor,
        "beta_amb": args.beta_amb,
        "seed": args.seed,
        "steps": args.steps,
        "size": args.size,
        **ginfo,
        "n_groups_total": len(groups),
        "n_main": len(main_clean),
        "n_minor": len(minor),
        "n_ambiguous": len(ambiguous),
        "decomposable": decomposable,
        "main_groups": [describe(g, gc) for g, gc in zip(main, main_clean)],
        "minor_groups": [describe(g) for g in minor],
        "ambiguous_features": [top[j] for j in ambiguous],
        "feature_cleaning": {str(top[j]): v for j, v in feat_info.items()},
        "salience_minor_total": sal(minor_ids),
        "salience_ambiguous_total": sal(ambiguous),
        "n_active_features": int(len(active)),
        "n_rest_features": int(len(rest)),
        "n_assigned_total": int(sum(len(a) for a in assigned)),
        "merge_mode": args.merge_mode,
        "merge_threshold": args.merge_threshold if args.merge_mode == "absolute" else None,
        "merge_margin": args.merge_margin if args.merge_mode == "relative" else None,
        "n_min": args.n_min,
        "n_main_before_merge": n_main_before_merge,
        "merges": merge_log,
        "assigned_per_group": [len(a) for a in assigned],
        "assigned_mean_similarity": [float(np.mean(s)) if s else None for s in assign_sims],
        "assigned_feature_ids": assigned,
        "norm_R_over_e": float(R.norm() / e.norm()),
        "norm_top_k_over_e": float(D_top.norm() / e.norm()),
        "conditions": cond,
        "shared_weakening": {
            "alpha_r_requested": args.alpha_r,
            "alpha_r_used": alpha_used,
            "renorm": renorm,
            "cos_e_bias": cos_e_bias,
            "norm_shared_over_e": float(S_shared.norm() / e.norm()),
            "R_split": {
                "unassigned_features": {"norm_over_e": float(R_feat.norm() / e.norm()), "cos_e": cos(R_feat, e)},
                "bias": {"norm_over_e": float(b_vec.norm() / e.norm()), "cos_e": cos_e_bias},
                "sae_error": {"norm_over_e": float(err_vec.norm() / e.norm()), "cos_e": cos(err_vec, e)},
            },
        },
        "debug": debug,
        "input_embedding_cosine": input_cos,
        "top_features": [{"feature_id": f, "activation": float(a)} for f, a in zip(top, a_top.tolist())],
        "finch_levels": [[[top[j] for j in c] for c in lv] for lv in levels],
    }
    with open(out / "groups.json", "w") as f:
        json.dump(meta, f, indent=2)

    if len(input_cos) > 1:
        S = np.array(input_cos)
        pair_mean = float(S[np.triu_indices(len(S), 1)].mean())
    else:
        pair_mean = None
    c2e = float(np.mean([c["cos_to_e"] for c in cond])) if cond else None
    print(f"{image_path.name}: delta={ginfo['delta']:.3f}, main={len(main_clean)} (before merge {n_main_before_merge}) "
          f"(sizes {[len(g) for g in main_clean]}, +assigned {[len(a) for a in assigned]} of {len(rest)} other active), "
          f"||R||/||e||={float(R.norm() / e.norm()):.2f}, alpha_r={alpha_used:.2f}"
          + (f", cos_to_e={c2e:.2f}, input-input={pair_mean:.2f}" if pair_mean is not None else "")
          + ("" if decomposable else " [not decomposable: < 2 main groups]"))
    return {"image": str(image_path), "folder": str(out), "n_main": len(main_clean), "n_minor": len(minor),
            "n_main_before_merge": n_main_before_merge, "n_merges": len(merge_log),
            "assigned_per_group": [len(a) for a in assigned], "norm_R_over_e": float(R.norm() / e.norm()),
            "mean_cos_to_e": c2e, "mean_input_input_cos": pair_mean, "alpha_r_used": alpha_used,
            "max_cos_to_bias": max((c["cos_to_bias"] for c in cond), default=None), "cos_e_bias": cos_e_bias,
            "n_ambiguous": len(ambiguous), "decomposable": decomposable,
            "main_sizes": [len(g) for g in main_clean],
            "main_saliences": [sal(g) for g in main]}


# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="decompose_v6: adaptive SAE grouping with shared-part weakening")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="One source image")
    src.add_argument("--image-dir", help="Folder of source images")
    p.add_argument("--output-dir", default="groups_out")
    p.add_argument("--top-k", type=int, default=64, help="Number of top-activated SAE features (K)")
    p.add_argument("--n-max", type=int, default=4, help="Maximum number of main groups")
    p.add_argument("--min-salience", type=float, default=0.1,
                   help="A group needs at least this contribution share (and >= 2 features) to be main")
    # grouping
    p.add_argument("--delta-mode", choices=["quantile", "max"], default="quantile")
    p.add_argument("--delta-quantile", type=float, default=0.2)
    p.add_argument("--delta-cap", type=float, default=0.9, help="Upper limit on delta in quantile mode (0 = none)")
    p.add_argument("--delta-fallback", type=float, default=None)
    p.add_argument("--chain-factor", type=float, default=1.0)
    # cleaning
    p.add_argument("--clean-mode", choices=["fraction", "margin"], default="fraction")
    p.add_argument("--clean-frac", type=float, default=0.2,
                   help="Fraction of each main group removed (closest to another main group); 0 = no cleaning")
    p.add_argument("--clean-margin", type=float, default=0.1, help="Margin threshold for --clean-mode margin")
    # construction
    p.add_argument("--merge-mode", choices=["relative", "absolute", "off"], default="absolute",
                   help="relative: merge two groups if their inputs are more similar to each other than to the source; "
                        "absolute: merge if input cosine >= --merge-threshold; off: no merging (= v4)")
    p.add_argument("--merge-threshold", type=float, default=0.88, help="Input cosine threshold for --merge-mode absolute")
    p.add_argument("--merge-margin", type=float, default=0.0,
                   help="Relative mode: merge if cos(e_i, e_j) >= min(cos(e_i, e), cos(e_j, e)) - margin")
    p.add_argument("--n-min", type=int, default=2, help="Merging never goes below this number of main groups")
    p.add_argument("--assign-threshold", type=float, default=0.2,
                   help="Assign other active features to the most similar main group if cosine >= this (1 = off)")
    p.add_argument("--gamma", type=float, default=1.0, help="Amplification of the group's own contribution")
    p.add_argument("--beta-minor", type=float, default=1.0, help="How much of the minor groups to remove (0..1)")
    p.add_argument("--beta-amb", type=float, default=1.0, help="How much of the ambiguous features to remove (0..1)")
    # generation
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=50, help="Kandinsky inference steps (decompose.py uses 18)")
    p.add_argument("--size", type=int, default=768, help="Kandinsky output size (decompose.py uses 512)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--alpha-r", type=float, default=0.7,
                   help="Fraction of the shared part kept in every input (default 0.7; 1 = v5 behaviour)")
    p.add_argument("--renorm", choices=["auto", "on", "off"], default="auto",
                   help="Rescale every input to ||e||; auto = only when alpha_r != 1")
    p.add_argument("--debug-images", action="store_true",
                   help="Also decode R alone, the SAE bias alone, and each input without the shared group component")
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
        root = Path(args.output_dir)
        with open(root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        M = [s["n_main"] for s in summary]
        B = [s["n_main_before_merge"] for s in summary]
        print(f"\n{len(summary)} image(s). main groups before merging: "
              + ", ".join(f"{m}: {B.count(m)}" for m in sorted(set(B))))
        print(f"main groups per image (after merging): "
              + ", ".join(f"{m}: {M.count(m)}" for m in sorted(set(M))))
        print(f"not decomposable (< 2 main groups): {sum(not s['decomposable'] for s in summary)}")
        ok = [s for s in summary if s["mean_input_input_cos"] is not None]
        if ok:
            sep = sum(s["mean_input_input_cos"] < s["mean_cos_to_e"] for s in ok)
            print("alpha_r used: " + ", ".join(f"{a}: {[s['alpha_r_used'] for s in ok].count(a)}"
                                               for a in sorted({s['alpha_r_used'] for s in ok})))
            print(f"mean ||R||/||e|| = {np.mean([s['norm_R_over_e'] for s in ok]):.2f}; "
                  f"input-input < cos_to_e in {sep}/{len(ok)} images")
        print(f"summary: {root / 'summary.json'}")


if __name__ == "__main__":
    main()
