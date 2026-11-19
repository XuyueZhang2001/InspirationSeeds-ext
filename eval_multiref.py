"""
eval_multiref.py -- designation gap and first-position bias of a train_multiref.py checkpoint.
Last modified: 2026-11-19 (role mode "tid")

Question answered: when the model has learned to follow a designated main image, does it follow
the DESIGNATION, or has it learned "the first input is the main image" regardless of the prompt?
With --role slot the main image is always moved to the first position during training, so the
second behaviour is a real risk. With --role tid the main image keeps its position (marked by
t = 2) and no position is special without a designation, so the bias is expected to stay near
the base model's value; prompt_effect is not meaningful there (order and marker both change).

Metrics (designated samples; per sample with main k, target_k, the same noise and sigma for
every condition; references shuffled once with the same seed as in training)
    designated, input j  : designated as main by the training role mode
                           (slot: input j moved first + "first image" prompt;
                            tid: input j gets t = main_t, order unchanged)
    base, input j first  : base prompt (no main image), input j moved first, no E_main

    gap            = mean_{j != k} L(designated j)   - L(designated k)
                     the training metric; > 0: the model uses the designation
    bias           = mean_{j != k} L(base, j first)  - L(base, k first)
                     > 0: the model favours target_k when k is FIRST, although no main image is
                     designated, i.e. the first position itself acts as the main image
    prompt_effect  = L(base, k first) - L(designated k)
                     slot only: same order, only the prompt differs -> what the prompt adds
    gap_acc / bias_acc : share of comparisons where the k condition has the lowest loss
                     (chance: 1 / number of inputs)
    bias_ratio     = bias / gap

    Reading (heuristic, look at the high-sigma values first):
        gap small (< 0.01)        designation not learned yet; bias cannot be interpreted
        bias_ratio < 0.2          designation conditional on the prompt (wanted behaviour)
        bias_ratio > 0.7          "first input = main image" even without a designation
        in between                partly conditional, partly a position preference

    With the same samples, sigmas and noise draws as the training run, the gap printed here
    equals the gap logged at the checkpoint's step (sanity check of the evaluation).

Rotation images (--image-folders > 0)
    For a few designated folders: inputs and targets, then for every seed one row
    "no main, input i first" (base prompt, input i moved first, the others in their original
    order) and one row "main = input i" (designated by the role mode). If the first row
    changes with the first input in the same way as the second row, the model uses the first
    position as the main image.
    -> <output-dir>/rotation_<folder>.jpeg

Output (default <checkpoint>/eval)
    summary.json     metrics overall and per sigma, settings
    per_sample.csv   one line per sample, sigma and noise draw
    rotation_*.jpeg  rotation images

Usage
-----
    # checkpoint of an overfitting run (evaluated on the same overfit subset)
    python eval_multiref.py --checkpoint runs/overfit_slot_hs_long/checkpoint-005000

    # base model with the run's settings (no LoRA), for comparison
    python eval_multiref.py --checkpoint runs/overfit_slot_hs_long/checkpoint-005000 --base-only \\
        --output-dir runs/overfit_slot_hs_long/eval_base

    # validation split of a normal run, more samples
    python eval_multiref.py --checkpoint runs/v0_slot_hs/checkpoint-003000 --split val --max-samples 16

Settings (role, layout, LoRA, data, overfit subset, seed) are read from <run>/args.json, the
file train_multiref.py writes next to the checkpoints.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from PIL import Image, ImageDraw, ImageFont

from multiref.conditioning import MAIN_T, MainRoleEmbedding, MultiRefKontextPipeline
from multiref.prompts import arrange, prompt_key
from multiref.train_data import LatentStore, load_rows, make_example, pick_overfit_rows
from train_multiref import DEVICE, DTYPE, build_transformer, forward_loss

# fields of the training args that are paths
PATH_ARGS = ("data_dir", "cache_dir", "output_dir", "resume")


# --------------------------------------------------------------------------------------
# Arguments and training settings
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Designation gap and first-position bias of a multiref checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True, help="checkpoint-XXXXXX folder of a train_multiref.py run")
    p.add_argument("--run-args", type=Path, default=None, help="args.json of the run (default: <checkpoint>/../args.json)")
    p.add_argument("--base-only", action="store_true", help="Evaluate the base model (no LoRA, no E_main)")
    p.add_argument("--split", choices=["auto", "train", "val", "overfit"], default="auto",
                   help="Samples: auto = the overfit subset for an overfitting run, else the validation split")
    p.add_argument("--data-dir", type=Path, default=None, help="Override the run's --data-dir")
    p.add_argument("--cache-dir", type=Path, default=None, help="Override the run's cache directory")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Designated samples (default: the run's --gap-samples, else 8)")
    p.add_argument("--sigmas", default=None, help="Sigmas (default: the run's --gap-sigmas, else 0.9,0.75,0.5,0.25)")
    p.add_argument("--noises", type=int, default=None, help="Noise draws per sigma (default: the run's --gap-noises, else 1)")
    p.add_argument("--image-folders", type=int, default=2, help="Designated folders with rotation images (0 = none)")
    p.add_argument("--image-seeds", type=int, default=2, help="Seeds per rotation image")
    p.add_argument("--image-steps", type=int, default=28, help="Denoising steps of the rotation images")
    p.add_argument("--output-dir", type=Path, default=None, help="Default: <checkpoint>/eval")
    return p.parse_args()


def load_run_args(args):
    """Training settings of the run as a namespace (paths restored, eval overrides applied)."""
    path = args.run_args or args.checkpoint.parent / "args.json"
    if path.exists():
        with open(path) as f:
            run = json.load(f)
    else:
        state = torch.load(args.checkpoint / "train_state.pt", map_location="cpu", weights_only=False)
        run = state["args"]
        print(f"[note] {path} not found, settings taken from train_state.pt")
    for key in PATH_ARGS:
        if run.get(key) not in (None, "None"):
            run[key] = Path(run[key])
        else:
            run[key] = None
    if args.data_dir is not None:
        run["data_dir"] = args.data_dir
    if args.cache_dir is not None:
        run["cache_dir"] = args.cache_dir
    run["init_lora"] = None if args.base_only else str(args.checkpoint / "lora")
    run.setdefault("no_shuffle", False)
    return argparse.Namespace(**run)


def select_rows(run, split):
    if split == "auto":
        split = "overfit" if run.overfit else "val"
    if split == "overfit":
        if not run.overfit:
            raise ValueError("--split overfit needs a run trained with --overfit")
        return pick_overfit_rows(load_rows(run.data_dir, "train"), run.overfit, seed=run.seed), "overfit"
    rows = load_rows(run.data_dir, split)
    if not rows:
        raise ValueError(f"No samples in {run.data_dir}/{split}.jsonl")
    return rows, split


def designated_rows(rows, n_max):
    """Whole designated folders first, most mains first (same selection as the training gap)."""
    by_folder = defaultdict(list)
    for r in rows:
        if r["pool"] == "designated":
            by_folder[r["folder"]].append(r)
    picked = []
    for folder in sorted(by_folder, key=lambda f: (-len(by_folder[f]), f)):
        picked += sorted(by_folder[folder], key=lambda r: r["main"])
        if len(picked) >= n_max:
            break
    return picked[:n_max], by_folder


# --------------------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------------------
def first(order, j):
    """Order with input j moved to the front, the others unchanged."""
    return [j] + [i for i in order if i != j]


def base_example(row, target, store, order):
    """Base prompt, references in the given order, no main image."""
    return {"refs": [store.ref(row["inputs"][i]) for i in order], "target": store.target(target),
            "main_pos": None, "prompt_key": "base", "prompt": store.prompt("base"),
            "pool": row["pool"], "id": row["id"], "order": order}


def build_items(picked, store, run, sigmas, n_noises):
    """Per sample: designated and base-first examples for every input, and the noise draws.

    Shuffle seed 30000 + i and noise seed 40000 + i as in train_multiref.designation_gap_set, so
    the gap equals the gap logged during training for the same samples.
    """
    items = []
    for i, row in enumerate(picked):
        target = row["targets"][0]
        n = len(row["inputs"])
        shuffle = not run.no_shuffle
        designated = {j: make_example(dict(row, main=j), target, store, run.role, random.Random(30_000 + i), shuffle)
                      for j in range(n)}
        base_order, _, _ = arrange(n, None, run.role, shuffle=shuffle, rng=random.Random(30_000 + i))
        base_first = {j: base_example(row, target, store, first(base_order, j)) for j in range(n)}
        gen = torch.Generator().manual_seed(40_000 + i)
        shape = designated[row["main"]]["target"].unsqueeze(0).shape
        noises = {s: [torch.randn(shape, generator=gen) for _ in range(n_noises)] for s in sigmas}
        items.append({"row": row, "k": row["main"], "designated": designated, "base_first": base_first,
                      "noises": noises})
    return items


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(transformer, role, items, run):
    transformer.eval()
    records = []
    for it in items:
        k, row = it["k"], it["row"]
        others = [j for j in it["designated"] if j != k]
        for sig, noises in it["noises"].items():
            s = torch.tensor([sig])
            for ni, noise in enumerate(noises):
                d = {j: forward_loss(transformer, role, ex, s, noise, run).item() for j, ex in it["designated"].items()}
                b = {j: forward_loss(transformer, role, ex, s, noise, run).item() for j, ex in it["base_first"].items()}
                d_other = sum(d[j] for j in others) / len(others)
                b_other = sum(b[j] for j in others) / len(others)
                records.append({
                    "id": row["id"], "folder": Path(row["folder"]).name, "main": k, "n_inputs": len(d),
                    "sigma": sig, "noise": ni,
                    "L_designated_k": d[k], "L_designated_others": d_other,
                    "L_base_k_first": b[k], "L_base_others_first": b_other,
                    "gap": d_other - d[k], "gap_hit": float(d[k] < min(d[j] for j in others)),
                    "bias": b_other - b[k], "bias_hit": float(b[k] < min(b[j] for j in others)),
                    "prompt_effect": b[k] - d[k],
                })
        print(f"  {row['id']}: done")
    return records


def summarize(records):
    def agg(rs):
        mean = lambda key: sum(r[key] for r in rs) / len(rs)
        out = {"gap": mean("gap"), "gap_acc": mean("gap_hit"), "bias": mean("bias"), "bias_acc": mean("bias_hit"),
               "prompt_effect": mean("prompt_effect"), "n": len(rs)}
        out["bias_ratio"] = out["bias"] / out["gap"] if abs(out["gap"]) > 1e-4 else None
        return out

    by_sigma = defaultdict(list)
    for r in records:
        by_sigma[r["sigma"]].append(r)
    chance = sum(1 / r["n_inputs"] for r in records) / len(records)
    return {"overall": agg(records), "per_sigma": {f"{s:g}": agg(rs) for s, rs in sorted(by_sigma.items(), reverse=True)},
            "chance_acc": chance}


def verdict(summary):
    hi = summary["per_sigma"][next(iter(summary["per_sigma"]))]  # highest sigma
    gap, ratio = hi["gap"], hi["bias_ratio"]
    if gap < 0.01:
        return "designation hardly learned at high sigma (gap < 0.01): the bias cannot be interpreted yet"
    if ratio is None or ratio < 0.2:
        return "designation is conditional on the prompt: little preference for the first input without a main image"
    if ratio > 0.7:
        return "the first input acts as the main image even without a designation (position shortcut)"
    return "partly conditional: the first input is favoured without a designation, but less than with one"


# --------------------------------------------------------------------------------------
# Rotation images
# --------------------------------------------------------------------------------------
@torch.no_grad()
def rotation_images(pipe, role, by_folder, folders, store, run, args, out_dir):
    pipe.transformer.eval()
    size = store.target_size
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    paths = []
    for folder in folders:
        rows = sorted(by_folder[folder], key=lambda r: r["main"])
        inputs = rows[0]["inputs"]
        n = len(inputs)
        mains = {r["main"]: r for r in rows}

        def generate(order, main_pos, key, seed):
            refs = [store.ref(inputs[i]).unsqueeze(0).to(DEVICE, DTYPE) for i in order]
            pr = store.prompt(key)
            with torch.autocast("cuda", dtype=DTYPE):
                out = pipe.generate_from_latents(
                    refs, main_pos, run.role, run.layout, role, height=size, width=size,
                    main_t=getattr(run, "main_t", MAIN_T), prompt_embeds=pr["prompt_embeds"].unsqueeze(0).to(DEVICE, DTYPE),
                    pooled_prompt_embeds=pr["pooled_prompt_embeds"].unsqueeze(0).to(DEVICE, DTYPE),
                    num_inference_steps=args.image_steps, guidance_scale=run.guidance_scale,
                    generator=torch.Generator(device=DEVICE).manual_seed(seed))
            return out.images[0]

        grid = [[(Image.open(p).convert("RGB"), f"input_{i}") for i, p in enumerate(inputs)]]
        grid.append([(Image.open(mains[i]["targets"][0]).convert("RGB"), f"target, main = input_{i}")
                     if i in mains else None for i in range(n)])
        for s in range(args.image_seeds):
            seed = run.seed + s
            grid.append([(generate(first(list(range(n)), i), None, "base", seed),
                          f"no main, input_{i} first (seed {seed})") for i in range(n)])
            row = []
            for i in range(n):
                order, main_pos, _ = arrange(n, i, run.role, shuffle=False)
                row.append((generate(order, main_pos, prompt_key(main_pos, run.role), seed),
                            f"main = input_{i} (seed {seed})"))
            grid.append(row)

        tile, gap, cap = 256, 6, 20
        W = gap + n * (tile + gap)
        H = 30 + len(grid) * (tile + cap + gap)
        canvas = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((gap, 8), f"{Path(folder).name}: rows = inputs, targets, then per seed: no main (rotated) / designated",
                  fill="black", font=font)
        y = 30
        for cells in grid:
            for c, cell in enumerate(cells):
                if cell is None:
                    continue
                img, text = cell
                x = gap + c * (tile + gap)
                canvas.paste(img.resize((tile, tile)), (x, y))
                draw.text((x + 2, y + tile + 2), text, fill="black", font=font)
            y += tile + cap + gap
        path = out_dir / f"rotation_{Path(folder).name}.jpeg"
        canvas.save(path, quality=92)
        paths.append(path)
        print(f"  rotation images: {path}")
    return paths


# --------------------------------------------------------------------------------------
def main():
    args = parse_args()
    run = load_run_args(args)
    out_dir = args.output_dir or args.checkpoint / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    sigmas = [float(x) for x in (args.sigmas or getattr(run, "gap_sigmas", None) or "0.9,0.75,0.5,0.25").split(",")
              if x.strip()]
    n_noises = args.noises or getattr(run, "gap_noises", None) or 1
    n_samples = args.max_samples or getattr(run, "gap_samples", None) or 8

    rows, split = select_rows(run, args.split)
    picked, by_folder = designated_rows(rows, n_samples)
    if not picked:
        raise ValueError(f"No designated samples in the {split} split")
    store = LatentStore(run.cache_dir or run.data_dir / "cache")
    print(f"checkpoint {args.checkpoint} ({'base model only' if args.base_only else 'LoRA'}), role {run.role}, "
          f"split {split}: {len(picked)} designated samples, sigmas {sigmas}, {n_noises} noise draw(s)")

    transformer = build_transformer(run)
    role = None
    if run.role == "embed":
        role = MainRoleEmbedding(transformer.inner_dim).to(DEVICE).attach(transformer)
        emb = args.checkpoint / "role_embedding.safetensors"
        if not args.base_only and emb.exists():
            role.load(emb)
    elif not args.base_only:
        print(f"role {run.role}: no role embedding")

    items = build_items(picked, store, run, sigmas, n_noises)
    records = evaluate(transformer, role, items, run)
    summary = summarize(records)

    with open(out_dir / "per_sample.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"\n{'sigma':>7} {'gap':>9} {'gap_acc':>8} {'bias':>9} {'bias_acc':>9} {'prompt':>9} {'bias/gap':>9}")
    for name, m in list(summary["per_sigma"].items()) + [("all", summary["overall"])]:
        ratio = f"{m['bias_ratio']:.2f}" if m["bias_ratio"] is not None else "-"
        print(f"{name:>7} {m['gap']:9.4f} {m['gap_acc']:8.2f} {m['bias']:9.4f} {m['bias_acc']:9.2f} "
              f"{m['prompt_effect']:9.4f} {ratio:>9}")
    print(f"chance accuracy {summary['chance_acc']:.2f}")
    summary["verdict"] = verdict(summary)
    print(f"verdict (heuristic, highest sigma): {summary['verdict']}")

    images = []
    if args.image_folders > 0:
        folders = list(dict.fromkeys(r["folder"] for r in picked))[: args.image_folders]
        vae = AutoencoderKL.from_pretrained(run.base_model, subfolder="vae", torch_dtype=DTYPE).to(DEVICE)
        pipe = MultiRefKontextPipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(run.base_model, subfolder="scheduler"),
            vae=vae, text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None, transformer=transformer)
        pipe.set_progress_bar_config(disable=True)
        images = rotation_images(pipe, role, by_folder, folders, store, run, args, out_dir)

    summary.update({
        "checkpoint": str(args.checkpoint), "base_only": args.base_only, "role": run.role, "layout": run.layout,
        "split": split, "samples": [r["id"] for r in picked], "sigmas": sigmas, "noises": n_noises,
        "rotation_images": [str(p) for p in images],
    })
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
