"""
eval_multiref.py -- designation gap and first-position bias of a train_multiref.py checkpoint.
Last modified: 2026-11-22 (copy rate on generated images, --copy-seeds, --copy-max; --overfit-folders honoured)
               2026-11-19 (role mode "tid")

Question answered: when the model has learned to follow a designated main image, does it follow
the DESIGNATION, or has it learned "the first input is the main image" regardless of the prompt?
With --role slot the main image is always moved to the first position during training, so the
second behaviour is a real risk. With --role tid the main image keeps its position (marked by
t = --main-t of the run, read from args.json) and no position is special without a designation, so the bias is expected to stay near
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

Copy rate (--copy-seeds > 0, on generated images)
    For every evaluated designated sample (main k) and --copy-seeds seeds, an image is generated
    with main k designated (inputs in their original order), and for every folder one image
    without a main. Image-level CLIP features of the Kandinsky 2.2 prior (the encoder of
    synthesize.py / select_targets.py) are compared after the transformer is unloaded:
        cos_main     cos(output, input_k)
        cos_target   max over the kept targets of main k of cos(output, target)
        copy         cos_main > cos_target: the output is closer to the main INPUT than to its
                     own TARGET, i.e. the designation produced (a variant of) the input
        near_copy    cos_main >= --copy-max (0.80, the copy threshold of select_targets.py)
        input_hit    input k is the input most similar to the output (chance 1 / n inputs):
                     the main input dominates
        target_hit   target k is more similar to the output than every target of the other
                     mains of the folder (chance 1 / n mains; folders with >= 2 mains): the
                     image follows the designation, not only the loss
        no main      near_copy_none: max_j cos(output, input_j) >= --copy-max (one input copied
                     although no main is designated)
    Reading: copy_rate near 0 and target_acc near 1 is the wanted behaviour. A high copy_rate
    together with a high input_acc means "designated = copied". Compare with --base-only: the
    base model copies one reference under every condition.
    -> copy.csv (one line per image), copy_overview.jpeg, generated images in copy/

Rotation images (--image-folders > 0)
    For a few designated folders: inputs and targets, then for every seed one row
    "no main, input i first" (base prompt, input i moved first, the others in their original
    order) and one row "main = input i" (designated by the role mode). If the first row
    changes with the first input in the same way as the second row, the model uses the first
    position as the main image.
    -> <output-dir>/rotation_<folder>.jpeg

Output (default <checkpoint>/eval)
    summary.json     metrics overall and per sigma, copy metrics, settings
    per_sample.csv   one line per sample, sigma and noise draw
    copy.csv, copy_overview.jpeg, copy/   copy rate (see above)
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

    # loss metrics only (no generated images)
    python eval_multiref.py --checkpoint runs/<run>/checkpoint-005000 --copy-seeds 0 --image-folders 0

Settings (role, layout, LoRA, data, overfit subset, seed) are read from <run>/args.json, the
file train_multiref.py writes next to the checkpoints.
"""

import argparse
import csv
import gc
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
# image encoder for the copy rate (same features as synthesize.py / select_targets.py)
CLIP_PRIOR = "kandinsky-community/kandinsky-2-2-prior"


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
    p.add_argument("--image-steps", type=int, default=28, help="Denoising steps of the rotation / copy images")
    p.add_argument("--copy-seeds", type=int, default=2,
                   help="Seeds per designated sample for the copy rate (0 = no copy rate)")
    p.add_argument("--copy-max", type=float, default=0.80,
                   help="near_copy threshold of cos(output, main input) (select_targets.py --copy-max)")
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
        names = [x.strip() for x in (getattr(run, "overfit_folders", None) or "").split(",") if x.strip()]
        pool = load_rows(run.data_dir, "train") + (load_rows(run.data_dir, "val") if names else [])
        return pick_overfit_rows(pool, run.overfit, seed=run.seed, folders=names or None), "overfit"
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
def make_generator(pipe, role, store, run, args):
    """generate(inputs, order, main_pos, prompt_key, seed) -> PIL image, memoized (the copy rate
    and the rotation images share the designated images)."""
    memo = {}
    size = store.target_size

    @torch.no_grad()
    def generate(inputs, order, main_pos, key, seed):
        mk = (tuple(inputs), tuple(order), main_pos, key, seed)
        if mk not in memo:
            pipe.transformer.eval()
            refs = [store.ref(inputs[i]).unsqueeze(0).to(DEVICE, DTYPE) for i in order]
            pr = store.prompt(key)
            with torch.autocast("cuda", dtype=DTYPE):
                out = pipe.generate_from_latents(
                    refs, main_pos, run.role, run.layout, role, height=size, width=size,
                    main_t=getattr(run, "main_t", MAIN_T), prompt_embeds=pr["prompt_embeds"].unsqueeze(0).to(DEVICE, DTYPE),
                    pooled_prompt_embeds=pr["pooled_prompt_embeds"].unsqueeze(0).to(DEVICE, DTYPE),
                    num_inference_steps=args.image_steps, guidance_scale=run.guidance_scale,
                    generator=torch.Generator(device=DEVICE).manual_seed(seed))
            memo[mk] = out.images[0]
        return memo[mk]

    return generate


def designated_image(generate, inputs, k, run, seed):
    """Image with input k designated, inputs in their original order (as in generate_multi.py)."""
    order, main_pos, _ = arrange(len(inputs), k, run.role, shuffle=False)
    return generate(inputs, order, main_pos, prompt_key(main_pos, run.role), seed)


@torch.no_grad()
def rotation_images(generate, by_folder, folders, run, args, out_dir):
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

        grid = [[(Image.open(p).convert("RGB"), f"input_{i}") for i, p in enumerate(inputs)]]
        grid.append([(Image.open(mains[i]["targets"][0]).convert("RGB"), f"target, main = input_{i}")
                     if i in mains else None for i in range(n)])
        for s in range(args.image_seeds):
            seed = run.seed + s
            grid.append([(generate(inputs, first(list(range(n)), i), None, "base", seed),
                          f"no main, input_{i} first (seed {seed})") for i in range(n)])
            grid.append([(designated_image(generate, inputs, i, run, seed), f"main = input_{i} (seed {seed})")
                         for i in range(n)])

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
# Copy rate
# --------------------------------------------------------------------------------------
def generate_copy_images(generate, picked, run, args, out_dir):
    """Designated images of every picked sample and no-main images of every folder, saved to copy/.

    Returns a list of entries {kind, row or folder rows, seed, path}.
    """
    img_dir = out_dir / "copy"
    img_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    seeds = [run.seed + s for s in range(args.copy_seeds)]
    folders = list(dict.fromkeys(r["folder"] for r in picked))
    for folder in folders:
        rows = sorted([r for r in picked if r["folder"] == folder], key=lambda r: r["main"])
        inputs = rows[0]["inputs"]
        name = Path(folder).name
        for seed in seeds:
            img = generate(inputs, list(range(len(inputs))), None, "base", seed)
            path = img_dir / f"{name}__none__seed{seed}.jpeg"
            img.save(path, quality=92)
            entries.append({"kind": "none", "folder": folder, "inputs": inputs, "seed": seed, "path": path})
            for row in rows:
                img = designated_image(generate, inputs, row["main"], run, seed)
                path = img_dir / f"{name}__main{row['main']}__seed{seed}.jpeg"
                img.save(path, quality=92)
                entries.append({"kind": "main", "folder": folder, "inputs": inputs, "row": row, "seed": seed,
                                "path": path})
        print(f"  copy images: {name} done")
    return entries


class ClipImageFeatures:
    """Normalized image embeddings of the Kandinsky 2.2 prior's CLIP image encoder (cached per path)."""

    def __init__(self, device=DEVICE):
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        self.device = device
        self.proc = CLIPImageProcessor.from_pretrained(CLIP_PRIOR, subfolder="image_processor")
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            CLIP_PRIOR, subfolder="image_encoder", torch_dtype=torch.float16).to(device).eval()
        self.cache = {}

    @torch.no_grad()
    def __call__(self, path):
        key = str(path)
        if key not in self.cache:
            px = self.proc(Image.open(path).convert("RGB"), return_tensors="pt").pixel_values
            emb = self.model(px.to(self.device, torch.float16)).image_embeds.float()[0]
            self.cache[key] = torch.nn.functional.normalize(emb, dim=-1).cpu()
        return self.cache[key]


def score_copy(entries, by_folder, clip, args):
    """One record per generated image (see the module docstring)."""
    cos = lambda a, b: float(a @ b)
    records = []
    for e in entries:
        out = clip(e["path"])
        cos_inputs = [cos(out, clip(p)) for p in e["inputs"]]
        best_input = max(range(len(cos_inputs)), key=lambda j: cos_inputs[j])
        rec = {"image": e["path"].name, "folder": Path(e["folder"]).name, "kind": e["kind"], "seed": e["seed"],
               "n_inputs": len(cos_inputs), "best_input": best_input, "cos_best_input": max(cos_inputs)}
        if e["kind"] == "none":
            rec["near_copy"] = float(max(cos_inputs) >= args.copy_max)
            records.append(rec)
            continue
        row, k = e["row"], e["row"]["main"]
        cos_targets = [cos(out, clip(t)) for t in row["targets"]]
        cos_target_input = max(cos(clip(t), clip(e["inputs"][k])) for t in row["targets"])
        others = [r for r in by_folder[e["folder"]] if r["main"] != k]
        cos_other_targets = [max(cos(out, clip(t)) for t in r["targets"]) for r in others]
        rec.update({
            "main": k, "cos_main": cos_inputs[k], "cos_target": max(cos_targets),
            "cos_target_input": cos_target_input,
            "copy": float(cos_inputs[k] > max(cos_targets)),
            "near_copy": float(cos_inputs[k] >= args.copy_max),
            "input_hit": float(best_input == k),
            "n_mains": len(others) + 1,
            "target_hit": float(max(cos_targets) > max(cos_other_targets)) if others else None,
        })
        records.append(rec)
    return records


def summarize_copy(records):
    mains = [r for r in records if r["kind"] == "main"]
    none = [r for r in records if r["kind"] == "none"]
    mean = lambda rs, key: sum(r[key] for r in rs) / len(rs) if rs else None
    with_targets = [r for r in mains if r["target_hit"] is not None]
    return {
        "n_designated_images": len(mains),
        "copy_rate": mean(mains, "copy"),
        "near_copy_rate": mean(mains, "near_copy"),
        "input_acc": mean(mains, "input_hit"),
        "input_chance": mean([{"c": 1 / r["n_inputs"]} for r in mains], "c"),
        "target_acc": mean(with_targets, "target_hit"),
        "target_chance": mean([{"c": 1 / r["n_mains"]} for r in with_targets], "c"),
        "cos_main": mean(mains, "cos_main"),
        "cos_target": mean(mains, "cos_target"),
        "cos_target_input": mean(mains, "cos_target_input"),
        "n_no_main_images": len(none),
        "near_copy_rate_no_main": mean(none, "near_copy"),
        "per_main": {
            key: {"copy_rate": mean(rs, "copy"), "cos_main": mean(rs, "cos_main"), "cos_target": mean(rs, "cos_target")}
            for key, rs in _group(mains, lambda r: f"{r['folder']}__main{r['main']}").items()},
    }


def _group(records, key):
    out = defaultdict(list)
    for r in records:
        out[key(r)].append(r)
    return out


def copy_overview(entries, records, out_dir, tile=200):
    """Per folder: inputs; then per seed: no main, each designated main (red caption = copy)."""
    try:
        font = ImageFont.load_default(size=12)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    rec_by_image = {r["image"]: r for r in records}
    gap, cap = 6, 30
    rows = []
    for folder, es in _group(entries, lambda e: e["folder"]).items():
        inputs = es[0]["inputs"]
        rows.append([(Image.open(p).convert("RGB"), f"{Path(folder).name[:24]} input_{i}", None)
                     for i, p in enumerate(inputs)])
        for seed, ses in _group(es, lambda e: e["seed"]).items():
            row = []
            for e in ses:
                r = rec_by_image[e["path"].name]
                if e["kind"] == "none":
                    text = f"no main, seed {seed}\nbest input_{r['best_input']} {r['cos_best_input']:.2f}"
                    bad = r["near_copy"]
                else:
                    text = (f"main {r['main']}, seed {seed}\nin {r['cos_main']:.2f} tgt {r['cos_target']:.2f}"
                            + (" COPY" if r["copy"] else ""))
                    bad = r["copy"]
                row.append((Image.open(e["path"]).convert("RGB"), text, bad))
            rows.append(row)
    n_cols = max(len(r) for r in rows)
    canvas = Image.new("RGB", (gap + n_cols * (tile + gap), gap + len(rows) * (tile + cap + gap)), "white")
    draw = ImageDraw.Draw(canvas)
    for ri, row in enumerate(rows):
        y = gap + ri * (tile + cap + gap)
        for ci, (img, text, bad) in enumerate(row):
            x = gap + ci * (tile + gap)
            canvas.paste(img.resize((tile, tile)), (x, y))
            draw.multiline_text((x + 2, y + tile + 2), text, fill=(200, 0, 0) if bad else "black", font=font)
    path = out_dir / "copy_overview.jpeg"
    canvas.save(path, quality=90)
    return path


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

    images, copy_entries = [], []
    if args.image_folders > 0 or args.copy_seeds > 0:
        vae = AutoencoderKL.from_pretrained(run.base_model, subfolder="vae", torch_dtype=DTYPE).to(DEVICE)
        pipe = MultiRefKontextPipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(run.base_model, subfolder="scheduler"),
            vae=vae, text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None, transformer=transformer)
        pipe.set_progress_bar_config(disable=True)
        generate = make_generator(pipe, role, store, run, args)
        if args.copy_seeds > 0:
            print(f"\ncopy rate: generating {len(picked)} designated sample(s) x {args.copy_seeds} seed(s) "
                  f"+ no-main images")
            copy_entries = generate_copy_images(generate, picked, run, args, out_dir)
        if args.image_folders > 0:
            folders = list(dict.fromkeys(r["folder"] for r in picked))[: args.image_folders]
            images = rotation_images(generate, by_folder, folders, run, args, out_dir)
        # free the GPU for the image encoder
        del generate, pipe, vae
    del transformer, role
    gc.collect()
    torch.cuda.empty_cache()

    if copy_entries:
        clip = ClipImageFeatures()
        copy_records = score_copy(copy_entries, by_folder, clip, args)
        with open(out_dir / "copy.csv", "w", newline="") as f:
            fields = list(dict.fromkeys(k for r in copy_records for k in r))
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(copy_records)
        copy_summary = summarize_copy(copy_records)
        copy_summary["overview"] = str(copy_overview(copy_entries, copy_records, out_dir))
        copy_summary["copy_max"] = args.copy_max
        summary["copy"] = copy_summary
        fmt = lambda v: "-" if v is None else f"{v:.2f}"
        c = copy_summary
        print(f"\ncopy rate ({c['n_designated_images']} designated images, {c['n_no_main_images']} without a main):")
        print(f"  copy (closer to the main input than to its target) {fmt(c['copy_rate'])}, "
              f"near copy (cos_main >= {args.copy_max:g}) {fmt(c['near_copy_rate'])}")
        print(f"  input_acc {fmt(c['input_acc'])} (chance {fmt(c['input_chance'])}), "
              f"target_acc {fmt(c['target_acc'])} (chance {fmt(c['target_chance'])})")
        print(f"  mean cos(output, main input) {fmt(c['cos_main'])}, cos(output, target) {fmt(c['cos_target'])}, "
              f"cos(target, main input) {fmt(c['cos_target_input'])}")
        print(f"  no main: near copy of one input {fmt(c['near_copy_rate_no_main'])}")
        for key, m in c["per_main"].items():
            print(f"    {key}: copy {fmt(m['copy_rate'])}, cos_main {fmt(m['cos_main'])}, cos_target {fmt(m['cos_target'])}")
        print(f"  overview: {copy_summary['overview']}")

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
