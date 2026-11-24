"""
synthesize_sweep.py -- designated-subject target candidates over a (t, strength, seed) grid.
Last modified: 2026-11-24 (--attr-mains: also attribute inputs as mains)

Generation step of the automatic target selection (selection: score_synth.py). Same method,
models and measurements as synthesize.py; what changes:

* Grid. Every main gets a t x strength x seed grid chosen by its class (close / far, as in v2.1),
  3 x 3 x 2 by default instead of 2 x 2 x 1. The centres of the grids are the v2.1 defaults:
      close : t in --close-t (0, 0.15, 0.3)   s in --close-s (0.6, 0.75, 0.9)
      far   : t in --far-t  (0.3, 0.45, 0.6)  s in --far-s  (0.4, 0.55, 0.7)
      seeds : --seeds (0, 1), the seeds of the default decomposition sweep (run_sweep.sh)
* Mains. By default only the inputs that score_sweep.py found to show a prominent object of
  their own (score.json of the decomposition folder: label "unique", own_type "object",
  own_p >= --prominent). Material / color / scene inputs are not synthesized as mains.
  --attr-mains also synthesizes the unique inputs whose own concept is an attribute (material,
  color, shape, pattern, style, scene); score_synth.py considers them only with its own
  --attr-mains (presence check of the main concept).
      --main-mode auto   score.json if the folder has one, otherwise every input (default)
      --main-mode score  folders without score.json are skipped
      --main-mode all    every input (as synthesize.py)
      --mains 0,2        these indices in every folder (overrides --main-mode)
* Incremental. Cells already on disk (same file name, same --size / --steps / --guidance-scale)
  are not generated again, so the grid can be extended later (more t / s values or seeds)
  and an interrupted run continues where it stopped.

File names: main{k}/t{t:.2f}_s{s:.2f}_seed{seed}.jpeg. synth_meta.json keeps the v2.1 layout
(results -> cells) with an additional "seed" per cell; only the cells of the current grid and
mains are listed (files of other cells stay on disk and are ignored).

Usage
-----
    python synthesize_sweep.py                                   # generate_source -> data/synth
    python synthesize_sweep.py --root generate_source --output-dir data/synth --seeds 1
    python synthesize_sweep.py --limit 3                         # quick test on three folders

Then: python score_synth.py (see its docstring).
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from diffusers import KandinskyV22Img2ImgPipeline
from diffusers.utils import load_image
from PIL import Image
from tqdm import tqdm

from decompose_kmeans import load_kandinsky
from synthesize import (DEVICE, clip_feature, cos, find_folders, generate_img2img, generate_plain, list_inputs,
                             make_overview, parse_float_list, parse_int_list, target_embedding)

SWEEP_VERSION = 1


def cell_name(t: float, s: float, seed: int) -> str:
    """File stem of a cell, e.g. t0.30_s0.40_seed0."""
    return f"t{t:.2f}_s{s:.2f}_seed{seed}"


ATTRIBUTE_TYPES = ("material", "color", "shape", "pattern", "style", "scene", "other")


def choose_mains(folder: Path, n: int, args):
    """(main indices, where the choice came from, {index: own concept name}). Indices None = skip the folder."""
    if args.mains is not None:
        return [k for k in args.mains if 0 <= k < n], "--mains", {}
    if args.main_mode == "all":
        return list(range(n)), "all inputs", {}
    path = folder / "score.json"
    if not path.exists():
        if args.main_mode == "score":
            return None, "no score.json", {}
        return list(range(n)), "all inputs (no score.json)", {}
    rec = json.loads(path.read_text())
    names = {x["index"]: x.get("own") for x in rec.get("inputs", [])}
    mains = []
    for x in rec.get("inputs", []):
        if x.get("label") != "unique" or x["index"] >= n or x.get("own_p") is None:
            continue
        if x.get("own_type") == "object" and x["own_p"] >= args.prominent:
            mains.append(x["index"])
        elif args.attr_mains and x.get("own_type") in ATTRIBUTE_TYPES:
            mains.append(x["index"])
    return mains, "score.json" + (" (objects + attributes)" if args.attr_mains else ""), names


def settings(args) -> dict:
    """Generation settings that must match for a cell on disk to be reused."""
    return {"size": args.size, "steps": args.steps, "guidance_scale": args.guidance_scale}


@torch.inference_mode()
def process_folder(folder: Path, out: Path, args, models):
    prior, decoder, img2img, negative_emb = models
    input_paths = list_inputs(folder)
    n = len(input_paths)
    if n < 2:
        print(f"[skip] {folder}: fewer than 2 input images")
        return None
    mains, mains_from, own_names = choose_mains(folder, n, args)
    if not mains:
        print(f"[skip] {folder}: no main ({mains_from if mains is None else 'no prominent object input'})")
        return None

    # cells already generated with the same settings are reused
    meta_path = out / "synth_meta.json"
    done = {}
    if meta_path.exists() and not args.overwrite:
        old = json.loads(meta_path.read_text())
        if old.get("sweep_version") != SWEEP_VERSION:
            print(f"[skip] {out}: synth_meta.json is not from this script (use another --output-dir or --overwrite)")
            return None
        if old.get("settings") != settings(args):
            print(f"[skip] {out}: generated with other settings {old.get('settings')} (use --overwrite)")
            return None
        done = {c["file"]: c for r in old.get("results", []) for c in r["cells"] if (out / c["file"]).exists()}
    out.mkdir(parents=True, exist_ok=True)
    dtype = negative_emb.dtype

    inputs = [load_image(str(p)).convert("RGB").resize((args.size, args.size), Image.LANCZOS) for p in input_paths]
    f_inputs = [clip_feature(img, prior) for img in inputs]
    E = torch.load(folder / "embeds.pt", map_location="cpu")
    e = E["e"].float().to(DEVICE)
    e_inputs = E.get("e_inputs")
    if e_inputs is None:
        print(f"[note] {folder}: no e_inputs in embeds.pt, using re-encoded input images instead")
        e_inputs, embedding_source = torch.stack(f_inputs), "reencoded_inputs"
    else:
        e_inputs, embedding_source = e_inputs.float().to(DEVICE), "embeds.pt"
    if e_inputs.shape[0] != n:
        print(f"[skip] {folder}: {n} input images but {e_inputs.shape[0]} input embeddings")
        return None

    if (folder / "recon.jpeg").exists():
        recon, recon_source = load_image(str(folder / "recon.jpeg")).convert("RGB"), "decomposition folder"
    elif (out / "recon.jpeg").exists():
        recon, recon_source = load_image(str(out / "recon.jpeg")).convert("RGB"), "decoded in an earlier run"
    else:
        print(f"[note] {folder}: recon.jpeg missing, decoding it from e")
        recon = generate_plain(decoder, e.to(dtype), negative_emb, args.seeds[0], args.steps, args.size,
                               args.guidance_scale)
        recon.save(out / "recon.jpeg", quality=95)
        recon_source = "decoded in this run"
    f_recon = clip_feature(recon, prior)
    closeness = [cos(f, f_recon) for f in f_inputs]
    classes = ["close" if c >= args.close_threshold else "far" for c in closeness]

    header = [(recon, "recon")]
    for i, img in enumerate(inputs):
        tag = (" MAIN" if i in mains else "") + (f" [{own_names[i]}]" if own_names.get(i) else "")
        header.append((img, f"input_{i}{tag}\nclose={closeness[i]:.2f} ({classes[i]})"))
    rows, results, n_new = [header], [], 0

    for k in mains:
        e_k = e_inputs[k]
        others = [i for i in range(n) if i != k]
        t_list, s_list = (args.close_t, args.close_s) if classes[k] == "close" else (args.far_t, args.far_s)
        (out / f"main{k}").mkdir(exist_ok=True)
        rows.append(None)
        cells = []
        for seed in args.seeds:
            for t in t_list:
                e_T = target_embedding(e, e_k, t)
                row = [(inputs[k], f"main = input_{k} ({classes[k]})\nt = {t:g}, seed {seed}")]
                for s in s_list:
                    file = f"main{k}/{cell_name(t, s, seed)}.jpeg"
                    if file in done:
                        img, c = Image.open(out / file).convert("RGB"), done[file]
                    else:
                        img = generate_img2img(img2img, e_T.to(dtype), inputs[k], negative_emb, s, seed,
                                               args.steps, args.size, args.guidance_scale)
                        img.save(out / file, quality=95)
                        n_new += 1
                        f_t = clip_feature(img, prior)
                        cos_inputs = [cos(f_t, f) for f in f_inputs]
                        c = {"t": t, "strength": s, "seed": seed, "file": file,
                             "denoising_steps": int(args.steps * s),
                             "cos_eT_e": cos(e_T, e), "cos_eT_ek": cos(e_T, e_k),
                             "cos_main": cos_inputs[k], "lift": cos_inputs[k] - closeness[k],
                             "margin": cos_inputs[k] - max(cos_inputs[i] for i in others),
                             "cos_inputs": cos_inputs, "cos_recon": cos(f_t, f_recon)}
                    cells.append(c)
                    row.append((img, f"s={s:g}  main={c['cos_main']:.2f}\n"
                                     f"lift={c['lift']:+.2f} margin={c['margin']:+.2f}"))
                rows.append(row)
        results.append({"main": k, "class": classes[k], "closeness": closeness[k], "own": own_names.get(k),
                        "t": t_list, "strength": s_list, "seeds": args.seeds, "cells": cells})

    title = (f"{folder.name}   candidates for score_synth.py; e_T = (1-t) e + t e_main, img2img init = main input\n"
             "close = cos(input, recon); main = cos(target, main); lift = main - close; margin = main - best other")
    make_overview(rows, out / "overview.jpeg", title=title, tile=args.tile)

    meta = {"sweep_version": SWEEP_VERSION, "folder": str(folder), "n_inputs": n, "mains": mains,
            "mains_from": mains_from, "own": {str(i): v for i, v in own_names.items()},
            "close_threshold": args.close_threshold,
            "close_grid": {"t": args.close_t, "strength": args.close_s},
            "far_grid": {"t": args.far_t, "strength": args.far_s}, "seeds": args.seeds,
            "settings": settings(args), "closeness": closeness, "classes": classes, "recon_source": recon_source,
            "embedding_source": embedding_source, "results": results}
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"folder": str(folder), "output": str(out), "mains": mains, "mains_from": mains_from,
            "classes": [classes[k] for k in mains], "cells": sum(len(r["cells"]) for r in results), "new": n_new}


def main():
    p = argparse.ArgumentParser(description="Designated-subject target candidates over a t x strength x seed grid")
    p.add_argument("--root", default="generate_source", help="Decomposition folders (or one such folder)")
    p.add_argument("--output-dir", default="data/synth")
    p.add_argument("--exclude", default="screening", help="Comma-separated directory names under --root to skip")
    p.add_argument("--main-mode", choices=("auto", "score", "all"), default="auto",
                   help="auto: object inputs from score.json if present, else every input; score: only folders "
                        "with score.json; all: every input")
    p.add_argument("--prominent", type=float, default=0.5, help="An object input is a main if its own_p >= this")
    p.add_argument("--attr-mains", action="store_true",
                   help="With score.json: also unique inputs whose own concept is an attribute (color, pattern, ...)")
    p.add_argument("--mains", type=parse_int_list, default=None, help="These input indices in every folder")
    p.add_argument("--close-threshold", type=float, default=0.6, help="close if cos(input, recon) >= this")
    p.add_argument("--close-t", type=parse_float_list, default=[0.0, 0.15, 0.3])
    p.add_argument("--close-s", type=parse_float_list, default=[0.6, 0.75, 0.9])
    p.add_argument("--far-t", type=parse_float_list, default=[0.3, 0.45, 0.6])
    p.add_argument("--far-s", type=parse_float_list, default=[0.4, 0.55, 0.7])
    p.add_argument("--seeds", type=parse_int_list, default=[0, 1], help="Kandinsky seeds of the grid")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--tile", type=int, default=200, help="Tile size in the overview image")
    p.add_argument("--limit", type=int, default=None, help="Only the first LIMIT folders")
    p.add_argument("--overwrite", action="store_true", help="Regenerate every cell")
    args = p.parse_args()

    for name in ("close_t", "close_s", "far_t", "far_s", "seeds"):
        if not getattr(args, name):
            raise ValueError(f"--{name.replace('_', '-')} must not be empty.")
    for s in args.close_s + args.far_s:
        if not (0.0 < s <= 1.0) or int(args.steps * s) < 1:
            raise ValueError(f"Invalid strength {s}: must be in (0, 1] and give at least one step.")

    root = Path(args.root)
    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}
    folders = find_folders(root, exclude)[: args.limit]
    print(f"Found {len(folders)} decomposition folder(s) under {root}")
    if not folders:
        raise SystemExit("No folders with embeds.pt and input_0.jpeg found.")

    prior, decoder, negative_emb = load_kandinsky()
    img2img = KandinskyV22Img2ImgPipeline(unet=decoder.unet, scheduler=decoder.scheduler, movq=decoder.movq)
    decoder.set_progress_bar_config(disable=True)
    img2img.set_progress_bar_config(disable=True)
    models = (prior, decoder, img2img, negative_emb)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary, t0 = [], time.time()
    for folder in tqdm(folders, desc="folders"):
        rel = folder.relative_to(root)
        res = process_folder(folder, out_root / (rel if str(rel) != "." else folder.name), args, models)
        if res is not None:
            summary.append(res)

    if summary:
        (out_root / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
        mains = sum(len(s["mains"]) for s in summary)
        cls = [c for s in summary for c in s["classes"]]
        print(f"\n{len(summary)} folder(s), {mains} main(s) ({cls.count('close')} close, {cls.count('far')} far), "
              f"{sum(s['cells'] for s in summary)} cell(s), {sum(s['new'] for s in summary)} generated in "
              f"{time.time() - t0:.0f} s")
        print("mains from: " + ", ".join(f"{k} {v}" for k, v in Counter(s["mains_from"] for s in summary).items()))
        print(f"next: python score_synth.py --synth-root {out_root}")


if __name__ == "__main__":
    main()
