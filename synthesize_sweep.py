"""
synthesize_sweep.py -- designated-subject target candidates over a (t, strength, seed) grid.
Last modified: 2026-11-24 (batched img2img: one call per main and strength; --gpus for several GPUs,
               --shard for manual splitting)
               2026-11-24 (attribute inputs are synthesized as mains by default; --no-attr-mains)

Generation step of the automatic target selection (selection: score_synth.py). Same method,
models and measurements as synthesize.py; what changes:

* Grid. Every main gets a t x strength x seed grid chosen by its class (close / far, as in v2.1),
  3 x 3 x 2 by default instead of 2 x 2 x 1. The centres of the grids are the v2.1 defaults:
      close : t in --close-t (0, 0.15, 0.3)   s in --close-s (0.6, 0.75, 0.9)
      far   : t in --far-t  (0.3, 0.45, 0.6)  s in --far-s  (0.4, 0.55, 0.7)
      seeds : --seeds (0, 1), the seeds of the default decomposition sweep (run_sweep.sh)
* Mains. From score.json of the decomposition folder (score_sweep.py): the unique inputs that
  show a prominent object of their own (own_type "object", own_p >= --prominent) and, by default,
  the unique inputs whose own concept is an attribute (material, color, shape, pattern, style,
  scene); score_synth.py checks the latter by the presence of their concept in the target.
  --no-attr-mains synthesizes object inputs only (use the same flag in score_synth.py).
      --main-mode auto   score.json if the folder has one, otherwise every input (default)
      --main-mode score  folders without score.json are skipped
      --main-mode all    every input (as synthesize.py)
      --mains 0,2        these indices in every folder (overrides --main-mode)
* Incremental. Cells already on disk (same file name, same --size / --steps / --guidance-scale)
  are not generated again, so the grid can be extended later (more t / s values or seeds)
  and an interrupted run continues where it stopped.
* Batched. The cells of one main with the same strength share the init image and the number
  of denoising steps, so they are generated in one img2img call (all t values x all seeds,
  at most --batch-size images per call). Every image keeps its own torch.Generator(seed): the
  noise is the same as in a call with one image.
* Several GPUs. --gpus 2,3 (or 0-7) starts one process per listed GPU; the folders are split
  round-robin between them (--shard i/n), every process writes its own log to
  <output-dir>/_logs/shard_<i>of<n>.log, and the summaries are merged at the end. --gpus 3 runs
  a single process on GPU 3. Without --gpus the script runs on the default GPU. --shard i/n
  alone processes the i-th of n parts (0-based), e.g. to start the parts by hand.

File names: main{k}/t{t:.2f}_s{s:.2f}_seed{seed}.jpeg. synth_meta.json keeps the v2.1 layout
(results -> cells) with an additional "seed" per cell; only the cells of the current grid and
mains are listed (files of other cells stay on disk and are ignored).

Usage
-----
    python synthesize_sweep.py                                   # generate_source -> data/synth
    python synthesize_sweep.py --root generate_source --output-dir data/synth --seeds 1
    python synthesize_sweep.py --limit 3                         # quick test on three folders
    python synthesize_sweep.py --gpus 3                          # one process on GPU 3
    python synthesize_sweep.py --gpus 2,3                        # two processes on GPUs 2 and 3
    python synthesize_sweep.py --gpus 0-7                        # eight GPUs

Then: python score_synth.py (see its docstring).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from diffusers import KandinskyV22Img2ImgPipeline
from diffusers.utils import load_image
from PIL import Image
from tqdm import tqdm

from decompose_kmeans import load_kandinsky
from synthesize import (DEVICE, clip_feature, cos, find_folders, generate_plain, list_inputs,
                             make_overview, parse_float_list, parse_int_list, target_embedding)

SWEEP_VERSION = 1


def cell_name(t: float, s: float, seed: int) -> str:
    """File stem of a cell, e.g. t0.30_s0.40_seed0."""
    return f"t{t:.2f}_s{s:.2f}_seed{seed}"


ATTRIBUTE_TYPES = ("material", "color", "shape", "pattern", "style", "scene", "other")


@torch.inference_mode()
def generate_img2img_batch(img2img, image_embeds, init_image, negative_emb, strength: float, seeds,
                           steps: int, size: int, guidance: float):
    """Kandinsky 2.2 img2img for several embeddings at once (same init image and strength).

    image_embeds: [B, D]; seeds: B seeds. One generator per image, so every image gets the noise of
    a single-image call with its seed.
    """
    b = image_embeds.shape[0]
    generators = [torch.Generator(DEVICE).manual_seed(int(sd)) for sd in seeds]
    return img2img(
        image_embeds=image_embeds,
        image=[init_image] * b,
        negative_image_embeds=negative_emb.reshape(1, -1).repeat(b, 1),
        height=size,
        width=size,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        num_images_per_prompt=1,
        generator=generators,
    ).images


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
        grid = [(seed, t, s) for seed in args.seeds for t in t_list for s in s_list]
        images, cell_meta = {}, {}
        todo = []
        for seed, t, s in grid:
            file = f"main{k}/{cell_name(t, s, seed)}.jpeg"
            if file in done:
                images[file], cell_meta[file] = Image.open(out / file).convert("RGB"), done[file]
            else:
                todo.append((seed, t, s, file))

        # one img2img call per strength (same init image and number of steps), split by --batch-size
        for s in s_list:
            group = [x for x in todo if x[2] == s]
            for i in range(0, len(group), args.batch_size):
                chunk = group[i:i + args.batch_size]
                e_Ts = [target_embedding(e, e_k, t) for _, t, _, _ in chunk]
                imgs = generate_img2img_batch(img2img, torch.stack(e_Ts).to(dtype), inputs[k], negative_emb, s,
                                              [seed for seed, _, _, _ in chunk], args.steps, args.size,
                                              args.guidance_scale)
                for (seed, t, _, file), e_T, img in zip(chunk, e_Ts, imgs):
                    img.save(out / file, quality=95)
                    n_new += 1
                    f_t = clip_feature(img, prior)
                    cos_inputs = [cos(f_t, f) for f in f_inputs]
                    images[file] = img
                    cell_meta[file] = {
                        "t": t, "strength": s, "seed": seed, "file": file, "denoising_steps": int(args.steps * s),
                        "cos_eT_e": cos(e_T, e), "cos_eT_ek": cos(e_T, e_k),
                        "cos_main": cos_inputs[k], "lift": cos_inputs[k] - closeness[k],
                        "margin": cos_inputs[k] - max(cos_inputs[i] for i in others),
                        "cos_inputs": cos_inputs, "cos_recon": cos(f_t, f_recon)}

        # overview rows and meta in grid order (seed, t, s)
        rows.append(None)
        cells = []
        for seed in args.seeds:
            for t in t_list:
                row = [(inputs[k], f"main = input_{k} ({classes[k]})\nt = {t:g}, seed {seed}")]
                for s in s_list:
                    file = f"main{k}/{cell_name(t, s, seed)}.jpeg"
                    c = cell_meta[file]
                    cells.append(c)
                    row.append((images[file], f"s={s:g}  main={c['cos_main']:.2f}\n"
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


def parse_gpus(text: str):
    """'3' -> [3]; '2,3' -> [2, 3]; '0-7' -> [0, ..., 7]; '0,2-3' -> [0, 2, 3]."""
    gpus = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            gpus.extend(range(int(a), int(b) + 1))
        else:
            gpus.append(int(part))
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError(f"--gpus {text!r}: give distinct GPU indices, e.g. 3, 2,3 or 0-7")
    return gpus


def parse_shard(text: str):
    """'1/4' -> (1, 4): the second of four parts (0-based)."""
    i, n = (int(x) for x in text.split("/"))
    if not (n >= 1 and 0 <= i < n):
        raise ValueError(f"--shard {text!r}: expected i/n with 0 <= i < n")
    return i, n


def strip_options(argv, names):
    """Remove the given options (and their values) from an argument list."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in names:
            skip = True
            continue
        if any(a.startswith(name + "=") for name in names):
            continue
        out.append(a)
    return out


def summary_path(out_root: Path, shard) -> Path:
    return out_root / ("sweep_summary.json" if shard is None else f"sweep_summary_shard{shard[0]}of{shard[1]}.json")


def launch(args, gpus):
    """One child process per GPU, each with its own --shard; waits for all, then merges the summaries."""
    n = len(gpus)
    out_root = Path(args.output_dir)
    logs = out_root / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    base = strip_options(sys.argv[1:], ("--gpus", "--shard"))
    procs = []
    for i, g in enumerate(gpus):
        cmd = [sys.executable, os.path.abspath(__file__)] + base + ["--shard", f"{i}/{n}"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        log_path = logs / f"shard_{i}of{n}.log"
        log = open(log_path, "w")
        procs.append((i, g, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, log_path))
        print(f"shard {i}/{n} on GPU {g}: log {log_path}")
    print("waiting for all shards (follow a log with: tail -f <log>)")
    t0, failed = time.time(), []
    for i, g, proc, log, log_path in procs:
        code = proc.wait()
        log.close()
        print(f"shard {i}/{n} on GPU {g} finished with exit code {code} after {time.time() - t0:.0f} s")
        if code != 0:
            failed.append(str(log_path))

    merged = []
    for i in range(n):
        path = summary_path(out_root, (i, n))
        if path.exists():
            merged.extend(json.loads(path.read_text()))
    if merged:
        summary_path(out_root, None).write_text(json.dumps(merged, indent=2))
        print(f"\n{len(merged)} folder(s), {sum(len(x['mains']) for x in merged)} main(s), "
              f"{sum(x['cells'] for x in merged)} cell(s), {sum(x['new'] for x in merged)} generated in "
              f"{time.time() - t0:.0f} s on {n} GPU(s); summary: {summary_path(out_root, None)}")
    if failed:
        raise SystemExit("failed shard(s), see: " + ", ".join(failed))
    print(f"next: python score_synth.py --synth-root {out_root}")


def main():
    p = argparse.ArgumentParser(description="Designated-subject target candidates over a t x strength x seed grid")
    p.add_argument("--root", default="generate_source", help="Decomposition folders (or one such folder)")
    p.add_argument("--output-dir", default="data/synth")
    p.add_argument("--exclude", default="screening", help="Comma-separated directory names under --root to skip")
    p.add_argument("--main-mode", choices=("auto", "score", "all"), default="auto",
                   help="auto: object (and attribute) inputs from score.json if present, else every input; score: only folders "
                        "with score.json; all: every input")
    p.add_argument("--prominent", type=float, default=0.5, help="An object input is a main if its own_p >= this")
    p.add_argument("--attr-mains", action=argparse.BooleanOptionalAction, default=True,
                   help="With score.json: also unique inputs whose own concept is an attribute (color, pattern, "
                        "...) (default); --no-attr-mains: object inputs only")
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
    p.add_argument("--batch-size", type=int, default=6,
                   help="Images per img2img call (cells of one main with the same strength; 3 t x 2 seeds = 6)")
    p.add_argument("--gpus", default=None,
                   help="GPUs to use: '3' = one process on GPU 3, '2,3' or '0-7' = one process per GPU, folders "
                        "split between them (default: the default GPU, one process)")
    p.add_argument("--shard", default=None, help="i/n: only the i-th of n parts of the folders (0-based)")
    args = p.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    for name in ("close_t", "close_s", "far_t", "far_s", "seeds"):
        if not getattr(args, name):
            raise ValueError(f"--{name.replace('_', '-')} must not be empty.")
    for s in args.close_s + args.far_s:
        if not (0.0 < s <= 1.0) or int(args.steps * s) < 1:
            raise ValueError(f"Invalid strength {s}: must be in (0, 1] and give at least one step.")

    shard = parse_shard(args.shard) if args.shard else None
    if args.gpus is not None:
        gpus = parse_gpus(args.gpus)
        if len(gpus) > 1:
            if shard is not None:
                raise ValueError("use either --gpus with several GPUs or --shard, not both")
            launch(args, gpus)
            return
        # one GPU: this process; must be set before CUDA is initialized (models are loaded below)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        if torch.cuda.is_initialized():
            print("[warn] CUDA was initialized before --gpus could take effect")

    root = Path(args.root)
    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}
    folders = find_folders(root, exclude)[: args.limit]
    print(f"Found {len(folders)} decomposition folder(s) under {root}")
    if shard is not None:
        folders = folders[shard[0]::shard[1]]  # round-robin split, stable because find_folders sorts
        print(f"shard {shard[0]}/{shard[1]}: {len(folders)} folder(s)")
    if not folders:
        raise SystemExit("No folders with embeds.pt and input_0.jpeg found" + (" in this shard." if shard else "."))
    print(f"device: cuda (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")

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
        summary_path(out_root, shard).write_text(json.dumps(summary, indent=2))
        mains = sum(len(s["mains"]) for s in summary)
        cls = [c for s in summary for c in s["classes"]]
        print(f"\n{len(summary)} folder(s), {mains} main(s) ({cls.count('close')} close, {cls.count('far')} far), "
              f"{sum(s['cells'] for s in summary)} cell(s), {sum(s['new'] for s in summary)} generated in "
              f"{time.time() - t0:.0f} s")
        print("mains from: " + ", ".join(f"{k} {v}" for k, v in Counter(s["mains_from"] for s in summary).items()))
        print(f"next: python score_synth.py --synth-root {out_root}")


if __name__ == "__main__":
    main()
