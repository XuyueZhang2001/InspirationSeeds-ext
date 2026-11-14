"""
synthesize_v2.py -- designated-subject targets: embedding interpolation + Kandinsky img2img.
Last modified: 2026-11-14

Purpose
-------
Grid experiment for "designated main subject" targets. It replaces the level-based
reweighting of synthesize_target.py (c_i = w_i / m_i) with a single knob per designated
input, and anchors the rough shape of the target to that input.

Input: decomposition folders written by decompose_v7_2.py (N = 2-4 inputs per folder):
    <root>/<name>/embeds.pt, groups.json, recon.jpeg, input_0.jpeg, input_1.jpeg, ...

For every folder and every input k that is designated as the main subject:

  1. Target embedding (one knob t, no per-image levels):
         e_T = (1 - t) * e + t * e_k,     rescaled to ||e||
     e   = CLIP image embedding of the source              (embeds.pt["e"])
     e_k = embedding that input_k.jpeg was decoded from     (embeds.pt["e_inputs"][k])
     t = 0 gives the source embedding, t = 1 gives input k itself. Moving toward e_k is
     equivalent to amplifying group k and attenuating the other groups.

  2. Decoding with Kandinsky 2.2 img2img: init image = input_k.jpeg, condition = e_T.
     `strength` is the diffusers img2img strength, i.e. the fraction of the noise schedule
     applied to the init image. LARGER strength = FURTHER from the init image:
         ~0.4  keeps outline, details and colors of input k
         ~0.7  keeps the rough layout / proportions only
         ~0.9  almost ignores the init image (close to a plain decode of e_T)
     The number of denoising steps actually run is int(steps * strength).

  3. Grid: every t in --t  x  every strength in --strength, same seed for every cell.

Measurements (recorded, not used as a filter yet)
-------------------------------------------------
Every target, every input and recon.jpeg are re-encoded with the CLIP image encoder of the
Kandinsky prior. With f(.) the re-encoded feature and k the main input:
    cos_main      = cos(f(target), f(input_k))
                    close to 1 -> the target is a copy of the main input
    shift_i       = cos(f(target) - f(input_k), f(input_i) - f(input_k))   for i != k
                    > 0 -> the target moved from the main input toward input i
    shift_others  = mean of shift_i over i != k
These are coarse (global CLIP features); the overview image is the primary result.

Output
------
    <output-dir>/<name>/overview.jpeg         one overview per folder (all designated inputs)
    <output-dir>/<name>/main{k}/t{t}_s{s}.jpeg    individual targets
    <output-dir>/<name>/synth_meta.json       parameters and measurements per cell
    <output-dir>/summary.json                 one entry per processed folder

Overview layout: first row = recon and all inputs; then one block per designated input,
one row per t (first cell = the main input), one column per strength.

Usage
-----
    # default grid: t in {0, 0.3, 0.6} x strength in {0.4, 0.7, 0.9}, every input designated
    python synthesize_v2.py --root hf_generate_source --output-dir output/synth/v2

    # only inputs 0 and 2 as main subject, first 5 folders
    python synthesize_v2.py --mains 0,2 --limit 5

    # add a reference column decoded from e_T alone (no init image)
    python synthesize_v2.py --embed-only

Notes
-----
* Models and loading are the same as in decompose.py / decompose_v7_2.py (Kandinsky 2.2
  prior + decoder); the img2img pipeline reuses the decoder's unet, scheduler and movq, so
  no additional weights are loaded. The SAE is not needed.
* Use the same --size / --steps as the decomposition run (decompose_v7_2.py: 768 / 50).
"""

import argparse
import json
from pathlib import Path

import torch
from diffusers import KandinskyV22Img2ImgPipeline
from diffusers.utils import load_image
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from decompose import load_kandinsky

DEVICE = "cuda"


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def parse_float_list(text: str):
    return [float(x) for x in text.split(",") if x.strip()]


def parse_int_list(text: str):
    return [int(x) for x in text.split(",") if x.strip()]


def cos(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(x, y, dim=0))


def target_embedding(e: torch.Tensor, e_k: torch.Tensor, t: float) -> torch.Tensor:
    """e_T = (1 - t) e + t e_k, rescaled to ||e||."""
    x = (1.0 - t) * e + t * e_k
    return x * (e.norm() / x.norm().clamp_min(1e-8))


def find_folders(root: Path):
    """Decomposition folders under root (root itself included): embeds.pt + input_0.jpeg."""
    found = []
    for emb in sorted(root.rglob("embeds.pt")):
        if (emb.parent / "input_0.jpeg").exists():
            found.append(emb.parent)
    return found


def list_inputs(folder: Path):
    """input_0.jpeg, input_1.jpeg, ... (consecutive indices)."""
    paths, i = [], 0
    while (folder / f"input_{i}.jpeg").exists():
        paths.append(folder / f"input_{i}.jpeg")
        i += 1
    return paths


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def generate_img2img(img2img, image_embeds, init_image, negative_emb, strength: float,
                     seed: int, steps: int, size: int, guidance: float):
    """Kandinsky 2.2 img2img: start from a noised init_image, condition on image_embeds."""
    generator = torch.Generator(DEVICE).manual_seed(seed)
    return img2img(
        image_embeds=image_embeds.unsqueeze(0),
        image=init_image,
        negative_image_embeds=negative_emb,
        height=size,
        width=size,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]


@torch.inference_mode()
def generate_plain(decoder, image_embeds, negative_emb, seed: int, steps: int, size: int, guidance: float):
    """Plain decode of an embedding (no init image), as in decompose_v7_2.generate."""
    generator = torch.Generator(DEVICE).manual_seed(seed)
    return decoder(
        image_embeds=image_embeds.unsqueeze(0),
        negative_image_embeds=negative_emb,
        height=size,
        width=size,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]


@torch.inference_mode()
def clip_feature(image: Image.Image, prior) -> torch.Tensor:
    """CLIP image embedding of a PIL image.

    Same computation as decompose.embed_image (prior.interpolate with weight 1), but calls
    the image encoder directly: interpolate() additionally runs the prior once per call to
    produce a negative embedding, which is not needed here.
    """
    px = prior.image_processor(image, return_tensors="pt").pixel_values
    px = px.to(device=DEVICE, dtype=prior.image_encoder.dtype)
    return prior.image_encoder(px)["image_embeds"][0].float()


# --------------------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------------------
def make_overview(rows, out_path: Path, title: str = "", tile: int = 256):
    """Grid overview.

    rows: list of rows; a row is a list of cells (image, caption) or None (empty cell).
          A row that is None is drawn as a thin separator line between blocks.
    """
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    gap, cap_h, sep_h = 6, 34, 12
    title_h = 40 if title else 0
    n_cols = max(len(r) for r in rows if r is not None)
    row_h = tile + cap_h + gap
    W = n_cols * (tile + gap) + gap
    H = title_h + gap + sum(sep_h if r is None else row_h for r in rows)
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.multiline_text((gap + 2, gap), title, fill="black", font=font, spacing=2)
    y = title_h + gap
    for row in rows:
        if row is None:
            draw.line([(gap, y + sep_h // 2 - gap // 2), (W - gap, y + sep_h // 2 - gap // 2)], fill="gray", width=1)
            y += sep_h
            continue
        for c, cell in enumerate(row):
            if cell is None:
                continue
            img, cap = cell
            x = gap + c * (tile + gap)
            canvas.paste(img.convert("RGB").resize((tile, tile)), (x, y))
            draw.multiline_text((x + 2, y + tile + 2), cap, fill="black", font=font, spacing=1)
        y += row_h
    canvas.save(out_path, quality=92)


# --------------------------------------------------------------------------------------
# Per folder
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def process_folder(folder: Path, out: Path, args, models):
    prior, decoder, img2img, negative_emb = models
    if (out / "synth_meta.json").exists() and not args.overwrite:
        print(f"[skip] {out} already exists (use --overwrite)")
        return None

    input_paths = list_inputs(folder)
    N = len(input_paths)
    if N < 2:
        print(f"[skip] {folder}: fewer than 2 input images")
        return None

    # resize to the generation size: used as img2img init images and for the overview
    inputs = [load_image(str(p)).convert("RGB").resize((args.size, args.size), Image.LANCZOS) for p in input_paths]
    recon = load_image(str(folder / "recon.jpeg")).convert("RGB") if (folder / "recon.jpeg").exists() else None

    # CLIP features of the decoded images (used by the measurements)
    f_inputs = [clip_feature(img, prior) for img in inputs]
    f_recon = clip_feature(recon, prior) if recon is not None else None

    # embeddings saved by the decomposition
    E = torch.load(folder / "embeds.pt", map_location="cpu")
    e = E["e"].float().to(DEVICE)
    e_inputs = E.get("e_inputs")
    if e_inputs is None:
        # folders written by a version that did not save e_inputs: use the re-encoded inputs.
        # These went through the Kandinsky -> CLIP round trip, so they are less accurate.
        print(f"[note] {folder}: no e_inputs in embeds.pt, using re-encoded input images instead")
        e_inputs = torch.stack(f_inputs)
        embedding_source = "reencoded_inputs"
    else:
        e_inputs = e_inputs.float().to(DEVICE)
        embedding_source = "embeds.pt"
    if e_inputs.shape[0] != N:
        print(f"[skip] {folder}: {N} input images but {e_inputs.shape[0]} input embeddings")
        return None

    # optional information from the decomposition (subject / keeper, bias similarity)
    info = {}
    if (folder / "groups.json").exists():
        with open(folder / "groups.json") as f:
            info = json.load(f)
    subject = info.get("subject") or {}
    noun, keeper = subject.get("noun"), subject.get("keeper_group")
    conds = info.get("conditions") or []
    cos_to_bias = [conds[i].get("cos_to_bias") if i < len(conds) else None for i in range(N)]

    mains = list(range(N)) if args.mains is None else [k for k in args.mains if 0 <= k < N]
    if not mains:
        print(f"[skip] {folder}: no valid main index in --mains for N = {N}")
        return None

    out.mkdir(parents=True, exist_ok=True)
    dtype = negative_emb.dtype

    # header row of the overview: recon and all inputs
    header = [(recon, "recon") if recon is not None else None]
    for i, img in enumerate(inputs):
        cap = f"input_{i}" + (f" [{noun}]" if (noun and i == keeper) else "")
        if cos_to_bias[i] is not None:
            cap += f"\ncos_to_bias={cos_to_bias[i]:.2f}"
        header.append((img, cap))
    rows = [header]

    meta_mains = []
    for k in mains:
        e_k = e_inputs[k]
        others = [i for i in range(N) if i != k]
        (out / f"main{k}").mkdir(exist_ok=True)
        rows.append(None)  # separator between blocks
        cells_meta = []
        for t in args.t:
            e_T = target_embedding(e, e_k, t)
            row = [(inputs[k], f"main = input_{k}\nt = {t:g}")]
            variants = [(s, f"s={s:g}") for s in args.strength]
            if args.embed_only:
                variants.append((None, "no init"))
            for s, label in variants:
                if s is None:
                    img = generate_plain(decoder, e_T.to(dtype), negative_emb, args.seed, args.steps,
                                         args.size, args.guidance_scale)
                    name = f"t{t:.2f}_noinit.jpeg"
                else:
                    img = generate_img2img(img2img, e_T.to(dtype), inputs[k], negative_emb, s, args.seed,
                                           args.steps, args.size, args.guidance_scale)
                    name = f"t{t:.2f}_s{s:.2f}.jpeg"
                img.save(out / f"main{k}" / name, quality=95)

                # measurements on re-encoded images
                f_t = clip_feature(img, prior)
                cos_main = cos(f_t, f_inputs[k])
                shifts = {i: cos(f_t - f_inputs[k], f_inputs[i] - f_inputs[k]) for i in others}
                shift_others = sum(shifts.values()) / len(shifts)
                cells_meta.append({
                    "t": t,
                    "strength": s,
                    "file": f"main{k}/{name}",
                    "denoising_steps": int(args.steps * s) if s is not None else args.steps,
                    "cos_eT_e": cos(e_T, e),
                    "cos_eT_ek": cos(e_T, e_k),
                    "cos_main": cos_main,
                    "cos_inputs": [cos(f_t, f) for f in f_inputs],
                    "cos_recon": cos(f_t, f_recon) if f_recon is not None else None,
                    "shift_toward": {str(i): v for i, v in shifts.items()},
                    "shift_others": shift_others,
                })
                row.append((img, f"{label}\nmain={cos_main:.2f} others={shift_others:+.2f}"))
            rows.append(row)
        meta_mains.append({"main": k, "is_keeper": (keeper is not None and k == keeper), "cells": cells_meta})

    title = (f"{folder.name}   e_T = (1-t) e + t e_main, img2img init = main input, larger s = further from it\n"
             "main = CLIP cos(target, main input); others = mean shift from the main input toward the other inputs")
    make_overview(rows, out / "overview.jpeg", title=title, tile=args.tile)

    meta = {
        "folder": str(folder),
        "n_inputs": N,
        "mains": mains,
        "t": args.t,
        "strength": args.strength,
        "embed_only_column": args.embed_only,
        "seed": args.seed,
        "steps": args.steps,
        "size": args.size,
        "guidance_scale": args.guidance_scale,
        "embedding_source": embedding_source,
        "subject": noun,
        "keeper": keeper,
        "cos_to_bias": cos_to_bias,
        "input_input_cos_reencoded": [[cos(a, b) for b in f_inputs] for a in f_inputs],
        "results": meta_mains,
    }
    with open(out / "synth_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return {"folder": str(folder), "output": str(out), "n_inputs": N, "mains": mains,
            "overview": str(out / "overview.jpeg")}


# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="synthesize_v2: designated-subject targets (interpolation + img2img grid)")
    p.add_argument("--root", default="hf_generate_source",
                   help="Directory with decomposition folders (or one such folder)")
    p.add_argument("--output-dir", default="output/synth/v2")
    p.add_argument("--t", type=parse_float_list, default=[0.0, 0.3, 0.6],
                   help="Comma-separated interpolation weights toward the main input (0 = source embedding)")
    p.add_argument("--strength", type=parse_float_list, default=[0.4, 0.7, 0.9],
                   help="Comma-separated img2img strengths in (0, 1]; larger = further from the main input")
    p.add_argument("--mains", type=parse_int_list, default=None,
                   help="Comma-separated input indices to designate as main (default: every input)")
    p.add_argument("--embed-only", action="store_true",
                   help="Add a reference column decoded from e_T alone (no init image)")
    p.add_argument("--seed", type=int, default=1, help="Kandinsky seed, shared by every cell")
    p.add_argument("--steps", type=int, default=50, help="Kandinsky steps (actual img2img steps = int(steps * strength))")
    p.add_argument("--size", type=int, default=768, help="Output resolution (decompose_v7_2.py uses 768)")
    p.add_argument("--guidance-scale", type=float, default=4.0, help="Decoder guidance scale (pipeline default 4.0)")
    p.add_argument("--tile", type=int, default=256, help="Tile size in the overview image")
    p.add_argument("--limit", type=int, default=None, help="Only process the first LIMIT folders")
    p.add_argument("--overwrite", action="store_true", help="Recompute folders that already have synth_meta.json")
    args = p.parse_args()

    if not args.t or not args.strength:
        raise ValueError("--t and --strength must not be empty.")
    for s in args.strength:
        if not (0.0 < s <= 1.0) or int(args.steps * s) < 1:
            raise ValueError(f"Invalid strength {s}: must be in (0, 1] and give at least one step.")

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    folders = find_folders(root)
    if args.limit is not None:
        folders = folders[: args.limit]
    print(f"Found {len(folders)} decomposition folder(s) under {root}")
    if not folders:
        raise ValueError("No folders with embeds.pt and input_0.jpeg found.")

    prior, decoder, negative_emb = load_kandinsky()
    # img2img shares the decoder's modules: no additional weights are loaded
    img2img = KandinskyV22Img2ImgPipeline(unet=decoder.unet, scheduler=decoder.scheduler, movq=decoder.movq)
    decoder.set_progress_bar_config(disable=True)
    img2img.set_progress_bar_config(disable=True)
    models = (prior, decoder, img2img, negative_emb)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for folder in tqdm(folders, desc="folders"):
        rel = folder.relative_to(root)
        out = out_root / (rel if str(rel) != "." else folder.name)
        res = process_folder(folder, out, args, models)
        if res is not None:
            summary.append(res)

    if summary:
        with open(out_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        n_targets = sum(len(s["mains"]) for s in summary) * len(args.t) * (len(args.strength) + int(args.embed_only))
        print(f"\n{len(summary)} folder(s), {n_targets} target(s). Summary: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
