"""
synthesize_v2_1.py (v2.1) -- designated-subject targets with an automatic 2 x 2 grid per main.
Last modified: 2026-11-18 (default --root hf_generate_source -> generate_source)
               2026-11-17 (default --size 768 -> 1024, the target size of the original Inspiration Seeds)

What changed in v2.1 (compared with synthesize_v2.py)
----------------------------------------------------
synthesize_v2.py ran the same 3 x 3 grid (t x img2img strength) for every designated input.
Hand-picked cells on 14 folders showed that the suitable cell depends on how close the
main input already is to the source, so v2.1 classifies every input before generating and
runs only a 2 x 2 grid chosen by that class.

* Closeness (new, computed before any generation):
      closeness_k = cos(f(input_k.jpeg), f(recon.jpeg))
  with f = CLIP image encoder of the Kandinsky prior. This is an IMAGE-level measure. The
  embedding-level cosine cos(e, e_k) is NOT used: it does not separate the inputs (e.g. three
  inputs with cos(e, e_k) = 0.84 each had image-level similarities 0.37 / 0.76 / 0.47).
  On the 14 analysed folders the input closest to the source ("dominant" input, usually the
  v7.2 keeper) had 0.61-0.85, the other inputs 0.26-0.68.
* Class and grid (new):
      close : closeness_k >= --close-threshold (0.6)   t in --close-t (0, 0.3)   s in --close-s (0.7, 0.9)
      far   : otherwise                                  t in --far-t  (0.3, 0.6)  s in --far-s  (0.4, 0.7)
  Reason: for a close main, t = 0 with a large strength already gives the wanted result
  (the target is close to the reconstruction); for a far main, t = 0 has no visible effect
  at strength >= 0.7, and a smaller strength is needed to keep the main at all.
  The defaults contain 6 of the 7 hand-picked cells. They are provisional (7 labels).
* Measurements (changed). Per cell, on re-encoded images:
      cos_main = cos(f(target), f(input_k))
      lift     = cos_main - closeness_k
                 how much more the target resembles the main than the plain reconstruction
                 does (about 0 = the designation had no effect)
      margin   = cos_main - max_{i != k} cos(f(target), f(input_i))
                 < 0 = the target still resembles another input more than the main
  Observed on the labelled cells: cos_main 0.52-0.78 (never above 0.8; higher values are
  near copies of the main), margin down to -0.15. These are recorded, not used as a filter.
  shift_toward (v2) is still written to synth_meta.json; the mean over the other inputs is
  no longer shown because it carried no information beyond cos_main.
* recon.jpeg is now required for the closeness. If it is missing, it is decoded from e
  with the settings of this run and saved in the output folder.

Method (unchanged from v2)
--------------------------
For one decomposition folder written by decompose_v7_2.py (N = 2-4 inputs) and every input k
that is designated as the main subject:
  1. e_T = (1 - t) * e + t * e_k, rescaled to ||e||
     (e = embeds.pt["e"], e_k = embeds.pt["e_inputs"][k]; t = 0 source, t = 1 input k).
  2. Kandinsky 2.2 img2img: init image = input_k.jpeg, condition = e_T.
     `strength` is the diffusers img2img strength: LARGER = FURTHER from the init image
     (0.4 keeps outline and colors, 0.7 the rough layout, 0.9 almost nothing).
     Denoising steps actually run = int(steps * strength).
  3. Same seed for every cell.

Output
------
    <output-dir>/<name>/overview.jpeg             one overview per folder
    <output-dir>/<name>/main{k}/t{t}_s{s}.jpeg    individual targets
    <output-dir>/<name>/synth_meta.json           closeness, class, grid and measurements
    <output-dir>/summary.json                     one entry per processed folder

Overview layout: first row = recon and all inputs (with closeness and class); then one block
per designated input, one row per t (first cell = the main input), one column per strength.

Usage
-----
    # default: class-dependent 2 x 2 grid for every input of every folder
    python synthesize_v2_1.py --root generate_source --output-dir output/synth/v2_1

    # other threshold / grids
    python synthesize_v2_1.py --close-threshold 0.65 --far-s 0.4,0.55,0.7

    # treat every input as "far" (threshold above 1) or as "close" (threshold 0)
    python synthesize_v2_1.py --close-threshold 1.1
    python synthesize_v2_1.py --close-threshold 0

Notes
-----
* Models and loading are the same as in decompose.py / decompose_v7_2.py; the img2img
  pipeline reuses the decoder's unet, scheduler and movq. The SAE is not needed.
* Resolution: targets are generated at --size (1024 by default, like the real source images
  that serve as targets of unspecified samples). The decomposed inputs are 768 px
  (decompose_v7_2.py --size 768); they are upsampled to --size only to serve as img2img
  init images. recon.jpeg is used only through CLIP, so its resolution does not matter.
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
    dtype = negative_emb.dtype

    # resize to the generation size: used as img2img init images and for the overview
    inputs = [load_image(str(p)).convert("RGB").resize((args.size, args.size), Image.LANCZOS) for p in input_paths]
    f_inputs = [clip_feature(img, prior) for img in inputs]

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

    mains = list(range(N)) if args.mains is None else [k for k in args.mains if 0 <= k < N]
    if not mains:
        print(f"[skip] {folder}: no valid main index in --mains for N = {N}")
        return None
    out.mkdir(parents=True, exist_ok=True)

    # reconstruction of the source: reference for the closeness
    if (folder / "recon.jpeg").exists():
        recon = load_image(str(folder / "recon.jpeg")).convert("RGB")
        recon_source = "decomposition folder"
    else:
        print(f"[note] {folder}: recon.jpeg missing, decoding it from e")
        recon = generate_plain(decoder, e.to(dtype), negative_emb, args.seed, args.steps, args.size, args.guidance_scale)
        recon.save(out / "recon.jpeg", quality=95)
        recon_source = "decoded in this run"
    f_recon = clip_feature(recon, prior)

    # closeness of every input to the source (image level) and its class
    closeness = [cos(f, f_recon) for f in f_inputs]
    classes = ["close" if c >= args.close_threshold else "far" for c in closeness]
    dominant = max(range(N), key=lambda i: closeness[i])

    # optional information from the decomposition (subject / keeper, bias similarity)
    info = {}
    if (folder / "groups.json").exists():
        with open(folder / "groups.json") as f:
            info = json.load(f)
    subject = info.get("subject") or {}
    noun, keeper = subject.get("noun"), subject.get("keeper_group")
    conds = info.get("conditions") or []
    cos_to_bias = [conds[i].get("cos_to_bias") if i < len(conds) else None for i in range(N)]

    # header row of the overview: recon and all inputs
    header = [(recon, "recon")]
    for i, img in enumerate(inputs):
        cap = f"input_{i}" + (f" [{noun}]" if (noun and i == keeper) else "")
        cap += f"\nclose={closeness[i]:.2f} ({classes[i]})"
        header.append((img, cap))
    rows = [header]

    meta_mains = []
    for k in mains:
        e_k = e_inputs[k]
        others = [i for i in range(N) if i != k]
        t_list, s_list = (args.close_t, args.close_s) if classes[k] == "close" else (args.far_t, args.far_s)
        (out / f"main{k}").mkdir(exist_ok=True)
        rows.append(None)  # separator between blocks
        cells_meta = []
        for t in t_list:
            e_T = target_embedding(e, e_k, t)
            row = [(inputs[k], f"main = input_{k} ({classes[k]})\nt = {t:g}")]
            variants = [(s, f"s={s:g}") for s in s_list]
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
                cos_inputs = [cos(f_t, f) for f in f_inputs]
                cos_main = cos_inputs[k]
                lift = cos_main - closeness[k]
                margin = cos_main - max(cos_inputs[i] for i in others)
                cells_meta.append({
                    "t": t,
                    "strength": s,
                    "file": f"main{k}/{name}",
                    "denoising_steps": int(args.steps * s) if s is not None else args.steps,
                    "cos_eT_e": cos(e_T, e),
                    "cos_eT_ek": cos(e_T, e_k),
                    "cos_main": cos_main,
                    "lift": lift,
                    "margin": margin,
                    "cos_inputs": cos_inputs,
                    "cos_recon": cos(f_t, f_recon),
                    "shift_toward": {str(i): cos(f_t - f_inputs[k], f_inputs[i] - f_inputs[k]) for i in others},
                })
                row.append((img, f"{label}  main={cos_main:.2f}\nlift={lift:+.2f} margin={margin:+.2f}"))
            rows.append(row)
        meta_mains.append({
            "main": k,
            "class": classes[k],
            "closeness": closeness[k],
            "is_dominant": k == dominant,
            "is_keeper": (keeper is not None and k == keeper),
            "t": t_list,
            "strength": s_list,
            "cells": cells_meta,
        })

    title = (f"{folder.name}   e_T = (1-t) e + t e_main, img2img init = main input, larger s = further from it\n"
             "close = cos(input, recon); main = cos(target, main); lift = main - close; margin = main - best other input")
    make_overview(rows, out / "overview.jpeg", title=title, tile=args.tile)

    meta = {
        "folder": str(folder),
        "n_inputs": N,
        "mains": mains,
        "close_threshold": args.close_threshold,
        "close_grid": {"t": args.close_t, "strength": args.close_s},
        "far_grid": {"t": args.far_t, "strength": args.far_s},
        "closeness": closeness,
        "classes": classes,
        "dominant": dominant,
        "recon_source": recon_source,
        "embed_only_column": args.embed_only,
        "seed": args.seed,
        "steps": args.steps,
        "size": args.size,
        "guidance_scale": args.guidance_scale,
        "embedding_source": embedding_source,
        "subject": noun,
        "keeper": keeper,
        "cos_to_bias": cos_to_bias,
        "cos_e_ek": [cos(e, e_inputs[i]) for i in range(N)],
        "input_input_cos_reencoded": [[cos(a, b) for b in f_inputs] for a in f_inputs],
        "results": meta_mains,
    }
    with open(out / "synth_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{folder.name}: closeness " + ", ".join(f"{c:.2f} ({cl})" for c, cl in zip(closeness, classes))
          + f"; dominant input_{dominant}" + (f", keeper input_{keeper}" if keeper is not None else ""))
    return {"folder": str(folder), "output": str(out), "n_inputs": N, "mains": mains,
            "closeness": closeness, "classes": classes, "dominant": dominant, "keeper": keeper,
            "n_targets": sum(len(m["cells"]) for m in meta_mains), "overview": str(out / "overview.jpeg")}


# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="synthesize_v2.1: designated-subject targets, class-dependent 2 x 2 grid")
    p.add_argument("--root", default="generate_source",
                   help="Directory with decomposition folders (or one such folder)")
    p.add_argument("--output-dir", default="output/synth/v2_1")
    p.add_argument("--close-threshold", type=float, default=0.6,
                   help="An input is 'close' if cos(input, recon) >= this value, otherwise 'far' "
                        "(0 = every input close, > 1 = every input far)")
    p.add_argument("--close-t", type=parse_float_list, default=[0.0, 0.3],
                   help="Interpolation weights toward the main input for close mains")
    p.add_argument("--close-s", type=parse_float_list, default=[0.7, 0.9],
                   help="img2img strengths for close mains (larger = further from the main input)")
    p.add_argument("--far-t", type=parse_float_list, default=[0.3, 0.6],
                   help="Interpolation weights toward the main input for far mains")
    p.add_argument("--far-s", type=parse_float_list, default=[0.4, 0.7],
                   help="img2img strengths for far mains")
    p.add_argument("--mains", type=parse_int_list, default=None,
                   help="Comma-separated input indices to designate as main (default: every input)")
    p.add_argument("--embed-only", action="store_true",
                   help="Add a reference column decoded from e_T alone (no init image)")
    p.add_argument("--seed", type=int, default=1, help="Kandinsky seed, shared by every cell")
    p.add_argument("--steps", type=int, default=50, help="Kandinsky steps (actual img2img steps = int(steps * strength))")
    p.add_argument("--size", type=int, default=1024,
                   help="Target resolution (1024 = size of the real source images; inputs are upsampled to it "
                        "as img2img init images)")
    p.add_argument("--guidance-scale", type=float, default=4.0, help="Decoder guidance scale (pipeline default 4.0)")
    p.add_argument("--tile", type=int, default=256, help="Tile size in the overview image")
    p.add_argument("--limit", type=int, default=None, help="Only process the first LIMIT folders")
    p.add_argument("--overwrite", action="store_true", help="Recompute folders that already have synth_meta.json")
    args = p.parse_args()

    for name in ("close_t", "close_s", "far_t", "far_s"):
        if not getattr(args, name):
            raise ValueError(f"--{name.replace('_', '-')} must not be empty.")
    for s in args.close_s + args.far_s:
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
        cls = [s["classes"][k] for s in summary for k in s["mains"]]
        val = sorted(s["closeness"][k] for s in summary for k in s["mains"])
        agree = [s["dominant"] == s["keeper"] for s in summary if s["keeper"] is not None]
        print(f"\n{len(summary)} folder(s), {sum(s['n_targets'] for s in summary)} target(s).")
        print(f"designated inputs: {cls.count('close')} close, {cls.count('far')} far "
              f"(threshold {args.close_threshold:g}); closeness min / median / max = "
              f"{val[0]:.2f} / {val[len(val) // 2]:.2f} / {val[-1]:.2f}")
        if agree:
            print(f"dominant input == keeper in {sum(agree)} / {len(agree)} folder(s)")
        print(f"summary: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
