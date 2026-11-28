"""
cache_features.py -- precompute VAE latents and prompt embeddings for multi-reference fine-tuning.
Last modified: 2026-11-26 (--ref-augs: augmented versions of every input for reference augmentation)
               2026-11-16

With everything cached, the training script only needs the transformer on the GPU.

* VAE latents (FLUX.1 Kontext VAE, mode of the latent distribution, normalized like the pipeline):
    every input image at --ref-size (512 -> [16, 64, 64]) and every target at --target-size
    (1024 -> [16, 128, 128]), as in the original Inspiration Seeds (512 px inputs, 1024 px
    targets). The 768 px decomposed inputs are downscaled to --ref-size here; images of any
    other size are resized as well, so all inputs and all targets have one resolution each.
    Non-square images are center-cropped first.
    Stored as <cache-dir>/latents/<key>.pt (bfloat16), key = multiref.data.cache_key(path, size).
* Prompt embeddings: every template of multiref/prompts.py (T5 + CLIP, max_sequence_length 512),
    stored in <cache-dir>/prompts.pt as {key: {prompt, prompt_embeds, pooled_prompt_embeds, text_ids}}.
* <cache-dir>/config.json: base model, resolutions and prompt templates. The training script checks
    it against its own settings.
* Augmented references (--ref-augs N, default 0 = none): N extra latents per INPUT image (targets
    are never augmented), stored as <cache-dir>/latents/<key>_aug<v>.pt, v = 0 .. N-1. Each version
    is made from the square center crop of the image (as for the original latent):
        v = 0      horizontal flip of the whole image (always present, the safest augmentation)
        v >= 1     random square crop with area in [--aug-crop-min, 1] of the image at a random
                   position, horizontally flipped with probability --aug-flip-prob
    then resized to --ref-size and encoded like the original. No color changes: an attribute
    main (color, texture, style) must survive the augmentation. The parameters of version v depend
    only on (--aug-seed, image path, v), so adding versions later (larger N) keeps the existing ones.
    train_multiref.py --ref-aug-prob uses them; config.json records "ref_augs" and "aug".
    --aug-preview K writes <cache-dir>/aug_preview.jpeg (original + every version for K inputs) to
    check that the crops keep the subject.

Existing latent files are skipped (use --overwrite to recompute), so the script can be rerun
after the manifests grow.

Usage
-----
    python cache_features.py --manifest-dir data/multiref --cache-dir data/multiref/cache

    # add 4 augmented versions per input to an existing cache (original latents are skipped)
    python cache_features.py --manifest-dir data/multiref --ref-augs 4 --aug-preview 8
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from diffusers import FluxKontextPipeline
from PIL import Image, ImageOps
from tqdm import tqdm

from multiref.conditioning import center_crop_square, encode_image, load_rgb
from multiref.data import cache_file, read_jsonl
from multiref.prompts import all_prompts
from multiref.train_data import aug_cache_file


# --------------------------------------------------------------------------------------
# Reference augmentation
# --------------------------------------------------------------------------------------
def aug_params(path, v: int, seed: int, crop_min: float, flip_prob: float) -> dict:
    """Deterministic parameters of augmented version v of one image (fractions of the square side)."""
    if v == 0:
        return {"flip": True, "scale": 1.0, "x": 0.0, "y": 0.0}
    h = hashlib.sha1(f"{seed}|{Path(path).resolve()}|{v}".encode()).hexdigest()
    rng = random.Random(int(h[:16], 16))
    area = rng.uniform(crop_min, 1.0)
    scale = math.sqrt(area)                     # side of the crop relative to the square side
    return {"flip": rng.random() < flip_prob, "scale": scale,
            "x": rng.uniform(0.0, 1.0 - scale), "y": rng.uniform(0.0, 1.0 - scale)}


def apply_aug(image: Image.Image, params: dict) -> Image.Image:
    """Square center crop, then the crop / flip of params (the result is resized by encode_image)."""
    image = center_crop_square(image.convert("RGB"))
    s = image.size[0]
    side = max(1, round(params["scale"] * s))
    left, top = round(params["x"] * s), round(params["y"] * s)
    left, top = min(left, s - side), min(top, s - side)
    if side < s:
        image = image.crop((left, top, left + side, top + side))
    if params["flip"]:
        image = ImageOps.mirror(image)
    return image


def save_aug_preview(paths, n_augs, args, out_path, tile=192):
    """One row per input: original (square center crop), then every augmented version."""
    rows = []
    for path in paths:
        orig = center_crop_square(load_rgb(path))
        ims = [orig] + [apply_aug(orig, aug_params(path, v, args.aug_seed, args.aug_crop_min, args.aug_flip_prob))
                        for v in range(n_augs)]
        rows.append(ims)
    if not rows:
        return
    sheet = Image.new("RGB", ((n_augs + 1) * tile, len(rows) * tile), "white")
    for r, ims in enumerate(rows):
        for c, im in enumerate(ims):
            sheet.paste(im.resize((tile, tile)), (c * tile, r * tile))
    sheet.save(out_path, quality=90)
    print(f"augmentation preview: {out_path}")


def check_aug_config(cache_dir, args):
    """Augmentation settings to record in config.json; refuse to mix versions made with other settings."""
    old = {}
    if (cache_dir / "config.json").exists():
        with open(cache_dir / "config.json") as f:
            old = json.load(f)
    old_n, old_aug = int(old.get("ref_augs", 0)), old.get("aug")
    new_aug = {"crop_min": args.aug_crop_min, "flip_prob": args.aug_flip_prob, "seed": args.aug_seed}
    if args.ref_augs <= 0:
        return old_n, old_aug                   # keep what is already in the cache
    if old_n > 0 and old_aug != new_aug and not args.overwrite:
        raise ValueError(f"The cache already has {old_n} augmented versions made with {old_aug}; "
                         f"use the same --aug-* settings or --overwrite (requested {new_aug})")
    return args.ref_augs, new_aug


def main():
    p = argparse.ArgumentParser(description="Cache VAE latents and prompt embeddings")
    p.add_argument("--manifest-dir", type=Path, default=Path("data/multiref"),
                   help="Folder with train.jsonl and val.jsonl from build_dataset.py")
    p.add_argument("--cache-dir", type=Path, default=None, help="Default: <manifest-dir>/cache")
    p.add_argument("--base-model", default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--ref-size", type=int, default=512, help="Resolution of each reference image")
    p.add_argument("--target-size", type=int, default=1024, help="Resolution of every target image")
    p.add_argument("--max-sequence-length", type=int, default=512, help="T5 sequence length")
    p.add_argument("--overwrite", action="store_true")
    g = p.add_argument_group("reference augmentation")
    g.add_argument("--ref-augs", type=int, default=0,
                   help="Augmented versions per input image (0 = keep the cache's current setting)")
    g.add_argument("--aug-crop-min", type=float, default=0.6,
                   help="Smallest crop area as a fraction of the square image (versions v >= 1)")
    g.add_argument("--aug-flip-prob", type=float, default=0.5, help="Flip probability of the versions v >= 1")
    g.add_argument("--aug-seed", type=int, default=0)
    g.add_argument("--aug-preview", type=int, default=0, help="Inputs shown in <cache-dir>/aug_preview.jpeg")
    args = p.parse_args()
    if not 0.0 < args.aug_crop_min <= 1.0:
        raise ValueError("--aug-crop-min must be in (0, 1]")
    cache_dir = args.cache_dir or args.manifest_dir / "cache"
    n_augs, aug_cfg = check_aug_config(cache_dir, args)

    rows = []
    for name in ("train", "val"):
        path = args.manifest_dir / f"{name}.jsonl"
        if path.exists():
            rows += read_jsonl(path)
    if not rows:
        raise FileNotFoundError(f"No train.jsonl / val.jsonl in {args.manifest_dir}")

    # every (image, resolution) pair, without duplicates (inputs are shared by several samples)
    jobs = {}
    for r in rows:
        for x in r["inputs"]:
            jobs[(str(Path(x).resolve()), args.ref_size)] = x
        for x in r["targets"]:
            jobs[(str(Path(x).resolve()), args.target_size)] = x
    todo = [(path, size) for (_, size), path in jobs.items()
            if args.overwrite or not cache_file(cache_dir, path, size).exists()]
    print(f"{len(rows)} samples, {len(jobs)} image latents ({len(jobs) - len(todo)} already cached)")

    # augmented versions of the inputs (only computed with --ref-augs > 0)
    inputs = {}
    for r in rows:
        for x in r["inputs"]:
            inputs.setdefault(str(Path(x).resolve()), x)
    aug_todo = []
    if args.ref_augs > 0:
        aug_todo = [(x, v) for x in inputs.values() for v in range(args.ref_augs)
                    if args.overwrite or not aug_cache_file(cache_dir, x, args.ref_size, v).exists()]
        n_all = len(inputs) * args.ref_augs
        print(f"augmented references: {len(inputs)} inputs x {args.ref_augs} versions "
              f"({n_all - len(aug_todo)} already cached)")

    pipe = FluxKontextPipeline.from_pretrained(args.base_model, transformer=None, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    (cache_dir / "latents").mkdir(parents=True, exist_ok=True)

    missing = []
    for path, size in tqdm(todo, desc="latents"):
        if not Path(path).exists():
            missing.append(path)
            continue
        lat = encode_image(pipe.vae, pipe.image_processor, load_rgb(path), size, "cuda", pipe.vae.dtype)
        torch.save(lat[0].to(torch.bfloat16).cpu(), cache_file(cache_dir, path, size))
    for path, v in tqdm(aug_todo, desc="augmented refs"):
        if not Path(path).exists():
            missing.append(path)
            continue
        params = aug_params(path, v, args.aug_seed, args.aug_crop_min, args.aug_flip_prob)
        image = apply_aug(load_rgb(path), params)
        lat = encode_image(pipe.vae, pipe.image_processor, image, args.ref_size, "cuda", pipe.vae.dtype)
        torch.save(lat[0].to(torch.bfloat16).cpu(), aug_cache_file(cache_dir, path, args.ref_size, v))
    if missing:
        missing = sorted(set(missing))
        print(f"[warn] {len(missing)} image(s) not found, e.g. {missing[0]}")
    if args.aug_preview > 0 and n_augs > 0:
        preview = [x for x in list(inputs.values())[: args.aug_preview] if Path(x).exists()]
        save_aug_preview(preview, n_augs, args if args.ref_augs > 0 else _args_from(aug_cfg, args),
                         cache_dir / "aug_preview.jpeg")

    prompts = {}
    with torch.no_grad():
        for key, text in all_prompts().items():
            pe, pooled, text_ids = pipe.encode_prompt(prompt=text, prompt_2=None, device="cuda",
                                                      num_images_per_prompt=1,
                                                      max_sequence_length=args.max_sequence_length)
            prompts[key] = {"prompt": text, "prompt_embeds": pe[0].to(torch.bfloat16).cpu(),
                            "pooled_prompt_embeds": pooled[0].to(torch.bfloat16).cpu(),
                            "text_ids": text_ids.float().cpu()}
    torch.save(prompts, cache_dir / "prompts.pt")

    config = {"base_model": args.base_model, "ref_size": args.ref_size, "target_size": args.target_size,
              "max_sequence_length": args.max_sequence_length,
              "prompts": {k: v["prompt"] for k, v in prompts.items()}, "n_latents": len(jobs),
              "ref_augs": n_augs, "aug": aug_cfg}
    with open(cache_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"cache: {cache_dir} ({len(prompts)} prompt templates)")


def _args_from(aug_cfg, args):
    """Copy of args with the augmentation settings stored in the cache (preview without --ref-augs)."""
    ns = argparse.Namespace(**vars(args))
    ns.aug_crop_min, ns.aug_flip_prob, ns.aug_seed = aug_cfg["crop_min"], aug_cfg["flip_prob"], aug_cfg["seed"]
    return ns


if __name__ == "__main__":
    main()
