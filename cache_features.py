"""
cache_features.py -- precompute VAE latents and prompt embeddings for multi-reference fine-tuning.
Last modified: 2026-11-16

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

Existing latent files are skipped (use --overwrite to recompute), so the script can be rerun
after the manifests grow.

Usage
-----
    python cache_features.py --manifest-dir data/multiref --cache-dir data/multiref/cache
"""

import argparse
import json
from pathlib import Path

import torch
from diffusers import FluxKontextPipeline
from tqdm import tqdm

from multiref.conditioning import encode_image, load_rgb
from multiref.data import cache_file, read_jsonl
from multiref.prompts import all_prompts


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
    args = p.parse_args()
    cache_dir = args.cache_dir or args.manifest_dir / "cache"

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
    if missing:
        print(f"[warn] {len(missing)} image(s) not found, e.g. {missing[0]}")

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
              "prompts": {k: v["prompt"] for k, v in prompts.items()}, "n_latents": len(jobs)}
    with open(cache_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"cache: {cache_dir} ({len(prompts)} prompt templates)")


if __name__ == "__main__":
    main()
