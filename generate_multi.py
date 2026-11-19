"""
generate_multi.py -- combine 2-4 reference images with FLUX.1 Kontext, optionally with a main image.
Last modified: 2026-11-18 (paths without the hf_ prefix: generate_source, samples/sources, output/decompose/v72_final)

Inputs are given either as a list of images (--images) or as a decomposition folder
(--folder, uses its input_0.jpeg, input_1.jpeg, ...). --main k (1-based, in the given order)
designates image k as the main object; without --main no image is designated.

How the main image is designated (--role, must match the training run):
    embed  the learnable role embedding E_main is added to the main image's tokens
           (load it with --role-embedding; without it E_main is zero and only the prompt
           differs between modes)
    slot   the main image is moved to the first position
    index  the main image is named in the prompt ("image k")

The prompt is always one of the fixed templates in multiref/prompts.py.

Usage
-----
    # base model, no fine-tuning (baseline)
    python generate_multi.py --folder generate_source/<name> --main 2

    # fine-tuned LoRA + role embedding
    python generate_multi.py --images a.jpg b.jpg c.jpg --main 1 \\
        --lora runs/<run>/lora --role-embedding runs/<run>/role_embedding.safetensors

Output (--output-dir): <prefix>__main{k|none}__seed{s}.jpeg and a preview with the inputs
(main image marked) next to the result, plus <prefix>__main{k|none}.json with the prompt
and the model order of the inputs.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from multiref.conditioning import LAYOUTS, MainRoleEmbedding, MultiRefKontextPipeline, load_rgb
from multiref.data import list_inputs
from multiref.prompts import ROLE_MODES


def parse_args():
    p = argparse.ArgumentParser(description="Combine 2-4 images with FLUX.1 Kontext, optionally with a main image.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", type=Path, nargs="+", help="2-4 input images")
    src.add_argument("--folder", type=Path, help="Decomposition folder with input_0.jpeg, input_1.jpeg, ...")
    p.add_argument("--main", type=int, default=None, help="Main image, 1-based in the given order (default: none)")
    p.add_argument("--role", choices=ROLE_MODES, default="embed", help="How the main image is designated")
    p.add_argument("--layout", choices=LAYOUTS, default="canvas", help="Position ids of the references")
    p.add_argument("--base-model", default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--lora", default=None, help="LoRA weights (local path or Hugging Face repo)")
    p.add_argument("--role-embedding", type=Path, default=None, help="role_embedding.safetensors (role 'embed')")
    p.add_argument("--ref-size", type=int, default=512, help="Resolution of each reference image")
    p.add_argument("--size", type=int, default=1024, help="Output resolution (square); use the training target size")
    p.add_argument("--steps", type=int, default=28, help="Denoising steps")
    p.add_argument("--guidance-scale", type=float, default=3.5, help="Kontext guidance (use the training value)")
    p.add_argument("--prompt", default=None, help="Override the template prompt (for experiments only)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-seeds", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/multi"))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def make_preview(inputs, main, output, prompt, tile=256):
    """Inputs in one row (main image framed in red) above the output."""
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    gap, cap = 8, 22
    n = len(inputs)
    out_size = max(tile * 2, min(n * tile + (n - 1) * gap, 768))
    W = max(n * tile + (n + 1) * gap, out_size + 2 * gap)
    H = gap + tile + cap + gap + out_size + gap + 40
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    for i, im in enumerate(inputs):
        x = gap + i * (tile + gap)
        canvas.paste(im.resize((tile, tile)), (x, gap))
        label = f"image {i + 1}"
        if main == i:
            draw.rectangle([x - 3, gap - 3, x + tile + 2, gap + tile + 2], outline=(220, 30, 30), width=4)
            label += " (main)"
        draw.text((x + 2, gap + tile + 3), label, fill="black", font=font)
    y = gap + tile + cap + gap
    canvas.paste(output.resize((out_size, out_size)), ((W - out_size) // 2, y))
    draw.multiline_text((gap, y + out_size + 6), prompt, fill="black", font=font)
    return canvas


def main():
    args = parse_args()

    paths = list_inputs(args.folder) if args.folder else list(args.images)
    if not 2 <= len(paths) <= 4:
        raise ValueError(f"Expected 2-4 input images, got {len(paths)}.")
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    main_idx = None
    if args.main is not None:
        if not 1 <= args.main <= len(paths):
            raise ValueError(f"--main must be between 1 and {len(paths)}.")
        main_idx = args.main - 1
    images = [load_rgb(p) for p in paths]

    print("Loading FLUX.1 Kontext ...")
    pipe = MultiRefKontextPipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    if args.lora:
        print(f"Loading LoRA: {args.lora}")
        pipe.load_lora_weights(args.lora)
    pipe.to("cuda")

    role = None
    if args.role == "embed":
        role = MainRoleEmbedding(pipe.transformer.inner_dim).to("cuda")
        if args.role_embedding:
            role.load(args.role_embedding)
        elif main_idx is not None:
            print("[note] no --role-embedding: E_main is zero, the main image is designated by the prompt only")
        role.attach(pipe.transformer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.folder.name if args.folder else "_x_".join(Path(p).stem for p in paths)
    tag = f"{prefix}__main{args.main if args.main is not None else 'none'}"

    info = None
    for i in range(args.num_seeds):
        seed = args.seed + i
        out_path = args.output_dir / f"{tag}__seed{seed:03d}.jpeg"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path.name} exists")
            continue
        generator = torch.Generator(device="cuda").manual_seed(seed)
        out, info = pipe.generate_multi(
            images, main=main_idx, role_mode=args.role, layout=args.layout, role_embedding=role,
            ref_size=args.ref_size, height=args.size, width=args.size, prompt=args.prompt,
            num_inference_steps=args.steps, guidance_scale=args.guidance_scale, generator=generator,
        )
        result = out.images[0]
        result.save(out_path, quality=95)
        make_preview(images, main_idx, result, info["prompt"]).save(
            args.output_dir / f"{tag}__seed{seed:03d}_preview.jpeg", quality=92)
        print(f"[{i + 1}/{args.num_seeds}] saved {out_path.name}")

    if info is not None:
        with open(args.output_dir / f"{tag}.json", "w") as f:
            json.dump({"inputs": [str(p) for p in paths], "main": args.main, "role": args.role,
                       "layout": args.layout, "lora": args.lora,
                       "role_embedding": str(args.role_embedding) if args.role_embedding else None,
                       **info}, f, indent=2)


if __name__ == "__main__":
    main()
