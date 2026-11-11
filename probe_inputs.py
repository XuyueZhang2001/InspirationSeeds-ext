"""
Probe whether the trained Inspiration Seeds model actually uses both input images.

Runs the original model (FLUX.1 Kontext dev + released LoRA, same grid and prompt as
generate.py) on several input conditions with the SAME seeds, then compares the outputs:

  AB       : (A, B)            normal input
  BA       : (B, A)            positions swapped          -> position sensitivity
  AA       : (A, A)            B replaced by A            -> how much B matters
  BB       : (B, B)            A replaced by B            -> how much A matters
  A_blank  : (A, white)        B removed
  blank_B  : (white, B)        A removed
  AX, XB   : (A, X), (X, B)    only with --alt X: one input replaced by another image

Metrics (DINOv2 and CLIP image features, cosine similarity, averaged over seeds):
  influence_of_B = 1 - sim(out_AB, out_AA)   large -> B changes the output
  influence_of_A = 1 - sim(out_AB, out_BB)   large -> A changes the output
  balance        = influence_of_A / (influence_of_A + influence_of_B)
                   ~0.5 both used; near 1 -> mostly A; near 0 -> mostly B
  position       = sim(out_AB, out_BA)       high -> insensitive to position
  sim_to_A/B     = sim(out_AB, A), sim(out_AB, B)

Usage:
  python probe_inputs.py \
    --image-a hf_decomposed/original/14_topk_32_frac_0.5/step_-1.jpeg \
    --image-b hf_decomposed/original/14_topk_32_frac_0.5/step_1.jpeg \
    --output-dir probe/14
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image, ImageDraw, ImageFont

from generate import DEFAULT_PROMPT, create_input_grid, load_lora_weights


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------
def build_conditions(img_a, img_b, img_x):
    white = Image.new("RGB", (512, 512), "white")
    conds = {
        "AB": (img_a, img_b),
        "BA": (img_b, img_a),
        "AA": (img_a, img_a),
        "BB": (img_b, img_b),
        "A_blank": (img_a, white),
        "blank_B": (white, img_b),
    }
    if img_x is not None:
        conds["AX"] = (img_a, img_x)
        conds["XB"] = (img_x, img_b)
    return conds


@torch.inference_mode()
def run_generation(args, conditions, out_dir: Path):
    print("Loading FLUX Kontext pipeline...")
    pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
    if args.lora_path:
        load_lora_weights(pipe, args.lora_path)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    for name, (im1, im2) in conditions.items():
        cdir = out_dir / name
        cdir.mkdir(parents=True, exist_ok=True)
        grid = create_input_grid(im1, im2)
        grid.save(cdir / "input_grid.jpeg")
        for seed in args.seeds:
            path = cdir / f"seed_{seed:03d}.jpeg"
            if path.exists() and not args.overwrite:
                print(f"[skip] {path}")
                continue
            print(f"{name}: seed {seed}")
            generator = torch.Generator(device="cuda").manual_seed(seed)
            img = pipe(image=grid, prompt=DEFAULT_PROMPT, guidance_scale=args.guidance_scale,
                       generator=generator).images[0]
            img.save(path)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
class Dino:
    name = "dino"

    def __init__(self):
        from transformers import AutoImageProcessor, AutoModel

        self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to("cuda").eval()

    @torch.inference_mode()
    def __call__(self, img):
        x = self.proc(images=img.convert("RGB"), return_tensors="pt").to("cuda")
        return torch.nn.functional.normalize(self.model(**x).pooler_output[0].float(), dim=0)


class Clip:
    name = "clip"

    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor

        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").eval()

    @torch.inference_mode()
    def __call__(self, img):
        x = self.proc(images=img.convert("RGB"), return_tensors="pt").to("cuda")
        return torch.nn.functional.normalize(self.model.get_image_features(**x)[0].float(), dim=0)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def compute_metrics(args, conditions, out_dir: Path, img_a, img_b):
    extractors = [Dino(), Clip()]
    results = {}
    for ex in extractors:
        f_a, f_b = ex(img_a), ex(img_b)
        feats = {name: {s: ex(Image.open(out_dir / name / f"seed_{s:03d}.jpeg")) for s in args.seeds}
                 for name in conditions}
        sim = lambda u, v: float(u @ v)

        per_seed = []
        for s in args.seeds:
            ab = feats["AB"][s]
            row = {
                "seed": s,
                "sim_AB_AA": sim(ab, feats["AA"][s]),
                "sim_AB_BB": sim(ab, feats["BB"][s]),
                "sim_AB_BA": sim(ab, feats["BA"][s]),
                "sim_AB_Ablank": sim(ab, feats["A_blank"][s]),
                "sim_AB_blankB": sim(ab, feats["blank_B"][s]),
                "sim_AB_to_A": sim(ab, f_a),
                "sim_AB_to_B": sim(ab, f_b),
            }
            if "AX" in feats:
                row["sim_AB_AX"] = sim(ab, feats["AX"][s])
                row["sim_AB_XB"] = sim(ab, feats["XB"][s])
            inf_b = 1 - row["sim_AB_AA"]
            inf_a = 1 - row["sim_AB_BB"]
            row["influence_of_A"] = inf_a
            row["influence_of_B"] = inf_b
            row["balance_A"] = inf_a / (inf_a + inf_b) if inf_a + inf_b > 1e-8 else None
            per_seed.append(row)

        keys = [k for k in per_seed[0] if k != "seed"]
        avg = {k: mean([r[k] for r in per_seed if r[k] is not None]) for k in keys}
        results[ex.name] = {"per_seed": per_seed, "mean": avg}

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"image_a": args.image_a, "image_b": args.image_b, "alt": args.alt,
                   "seeds": args.seeds, "guidance_scale": args.guidance_scale, "results": results}, f, indent=2)

    print("\n" + "=" * 72)
    print("Mean over seeds                      DINO      CLIP")
    rows = [
        ("influence of A (1 - sim(AB, BB))", "influence_of_A"),
        ("influence of B (1 - sim(AB, AA))", "influence_of_B"),
        ("balance (A share of influence)", "balance_A"),
        ("position: sim(AB, BA)", "sim_AB_BA"),
        ("sim(AB, A only + blank)", "sim_AB_Ablank"),
        ("sim(AB, blank + B only)", "sim_AB_blankB"),
        ("sim(output AB, input A)", "sim_AB_to_A"),
        ("sim(output AB, input B)", "sim_AB_to_B"),
    ]
    if args.alt:
        rows += [("sim(AB, AX)  [B replaced by X]", "sim_AB_AX"), ("sim(AB, XB)  [A replaced by X]", "sim_AB_XB")]
    for label, key in rows:
        d, c = results["dino"]["mean"][key], results["clip"]["mean"][key]
        fmtv = lambda v: "   -  " if v is None else f"{v:6.3f}"
        print(f"{label:36s} {fmtv(d)}    {fmtv(c)}")
    print("=" * 72)
    print("balance ~0.5: both inputs used; near 1: output driven mostly by A; near 0: mostly by B.")


# --------------------------------------------------------------------------------------
# Overview grid
# --------------------------------------------------------------------------------------
def make_overview(conditions, out_dir: Path, seeds, tile=192):
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    label_w, gap = 90, 6
    cols = 2 + len(seeds)  # two inputs + outputs
    W = label_w + cols * (tile + gap) + gap
    H = len(conditions) * (tile + gap) + gap + 22
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    headers = ["input 1", "input 2"] + [f"seed {s}" for s in seeds]
    for j, h in enumerate(headers):
        draw.text((label_w + j * (tile + gap) + gap, 4), h, fill="black", font=font)
    for i, (name, (im1, im2)) in enumerate(conditions.items()):
        y = 22 + i * (tile + gap) + gap
        draw.text((6, y + tile // 2 - 7), name, fill="black", font=font)
        imgs = [im1, im2] + [Image.open(out_dir / name / f"seed_{s:03d}.jpeg") for s in seeds]
        for j, im in enumerate(imgs):
            canvas.paste(im.convert("RGB").resize((tile, tile)), (label_w + j * (tile + gap) + gap, y))
    canvas.save(out_dir / "overview.jpeg", quality=92)


# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Check whether the model uses both input images")
    p.add_argument("--image-a", required=True, help="Top-left input (A)")
    p.add_argument("--image-b", required=True, help="Bottom-right input (B)")
    p.add_argument("--alt", default=None, help="Optional replacement image X (similar style, different content)")
    p.add_argument("--output-dir", default="probe")
    p.add_argument("--seeds", type=lambda v: [int(x) for x in v.split(",")], default=[1, 2, 3, 4])
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--lora-path", default="InspirationSeedsPaper/InspirationSeeds")
    p.add_argument("--cpu-offload", action="store_true", help="Use enable_model_cpu_offload() for low VRAM")
    p.add_argument("--overwrite", action="store_true", help="Regenerate images that already exist")
    p.add_argument("--no-metrics", action="store_true", help="Only generate images")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_a = load_image(args.image_a).convert("RGB")
    img_b = load_image(args.image_b).convert("RGB")
    img_x = load_image(args.alt).convert("RGB") if args.alt else None
    conditions = build_conditions(img_a, img_b, img_x)
    print(f"{len(conditions)} conditions x {len(args.seeds)} seeds = {len(conditions) * len(args.seeds)} images")

    run_generation(args, conditions, out_dir)
    make_overview(conditions, out_dir, args.seeds)
    if not args.no_metrics:
        compute_metrics(args, conditions, out_dir, img_a, img_b)
    print(f"\nOverview: {out_dir / 'overview.jpeg'}")


if __name__ == "__main__":
    main()
