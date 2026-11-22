"""
check_multiref.py -- sanity checks for multiref/conditioning.py (run once on the GPU server).
Last modified: 2026-11-22 (default --main-t 3, multiref.conditioning.MAIN_T)
               2026-11-19 (check 0b: t marker of role mode "tid"; baseline rows per role mode, --main-t)
               2026-11-18 (paths without the hf_ prefix: generate_source, samples/sources, output/decompose/v72_final)

Checks
------
0. ids / tokens (no model weights): the "canvas" reference tokens and ids are the same set as
   concatenating the latents along the width and packing them like the stock pipeline.
0b. t marker (no model weights): with a main reference (role mode "tid"), tokens, spans and
   (h, w) ids are unchanged; only the t id of the main reference's tokens becomes --main-t.
1. One reference: MultiRefKontextPipeline.generate_multi gives the SAME image as the stock
   FluxKontextPipeline call with that image (reference and output at --size, 1024 by default;
   the stock call uses _auto_resize=False, so the same check also works for other sizes).
   Expected max diff: 0.
2. Two references, "canvas": same result as the stock pipeline with the width-concatenated
   latents. Only the token order differs, so a small numerical difference is expected.
3. Role embedding with E_main = 0: identical output with and without the hook. Expected: 0.
4. Role embedding slice: with a random E_main, only the tokens of the main reference change
   at the output of transformer.x_embedder.
5. Baseline images of the base model (no LoRA), saved to --output-dir/baseline_<folder>.jpeg:
     first row:  no main image; the "marked image" prompt without any marker (role "embed"
                 with E_main = 0: the same for every main, so shown once); the "per_t" layout
     then one row per role mode in --baseline-roles (default: tid), each input as main:
       tid    input k gets t = --main-t. The base model should still use all references (no
              image dropped, no unrelated object); a reaction to the marker is welcome but
              not needed, since training teaches what it means
       slot   input k moved first, "first image" prompt
       index  "image k" in the prompt

Usage
-----
    python check_multiref.py --folder generate_source/<name>        # all checks
    python check_multiref.py --skip-model                               # checks 0 and 0b only
    python check_multiref.py --folder generate_source/<name> --baseline-roles tid,slot --main-t 2
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import FluxKontextPipeline
from PIL import Image, ImageDraw, ImageFont

from multiref.conditioning import MAIN_T, MainRoleEmbedding, MultiRefKontextPipeline, build_reference_tokens, load_rgb
from multiref.data import list_inputs
from multiref.prompts import render_prompt


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def check_ids_tokens():
    """Check 0: canvas tokens/ids vs stock packing of width-concatenated latents."""
    torch.manual_seed(0)
    lat = [torch.randn(1, 16, 64, 64) for _ in range(3)]
    tok, ids, spans = build_reference_tokens(lat, "canvas")
    cat = torch.cat(lat, dim=3)
    stock_tok = FluxKontextPipeline._pack_latents(cat, 1, 16, 64, 64 * 3)
    stock_ids = FluxKontextPipeline._prepare_latent_image_ids(1, 32, 32 * 3, "cpu", torch.float32)
    stock_ids[..., 0] = 1
    # match the two token orders through their (h, w) ids
    key = lambda x: (x[:, 1] * 100000 + x[:, 2]).long()
    o1, o2 = torch.argsort(key(ids)), torch.argsort(key(stock_ids))
    same_ids = torch.equal(ids[o1], stock_ids[o2])
    same_tok = torch.equal(tok[0, o1], stock_tok[0, o2])
    spans_ok = spans == [(0, 1024), (1024, 2048), (2048, 3072)]
    return report("0 canvas ids/tokens == stock packing", same_ids and same_tok and spans_ok,
                  f"ids {same_ids}, tokens {same_tok}, spans {spans}")


def check_t_marker(main_t):
    """Check 0b: the "tid" marker only changes the t id of the main reference's tokens."""
    torch.manual_seed(0)
    lat = [torch.randn(1, 16, 64, 64) for _ in range(3)]
    tok, ids, spans = build_reference_tokens(lat, "canvas")
    main = 1
    tok_m, ids_m, spans_m = build_reference_tokens(lat, "canvas", main_pos=main, main_t=main_t)
    a, b = spans[main]
    expected = ids.clone()
    expected[a:b, 0] = main_t
    ok = torch.equal(tok, tok_m) and spans == spans_m and torch.equal(ids_m, expected)
    t_values = sorted(set(ids_m[:, 0].tolist()))
    return report(f"0b t marker: only the main reference gets t = {main_t:g}", ok,
                  f"tokens equal {torch.equal(tok, tok_m)}, spans equal {spans == spans_m}, t values {t_values}")


def max_diff(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)).max())


def run_model_checks(args):
    paths = list_inputs(args.folder)
    if len(paths) < 2:
        raise ValueError(f"{args.folder} needs at least 2 input images")
    images = [load_rgb(p) for p in paths]
    prompt = render_prompt(None, "embed")
    common = dict(num_inference_steps=args.steps, guidance_scale=args.guidance_scale, output_type="np")
    gen = lambda: torch.Generator(device="cuda").manual_seed(args.seed)

    pipe = MultiRefKontextPipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    results = []

    # 1. one reference: our path == stock path
    #    _auto_resize=False keeps the reference at its own size (otherwise the stock pipeline
    #    resizes it to a preferred Kontext resolution, 1024 x 1024 for a square image);
    #    max_area = size^2 keeps the output at --size (the stock default is 1024^2)
    ref = images[0].resize((args.size, args.size))
    stock = pipe(image=ref, prompt=prompt, height=args.size, width=args.size, max_area=args.size * args.size,
                 _auto_resize=False, generator=gen(), **common).images
    ours, _ = pipe.generate_multi([ref], main=None, ref_size=args.size, height=args.size, width=args.size,
                                  prompt=prompt, generator=gen(), **common)
    d = max_diff(stock, ours.images)
    results.append(report(f"1 one reference == stock pipeline ({args.size} px)", d == 0.0, f"max abs diff {d:.2e}"))

    # 2. two references, canvas: our path ~ stock path with width-concatenated latents
    lat = pipe.encode_references(images[:2], 512)
    stock = pipe(image=torch.cat(lat, dim=3), prompt=prompt, height=args.size, width=args.size,
                 max_area=args.size * args.size, generator=gen(), **common).images
    ours, _ = pipe.generate_multi(images[:2], main=None, ref_size=512, height=args.size, width=args.size,
                                  prompt=prompt, generator=gen(), **common)
    d = max_diff(stock, ours.images)
    results.append(report("2 two references (canvas) ~ stock concatenated latents", d < 0.05,
                          f"max abs diff {d:.2e} (token order differs; small values are expected)"))

    # 3. zero role embedding: hook on == hook off
    role = MainRoleEmbedding(pipe.transformer.inner_dim).to("cuda")
    off, _ = pipe.generate_multi(images, main=0, role_mode="embed", role_embedding=None, height=args.size,
                                 width=args.size, generator=gen(), **common)
    role.attach(pipe.transformer)
    on, _ = pipe.generate_multi(images, main=0, role_mode="embed", role_embedding=role, height=args.size,
                                width=args.size, generator=gen(), **common)
    d = max_diff(off.images, on.images)
    results.append(report("3 zero E_main: hook on == hook off", d == 0.0, f"max abs diff {d:.2e}"))

    # 4. role embedding slice: only the main reference's tokens change
    captured = []
    cap = pipe.transformer.x_embedder.register_forward_hook(lambda m, i, o: captured.append(o.detach().float().cpu()))
    main_idx = len(images) - 1
    step1 = dict(common, num_inference_steps=1)
    pipe.generate_multi(images, main=main_idx, role_mode="embed", role_embedding=role, height=args.size,
                        width=args.size, generator=gen(), **step1)
    with torch.no_grad():
        role.e_main.normal_()
    pipe.generate_multi(images, main=main_idx, role_mode="embed", role_embedding=role, height=args.size,
                        width=args.size, generator=gen(), **step1)
    cap.remove()
    role.detach()
    with torch.no_grad():
        role.e_main.zero_()
    changed = (captured[0] - captured[-1]).abs().amax(dim=-1)[0] > 0
    seq = changed.shape[0]
    n_ref = 1024 * len(images)
    expected = torch.zeros(seq, dtype=torch.bool)
    expected[seq - n_ref + main_idx * 1024: seq - n_ref + (main_idx + 1) * 1024] = True
    results.append(report("4 E_main only changes the main reference's tokens", torch.equal(changed, expected),
                          f"changed tokens {int(changed.sum())}, expected {int(expected.sum())}, "
                          f"first changed {int(changed.nonzero()[0]) if changed.any() else -1}, "
                          f"expected first {seq - n_ref + main_idx * 1024}"))

    # 5. baseline images of the base model
    def baseline(main, role_mode, layout="canvas"):
        out, _ = pipe.generate_multi(images, main=main, role_mode=role_mode, layout=layout, height=args.size,
                                     width=args.size, num_inference_steps=args.baseline_steps,
                                     guidance_scale=args.guidance_scale, main_t=args.main_t, generator=gen())
        return out.images[0]

    rows = [[("no main", baseline(None, "embed")),
             ("marked prompt, no marker", baseline(0, "embed")),
             ("no main, per_t", baseline(None, "embed", "per_t"))]]
    labels = {"tid": "t={t:g} on image {k}", "slot": "image {k} first", "index": "'image {k}' in prompt"}
    for mode in args.baseline_roles:
        rows.append([(f"{mode}: " + labels[mode].format(t=args.main_t, k=k + 1), baseline(k, mode))
                     for k in range(len(images))])
        print(f"baseline row: {mode}")
    save_baseline(images, rows, args.output_dir / f"baseline_{args.folder.name}.jpeg")
    return results


def save_baseline(inputs, rows, path, tile=256):
    """Inputs in the first row, then one row per list of (label, image)."""
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    gap, cap = 8, 22
    grid = [[(f"image {i + 1}", im) for i, im in enumerate(inputs)]] + rows
    n = max(len(r) for r in grid)
    canvas = Image.new("RGB", (gap + n * (tile + gap), gap + len(grid) * (tile + cap + gap)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(grid):
        y = gap + r * (tile + cap + gap)
        for i, (label, im) in enumerate(row):
            x = gap + i * (tile + gap)
            canvas.paste(im.resize((tile, tile)), (x, y))
            draw.text((x + 2, y + tile + 3), label, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)
    print(f"baseline images: {path}")


def main():
    p = argparse.ArgumentParser(description="Sanity checks for multiref/conditioning.py")
    p.add_argument("--folder", type=Path, help="Decomposition folder with 2-4 input images (needed for checks 1-5)")
    p.add_argument("--base-model", default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--skip-model", action="store_true", help="Only run checks 0 and 0b (no model weights)")
    p.add_argument("--baseline-roles", default="tid",
                   type=lambda x: [m for m in x.split(",") if m.strip()],
                   help="Comma-separated role modes with one baseline row each: tid, slot, index ('' = none)")
    p.add_argument("--main-t", type=float, default=MAIN_T, help="t id of the main image in the 'tid' rows")
    p.add_argument("--steps", type=int, default=4, help="Denoising steps for checks 1-3")
    p.add_argument("--baseline-steps", type=int, default=28, help="Denoising steps for the baseline images")
    p.add_argument("--size", type=int, default=1024, help="Output size (training target size)")
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/check_multiref"))
    args = p.parse_args()

    bad = [m for m in args.baseline_roles if m not in ("tid", "slot", "index")]
    if bad:
        p.error(f"--baseline-roles: unknown role mode(s) {bad}; use tid, slot, index")
    results = [check_ids_tokens(), check_t_marker(args.main_t)]
    if not args.skip_model:
        if args.folder is None:
            p.error("--folder is required unless --skip-model is given")
        results += run_model_checks(args)
    print(f"\n{sum(results)} / {len(results)} checks passed")


if __name__ == "__main__":
    main()
