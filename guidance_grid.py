"""
guidance_grid.py -- one overview image of generate_multi.py --main-guidance results.
Last modified: 2026-11-21

Rows: one per (main, seed); columns: one per guidance weight w, then the target of that main
(from the build_dataset.py manifest). Reads the files
    <dir>/<folder>__main{k}__mg{w}__seed{s}.jpeg
and writes <dir>/_grid_<folder>.jpeg.

Usage
-----
    python guidance_grid.py outputs/main_guidance/104_t3_s2000 104_topk64_nmax4
    python guidance_grid.py outputs/main_guidance/104 104_topk64_nmax4 --data-dir data/multiref
"""

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_targets(data_dir: Path, folder: str):
    """First target of every designated main of the folder (0-based main -> image path)."""
    targets = {}
    for split in ("train", "val"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in open(path):
            row = json.loads(line)
            if Path(row["folder"]).name == folder and row.get("main") is not None and row.get("targets"):
                targets.setdefault(row["main"], row["targets"][0])
    return targets


def main():
    p = argparse.ArgumentParser(description="Overview of main-guidance results")
    p.add_argument("dir", type=Path, help="--output-dir of generate_multi.py")
    p.add_argument("folder", help="Folder name used as file prefix, e.g. 104_topk64_nmax4")
    p.add_argument("--data-dir", type=Path, default=Path("data/multiref"), help="Manifests with the targets")
    p.add_argument("--tile", type=int, default=256)
    args = p.parse_args()

    pattern = re.compile(re.escape(args.folder) + r"__main(\d+)__mg([0-9.]+)__seed(\d+)\.jpeg$")
    found = {}
    for f in args.dir.glob(f"{args.folder}__main*__mg*__seed*.jpeg"):
        m = pattern.search(f.name)  # preview files (..._preview.jpeg) do not match
        if m:
            found[(int(m[1]), int(m[3]), float(m[2]))] = f
    if not found:
        raise ValueError(f"No {args.folder}__main*__mg*__seed*.jpeg files in {args.dir}")
    mains = sorted({k for k, _, _ in found})
    seeds = sorted({s for _, s, _ in found})
    weights = sorted({w for _, _, w in found})
    targets = load_targets(args.data_dir, args.folder)

    try:
        font = ImageFont.load_default(size=14)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    t, gap, cap, head = args.tile, 6, 20, 28
    rows = [(k, s) for k in mains for s in seeds]
    n_cols = len(weights) + 1
    canvas = Image.new("RGB", (gap + n_cols * (t + gap), head + len(rows) * (t + cap + gap)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 6), f"{args.folder}: rows = --main k / seed, columns = main guidance w, last = target",
              fill="black", font=font)
    for r, (k, s) in enumerate(rows):
        y = head + r * (t + cap + gap)
        cells = [(found.get((k, s, w)), f"main {k}, seed {s}, w {w:g}") for w in weights]
        target = targets.get(k - 1)  # --main is 1-based, the manifest is 0-based
        cells.append((Path(target) if target else None, f"target main {k}"))
        for c, (path, label) in enumerate(cells):
            x = gap + c * (t + gap)
            if path is not None and path.exists():
                canvas.paste(Image.open(path).convert("RGB").resize((t, t)), (x, y))
            else:
                draw.rectangle([x, y, x + t - 1, y + t - 1], outline=(180, 180, 180))
                draw.text((x + 8, y + t // 2), "missing", fill=(150, 150, 150), font=font)
            draw.text((x + 2, y + t + 2), label, fill="black", font=font)
    out = args.dir / f"_grid_{args.folder}.jpeg"
    canvas.save(out, quality=90)
    print(f"saved {out}: {len(mains)} mains x {len(seeds)} seeds, weights {', '.join(f'{w:g}' for w in weights)}")


if __name__ == "__main__":
    main()
