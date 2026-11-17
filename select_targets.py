"""
select_targets.py -- suggest designated-subject targets for manual review (synthesize_v2_1.py outputs).
Last modified: 2026-11-17

This script only SUGGESTS which cells to keep. The final choice is made by hand in the
review folders; build_dataset.py reads what is left in the keep/ folders.

Checks (per generated cell (t, s) of every designated input k)
--------------------------------------------------------------
    copy          cos_main > --copy-max (0.80)        the target is a near copy of the main input
    no_effect     lift     < --lift-min (0.0)          the target is not closer to the main than the reconstruction
    not_dominant  margin   < --margin-min (-0.15)      the target still resembles another input more than the main
    not_distinct  distinct < --distinct-min (0.03)     the target cannot be told apart from the targets of the
                                                       other mains of the same folder

    distinct = cos(T, input_k) - max_{j != k} mean_{cells c of main j} cos(T_j^c, input_k)
               (only cells of main j that pass the first three checks are used; None, and the
               check skipped, if no other main has such a cell)

The thresholds were set on 14 folders (41 mains, 768 px targets) with 6 hand-picked cells:
all 6 pass. They are only suggestions and were not recalibrated for 1024 px targets.

Review folders (--output-dir/review)
------------------------------------
    review/<folder>/
        _inputs.jpeg            all inputs of the folder, labelled input_0, input_1, ...
        info.json               decomposition folder, inputs, original path of every cell (do not edit)
        main<k>/
            _main.jpeg          the main input k
            keep/t0.30_s0.40.jpeg            suggested to keep
            drop/t0.60_s0.40__copy.jpeg      suggested to drop (reasons after "__")

Manual selection:
    * keep a cell        -> its file is in keep/   (move it from drop/ to keep/ if needed)
    * drop a cell        -> move (or delete) it out of keep/
    * drop a whole main  -> leave its keep/ empty, or delete main<k>/
    * drop a whole folder-> delete review/<folder>/
    * add any other image -> copy it into keep/ (e.g. a cell from another synthesis run); it must
                            be generated from the same inputs. build_dataset.py uses the files in
                            keep/ as they are, so the file name does not matter.
    Moving a file from drop/ to keep/ without renaming is fine: the "__reason" suffix is ignored.

Rerunning the script never touches an existing review/<folder>/ (your edits are kept); new
folders are added. Use --overwrite to rebuild every review folder from the suggestions.

Other output (--output-dir)
---------------------------
    suggestions.csv   one row per cell: folder, main, class, t, s, suggestion, reasons, metrics
                      (for sorting / filtering in a spreadsheet; rewritten on every run)
    summary.json      thresholds and counts

Usage
-----
    python select_targets.py --synth-root output/synth/v2_1_1024 --output-dir output/selection
"""

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from multiref.data import list_inputs

REQUIRED_CELL_FIELDS = ("cos_main", "lift", "margin", "cos_inputs", "file", "t", "strength")


def cell_name(t: float, s: float) -> str:
    """File stem of a cell, e.g. t0.30_s0.40 (same format as synthesize_v2_1.py)."""
    return f"t{t:.2f}_s{s:.2f}"


def check_folder(meta: dict, args) -> dict:
    """Apply the checks to every cell of one synth_meta.json. Returns {main: {class, closeness, cells}}."""
    per_main = {}
    for r in meta["results"]:
        cells = []
        for c in r["cells"]:
            if c.get("strength") is None:  # reference column of --embed-only, not a target candidate
                continue
            missing = [f for f in REQUIRED_CELL_FIELDS if f not in c]
            if missing:
                raise ValueError(f"cell without {missing}: not a synthesize_v2_1.py output ({meta.get('folder')})")
            reasons = []
            if c["cos_main"] > args.copy_max:
                reasons.append("copy")
            if c["lift"] < args.lift_min:
                reasons.append("no_effect")
            if c["margin"] < args.margin_min:
                reasons.append("not_dominant")
            cells.append({"t": c["t"], "strength": c["strength"], "file": c["file"], "cos_main": c["cos_main"],
                          "lift": c["lift"], "margin": c["margin"], "cos_inputs": c["cos_inputs"],
                          "pre_ok": not reasons, "reasons": reasons})
        per_main[r["main"]] = {"class": r.get("class"), "closeness": r.get("closeness"), "cells": cells}

    # distinctness, using the cells of the other mains that passed the first three checks
    for k, m in per_main.items():
        others = []
        for j, o in per_main.items():
            ok = [c for c in o["cells"] if c["pre_ok"]]
            if j != k and ok:
                others.append(sum(c["cos_inputs"][k] for c in ok) / len(ok))
        for c in m["cells"]:
            if others:
                c["distinct"] = c["cos_inputs"][k] - max(others)
                if c["distinct"] < args.distinct_min:
                    c["reasons"].append("not_distinct")
            else:
                c["distinct"] = None
            c["keep"] = not c["reasons"]
    return per_main


def inputs_strip(paths, out_path: Path, tile=256):
    """All inputs of a folder in one row, labelled input_0, input_1, ..."""
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    gap, cap = 8, 24
    canvas = Image.new("RGB", (gap + len(paths) * (tile + gap), tile + cap + 2 * gap), "white")
    draw = ImageDraw.Draw(canvas)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB") if Path(p).exists() else Image.new("RGB", (tile, tile), "gray")
        x = gap + i * (tile + gap)
        canvas.paste(im.resize((tile, tile)), (x, gap))
        draw.text((x + 2, gap + tile + 4), f"input_{i}", fill="black", font=font)
    canvas.save(out_path, quality=92)


def write_review_folder(review_dir: Path, folder: Path, synth_dir: Path, inputs, per_main):
    """review/<folder>/ with _inputs.jpeg, info.json and main<k>/{_main.jpeg, keep/, drop/}."""
    review_dir.mkdir(parents=True, exist_ok=True)
    inputs_strip(inputs, review_dir / "_inputs.jpeg")
    info = {"decomposition_folder": str(folder), "synth_dir": str(synth_dir),
            "inputs": [str(p) for p in inputs], "cells": {}}
    for k, m in sorted(per_main.items()):
        main_dir = review_dir / f"main{k}"
        (main_dir / "keep").mkdir(parents=True, exist_ok=True)
        (main_dir / "drop").mkdir(parents=True, exist_ok=True)
        if Path(inputs[k]).exists():
            shutil.copy2(inputs[k], main_dir / "_main.jpeg")
        info["cells"][f"main{k}"] = {}
        for c in m["cells"]:
            src = synth_dir / c["file"]
            name = cell_name(c["t"], c["strength"])
            info["cells"][f"main{k}"][name] = str(src)
            if not src.exists():
                print(f"[warn] missing target {src}")
                continue
            if c["keep"]:
                shutil.copy2(src, main_dir / "keep" / f"{name}.jpeg")
            else:
                shutil.copy2(src, main_dir / "drop" / f"{name}__{'_'.join(c['reasons'])}.jpeg")
    with open(review_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)


# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Suggest designated-subject targets for manual review")
    p.add_argument("--synth-root", type=Path, default=Path("output/synth/v2_1_1024"),
                   help="Output root of synthesize_v2_1.py (searched recursively for synth_meta.json)")
    p.add_argument("--output-dir", type=Path, default=Path("output/selection"))
    p.add_argument("--copy-max", type=float, default=0.80, help="Suggest dropping cells with cos_main above this")
    p.add_argument("--lift-min", type=float, default=0.0, help="Suggest dropping cells with lift below this")
    p.add_argument("--margin-min", type=float, default=-0.15, help="Suggest dropping cells with margin below this")
    p.add_argument("--distinct-min", type=float, default=0.03, help="Suggest dropping cells with distinctness below this")
    p.add_argument("--overwrite", action="store_true",
                   help="Rebuild existing review folders from the suggestions (discards manual edits)")
    args = p.parse_args()

    metas = sorted(args.synth_root.rglob("synth_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No synth_meta.json under {args.synth_root}")
    review_root = args.output_dir / "review"
    review_root.mkdir(parents=True, exist_ok=True)

    rows = []
    reason_counts = Counter()
    n_new = n_kept_existing = n_keep = n_cells = n_mains = n_mains_keep = 0
    for meta_path in metas:
        with open(meta_path) as f:
            meta = json.load(f)
        synth_dir = meta_path.parent
        folder = Path(meta["folder"])
        inputs = list_inputs(folder)
        if len(inputs) != meta["n_inputs"]:
            print(f"[warn] {folder}: found {len(inputs)} input images, synth_meta.json says {meta['n_inputs']}; "
                  "run from the directory where synthesize_v2_1.py was run")
            inputs = [folder / f"input_{i}.jpeg" for i in range(meta["n_inputs"])]
        per_main = check_folder(meta, args)

        name = "__".join(synth_dir.relative_to(args.synth_root).parts) or synth_dir.name
        for k, m in sorted(per_main.items()):
            n_mains += 1
            n_mains_keep += any(c["keep"] for c in m["cells"])
            for c in m["cells"]:
                n_cells += 1
                n_keep += c["keep"]
                reason_counts.update(c["reasons"])
                rows.append({"folder": name, "main": k, "class": m["class"],
                             "closeness": "" if m["closeness"] is None else round(m["closeness"], 4),
                             "t": c["t"], "s": c["strength"], "cell": cell_name(c["t"], c["strength"]),
                             "suggestion": "keep" if c["keep"] else "drop", "reasons": " ".join(c["reasons"]),
                             "cos_main": round(c["cos_main"], 4), "lift": round(c["lift"], 4),
                             "margin": round(c["margin"], 4),
                             "distinct": "" if c["distinct"] is None else round(c["distinct"], 4),
                             "target": str(synth_dir / c["file"])})

        review_dir = review_root / name
        if review_dir.exists() and not args.overwrite:
            n_kept_existing += 1
            continue
        if review_dir.exists():
            shutil.rmtree(review_dir)
        write_review_folder(review_dir, folder, synth_dir, inputs, per_main)
        n_new += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "suggestions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "thresholds": {"copy_max": args.copy_max, "lift_min": args.lift_min, "margin_min": args.margin_min,
                       "distinct_min": args.distinct_min},
        "folders": len(metas), "mains": n_mains, "mains_with_a_suggested_cell": n_mains_keep,
        "cells": n_cells, "cells_suggested_keep": n_keep, "drop_reasons": dict(reason_counts),
        "review_folders_written": n_new, "review_folders_left_unchanged": n_kept_existing,
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{len(metas)} folder(s), {n_mains} main(s), {n_cells} cell(s)")
    print(f"suggested keep: {n_keep} / {n_cells} cells; mains with at least one suggested cell: {n_mains_keep} / {n_mains}")
    print("drop reasons (a cell can have several): " + ", ".join(f"{k} {v}" for k, v in reason_counts.most_common()))
    print(f"review folders: {n_new} written, {n_kept_existing} left unchanged (manual edits kept) -> {review_root}")
    print(f"suggestions table: {args.output_dir / 'suggestions.csv'}")


if __name__ == "__main__":
    main()
