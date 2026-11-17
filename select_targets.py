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
        _overview.jpeg          everything needed to decide, in one image (see below)
        _inputs.jpeg            all inputs of the folder, labelled input_0, input_1, ...
        info.json               decomposition folder, inputs, original path of every cell (do not edit)
        main<k>/
            _main.jpeg          the main input k
            keep/t0.30_s0.40.jpeg            suggested to keep
            drop/t0.60_s0.40__copy.jpeg      suggested to drop (reasons after "__")
    review/_created.txt         folders created so far (do not edit): a folder listed here is never
                                recreated, so deleting review/<folder>/ removes it for good

_overview.jpeg
    row 1        all inputs of the folder
    one row per main: the main input, then every candidate cell with
        - its name (= file name in keep/ / drop/)
        - cos_main, lift, margin, distinct
        - the suggestion: frame green (keep) or red (drop, with the reasons)
        - "now": whether the cell is CURRENTLY in keep/ (reflects your manual changes)
      images added to keep/ by hand (not one of the cells) are appended with a blue frame
    The overview is redrawn on every run, also for existing folders; keep/ and drop/ are
    never changed by a rerun.

Manual selection:
    * keep a cell        -> its file is in keep/   (move it from drop/ to keep/ if needed)
    * drop a cell        -> move (or delete) it out of keep/
    * drop a whole main  -> leave its keep/ empty, or delete main<k>/
    * drop a whole folder-> delete review/<folder>/ (it is not recreated by a rerun)
    * add any other image -> copy it into keep/ (e.g. a cell from another synthesis run); it must
                            be generated from the same inputs. build_dataset.py uses the files in
                            keep/ as they are, so the file name does not matter.
    Moving a file from drop/ to keep/ without renaming is fine: the "__reason" suffix is ignored.

Rerunning the script never touches keep/ and drop/ of an existing review/<folder>/ (your edits
are kept) and never recreates a deleted one; only _overview.jpeg is redrawn and new folders are
added. Use --overwrite to rebuild every review folder from the suggestions (discards all edits,
also deletions).

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

from multiref.data import IMAGE_EXTS, list_inputs

REQUIRED_CELL_FIELDS = ("cos_main", "lift", "margin", "cos_inputs", "file", "t", "strength")
CREATED_FILE = "_created.txt"
GREEN, RED, BLUE, GRAY = (40, 160, 40), (210, 40, 40), (40, 90, 220), (120, 120, 120)


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


def load_font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def open_tile(path, tile):
    path = Path(path)
    im = Image.open(path).convert("RGB") if path.exists() else Image.new("RGB", (tile, tile), "gray")
    return im.resize((tile, tile))


def write_overview(review_dir: Path, inputs, per_main, synth_dir: Path, tile: int = 288):
    """_overview.jpeg: inputs, then per main the main input and every candidate with metrics and status."""
    font, small = load_font(16), load_font(14)
    gap, line = 10, 17
    cap_cell = 5 * line + 6     # name, 2 metric lines, suggestion, current status
    cap_head = 3 * line + 6
    frame = 5

    # one row per main: (main tile, main caption lines, [cells]); a cell = (image, lines, frame color)
    main_rows = []
    for k, m in sorted(per_main.items()):
        main_dir = review_dir / f"main{k}"
        removed = not main_dir.is_dir()
        kept_files = []
        if (main_dir / "keep").is_dir():
            kept_files = sorted(x for x in (main_dir / "keep").iterdir() if x.suffix.lower() in IMAGE_EXTS)
        kept_names = {x.stem.split("__")[0] for x in kept_files}
        cells, cell_names = [], set()
        for c in sorted(m["cells"], key=lambda c: (c["t"], c["strength"])):
            name = cell_name(c["t"], c["strength"])
            cell_names.add(name)
            dist = "n/a" if c["distinct"] is None else f"{c['distinct']:+.2f}"
            suggestion = ("suggest: KEEP", GREEN) if c["keep"] else ("drop: " + ",".join(c["reasons"]), RED)
            now = ("now: removed with main", GRAY) if removed else (
                ("now: IN keep/", GREEN) if name in kept_names else ("now: not kept", GRAY))
            lines = [(name, "black"),
                     (f"cos_main {c['cos_main']:.2f}  lift {c['lift']:+.2f}", "black"),
                     (f"margin {c['margin']:+.2f}  distinct {dist}", "black"),
                     suggestion, now]
            cells.append((open_tile(synth_dir / c["file"], tile), lines, GREEN if c["keep"] else RED))
        for x in kept_files:  # images added to keep/ by hand
            if x.stem.split("__")[0] not in cell_names:
                cells.append((open_tile(x, tile), [(x.name, "black"), ("added by hand", BLUE),
                                                   ("", "black"), ("", "black"), ("now: IN keep/", GREEN)], BLUE))
        closeness = "" if m["closeness"] is None else f" {m['closeness']:.2f}"
        head = [(f"main = input_{k}", "black"), (f"{m['class'] or ''}{closeness}", "black"),
                ("REMOVED" if removed else f"in keep/: {len(kept_files)}", RED if removed else "black")]
        main_rows.append((open_tile(inputs[k], tile), head, cells))

    n_cols = max([len(inputs)] + [1 + len(cells) for _, _, cells in main_rows])
    W = gap + n_cols * (tile + gap)
    title_h = 2 * line + 2 * gap
    H = title_h + (tile + cap_head + gap) + len(main_rows) * (tile + cap_cell + 2 * gap)
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, gap), review_dir.name, fill="black", font=font)
    draw.text((gap, gap + line + 2), "frame: suggestion (green keep / red drop / blue added by hand); "
              "'now' = current content of keep/", fill=GRAY, font=small)

    y = title_h
    for i, p in enumerate(inputs):
        x = gap + i * (tile + gap)
        canvas.paste(open_tile(p, tile), (x, y))
        draw.text((x + 2, y + tile + 4), f"input_{i}", fill="black", font=font)
    y += tile + cap_head + gap

    for main_tile, head, cells in main_rows:
        draw.line([(gap, y - gap // 2), (W - gap, y - gap // 2)], fill=GRAY, width=1)
        canvas.paste(main_tile, (gap, y + gap // 2))
        for j, (text, color) in enumerate(head):
            draw.text((gap + 2, y + gap // 2 + tile + 4 + j * line), text, fill=color, font=small)
        for c, (im, lines, color) in enumerate(cells):
            x = gap + (c + 1) * (tile + gap)
            yy = y + gap // 2
            canvas.paste(im, (x, yy))
            draw.rectangle([x - 2, yy - 2, x + tile + 1, yy + tile + 1], outline=color, width=frame)
            for j, (text, tcolor) in enumerate(lines):
                draw.text((x + 2, yy + tile + 4 + j * line), text, fill=tcolor, font=small)
        y += tile + cap_cell + 2 * gap
    canvas.save(review_dir / "_overview.jpeg", quality=88)


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
    p.add_argument("--overview-tile", type=int, default=288, help="Image size in _overview.jpeg")
    p.add_argument("--overwrite", action="store_true",
                   help="Rebuild every review folder from the suggestions (discards manual edits and deletions)")
    args = p.parse_args()

    metas = sorted(args.synth_root.rglob("synth_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No synth_meta.json under {args.synth_root}")
    review_root = args.output_dir / "review"
    review_root.mkdir(parents=True, exist_ok=True)
    created_path = review_root / CREATED_FILE
    created = set() if args.overwrite or not created_path.exists() else \
        {x.strip() for x in created_path.read_text().splitlines() if x.strip()}
    # review folders made by an earlier version of this script (before _created.txt existed)
    created |= {d.name for d in review_root.iterdir() if d.is_dir() and (d / "info.json").exists()} \
        if not args.overwrite else set()

    rows = []
    reason_counts = Counter()
    n_new = n_kept_existing = n_deleted = n_keep = n_cells = n_mains = n_mains_keep = 0
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
        if name in created:
            if not review_dir.exists():   # deleted by hand: the folder stays excluded
                n_deleted += 1
                continue
            n_kept_existing += 1
        else:
            if review_dir.exists():
                shutil.rmtree(review_dir)
            write_review_folder(review_dir, folder, synth_dir, inputs, per_main)
            created.add(name)
            n_new += 1
        write_overview(review_dir, inputs, per_main, synth_dir, tile=args.overview_tile)

    created_path.write_text("\n".join(sorted(created)) + "\n")
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
        "review_folders_deleted_by_hand": n_deleted,
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{len(metas)} folder(s), {n_mains} main(s), {n_cells} cell(s)")
    print(f"suggested keep: {n_keep} / {n_cells} cells; mains with at least one suggested cell: {n_mains_keep} / {n_mains}")
    print("drop reasons (a cell can have several): " + ", ".join(f"{k} {v}" for k, v in reason_counts.most_common()))
    print(f"review folders: {n_new} written, {n_kept_existing} left unchanged (manual edits kept), "
          f"{n_deleted} deleted by hand (not recreated) -> {review_root}")
    print("overview images redrawn: review/<folder>/_overview.jpeg")
    print(f"suggestions table: {args.output_dir / 'suggestions.csv'}")


if __name__ == "__main__":
    main()
