"""
build_dataset.py -- training / validation manifests for multi-reference fine-tuning.
Last modified: 2026-11-19 (default paths: --decomp-root data/no_main, --review-dir data/selection/review)
               2026-11-18 (paths without the hf_ prefix: generate_source, samples/sources, output/decompose/v72_final)

Two sample pools:
    unspecified  every decomposition folder in --decomp-root (data/no_main, the hand-picked
                 decompositions of all v7.2 settings; output/decompose/ keeps the raw runs):
                 inputs = input_0.jpeg, input_1.jpeg, ...; no main image;
                 target = the original source image (path from groups.json "image", or found by
                 stem in --source-dir); recon.jpeg is not used
    designated   every main with at least one image left in a keep/ folder of the manual review
                 (--review-dir, written by select_targets.py): same kind of inputs, main = k,
                 targets = the image files in keep/, used as they are (training picks one of them
                 at random). Any image placed in keep/ counts, also cells that were suggested to
                 drop or images from other runs; the review folder is therefore part of the
                 dataset and must not be deleted before cache_features.py has run.

The designated samples were synthesized on generate_source, a hand-picked copy of part of
no_main (screen_sources.py, then manual choice). For every review folder, the inputs are compared byte by byte with the folder of
the same name in --decomp-root: if they are identical, the --decomp-root paths are used (both
pools then share the cached input latents); otherwise the inputs used for the synthesis are
kept and the folder is reported, because a target must always come with the inputs it was
synthesized from.

The ratio between the pools is NOT set here: the training sampler draws a pool at every step
according to a schedule (e.g. unspecified only for the first 30 % of the steps), then a sample
inside the pool. The manifests only record the pool of every sample.

Train / validation split by SOURCE IMAGE (stem), stratified: --val-frac of the sources that
have designated samples and --val-frac of the other sources go to validation, so validation
contains both pools (designated folders with >= 2 kept mains allow checking that the result
follows the designated main).

Output (--output-dir)
    train.jsonl, val.jsonl   one sample per line:
        {id, stem, pool, folder, n_inputs, inputs, main (0-based or null), targets, cells}
    stats.json               counts per split, pool and number of inputs

Usage
-----
    python build_dataset.py             # data/no_main + data/selection/review -> data/multiref
    python build_dataset.py --decomp-root data/no_main \\
        --review-dir data/selection/review --output-dir data/multiref_v2
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from multiref.data import IMAGE_EXTS, list_inputs, stem_from_folder, write_jsonl


def find_decomp_folders(root: Path):
    """Folders under root (root included) with at least input_0.jpeg and input_1.jpeg."""
    folders = {p.parent for p in root.rglob("input_0.jpeg")}
    return sorted(f for f in folders if (f / "input_1.jpeg").exists())


def index_sources(source_dir: Path):
    index = {}
    if source_dir and source_dir.is_dir():
        for p in sorted(source_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                index.setdefault(p.stem, p)
    return index


def source_image(folder: Path, sources: dict):
    """Original image of a decomposition folder: groups.json "image", else <source-dir>/<stem>.*"""
    groups = folder / "groups.json"
    if groups.exists():
        with open(groups) as f:
            path = json.load(f).get("image")
        if path and Path(path).exists():
            return Path(path)
    return sources.get(stem_from_folder(folder))


def file_hash(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def same_images(a, b) -> bool:
    return len(a) == len(b) and all(Path(x).exists() and Path(y).exists() and file_hash(x) == file_hash(y)
                                    for x, y in zip(a, b))


def stratified_split(stems_designated, stems_other, frac, rng):
    """Validation stems: frac of each group (at least one per non-empty group with > 1 stem)."""
    val = set()
    for group in (sorted(stems_designated), sorted(stems_other)):
        rng.shuffle(group)
        n = round(len(group) * frac)
        if frac > 0 and len(group) > 1:
            n = max(1, n)
        val.update(group[:n])
    return val


def main():
    p = argparse.ArgumentParser(description="Build manifests for multi-reference fine-tuning")
    p.add_argument("--decomp-root", type=Path, default=Path("data/no_main"),
                   help="Decomposition folders (searched recursively); each gives one unspecified sample")
    p.add_argument("--review-dir", type=Path, default=Path("data/selection/review"),
                   help="Manual review folders from select_targets.py (designated samples)")
    p.add_argument("--source-dir", type=Path, default=Path("samples/sources"),
                   help="Fallback location of the source images (<stem>.<ext>) if groups.json has no valid path")
    p.add_argument("--output-dir", type=Path, default=Path("data/multiref"))
    p.add_argument("--val-frac", type=float, default=0.1, help="Fraction of source images used for validation")
    p.add_argument("--min-inputs", type=int, default=2)
    p.add_argument("--max-inputs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="Seed of the train / validation split")
    args = p.parse_args()

    sources = index_sources(args.source_dir)
    skipped = Counter()
    samples = []

    # ---------------- unspecified pool ----------------
    decomp = {}
    for folder in find_decomp_folders(args.decomp_root):
        decomp.setdefault(folder.name, folder)
        inputs = list_inputs(folder)
        if not args.min_inputs <= len(inputs) <= args.max_inputs:
            skipped["unspecified: number of inputs"] += 1
            continue
        src = source_image(folder, sources)
        if src is None:
            skipped["unspecified: no source image"] += 1
            continue
        samples.append({"id": f"{folder.name}__none", "stem": stem_from_folder(folder), "pool": "unspecified",
                        "folder": str(folder), "n_inputs": len(inputs), "inputs": [str(x) for x in inputs],
                        "main": None, "targets": [str(src)], "cells": None})

    # ---------------- designated pool (manual review) ----------------
    review_dirs = sorted(d for d in args.review_dir.iterdir() if (d / "info.json").exists()) \
        if args.review_dir.is_dir() else []
    if not review_dirs:
        print(f"[warn] no review folders in {args.review_dir}: no designated samples")
    mismatched, matched = [], 0
    for rd in review_dirs:
        with open(rd / "info.json") as f:
            info = json.load(f)
        synth_folder = Path(info["decomposition_folder"])
        inputs = info["inputs"]
        same = decomp.get(synth_folder.name)
        if same is not None and same_images(inputs, list_inputs(same)):
            inputs, folder = [str(x) for x in list_inputs(same)], same
            matched += 1
        else:
            folder = synth_folder
            mismatched.append(rd.name)
        if not args.min_inputs <= len(inputs) <= args.max_inputs:
            skipped["designated: number of inputs"] += 1
            continue
        for main_dir in sorted(rd.glob("main*")):
            m = re.fullmatch(r"main(\d+)", main_dir.name)
            if not m or not (main_dir / "keep").is_dir():
                continue
            k = int(m.group(1))
            kept = sorted(x for x in (main_dir / "keep").iterdir() if x.suffix.lower() in IMAGE_EXTS)
            if not kept:
                skipped["designated: main with empty keep/"] += 1
                continue
            # the files in keep/ ARE the targets (what you see is what is trained on); this also
            # covers images added by hand that were never suggested or are not in the grid
            targets = [str(x) for x in kept]
            cells = [x.stem.split("__")[0] for x in kept]  # drop the "__reason" suffix of files moved from drop/
            samples.append({"id": f"{rd.name}__main{k}", "stem": stem_from_folder(synth_folder),
                            "pool": "designated", "folder": str(folder), "n_inputs": len(inputs),
                            "inputs": inputs, "main": k, "targets": targets, "cells": cells})

    # ---------------- split by source image (stratified) ----------------
    stems_designated = {s["stem"] for s in samples if s["pool"] == "designated"}
    stems_other = {s["stem"] for s in samples} - stems_designated
    val_stems = stratified_split(stems_designated, stems_other, args.val_frac, random.Random(args.seed))
    splits = {"train": [s for s in samples if s["stem"] not in val_stems],
              "val": [s for s in samples if s["stem"] in val_stems]}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"seed": args.seed, "val_frac": args.val_frac, "decomp_root": str(args.decomp_root),
             "review_dir": str(args.review_dir), "source_images": len(stems_designated | stems_other),
             "source_images_with_designated": len(stems_designated), "val_source_images": len(val_stems),
             "review_folders": len(review_dirs), "review_folders_matched_to_decomp_root": matched,
             "review_folders_with_different_inputs": mismatched, "skipped": dict(skipped)}
    for name, rows in splits.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)
        mains_per_folder = defaultdict(int)
        for s in rows:
            if s["pool"] == "designated":
                mains_per_folder[s["folder"]] += 1
        stats[name] = {
            "samples": len(rows),
            "by_pool": dict(Counter(s["pool"] for s in rows)),
            "by_n_inputs": {pool: {str(k): v for k, v in sorted(Counter(s["n_inputs"] for s in rows
                                                                         if s["pool"] == pool).items())}
                            for pool in ("unspecified", "designated")},
            "designated_targets": sum(len(s["targets"]) for s in rows if s["pool"] == "designated"),
            "designated_folders_with_2plus_mains": sum(v >= 2 for v in mains_per_folder.values()),
        }
    with open(args.output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    for name in ("train", "val"):
        st = stats[name]
        print(f"{name}: {st['samples']} samples {st['by_pool']}; designated targets {st['designated_targets']}; "
              f"folders with >= 2 designated mains {st['designated_folders_with_2plus_mains']}")
    print(f"source images: {stats['source_images']} ({len(stems_designated)} with designated samples, "
          f"{len(val_stems)} for validation)")
    print(f"review folders: {len(review_dirs)}, inputs identical to {args.decomp_root}: {matched}")
    if mismatched:
        print(f"[warn] {len(mismatched)} review folder(s) with inputs different from {args.decomp_root} "
              f"(kept with their own inputs): {', '.join(mismatched[:5])}{' ...' if len(mismatched) > 5 else ''}")
    if skipped:
        print("skipped: " + ", ".join(f"{k} {v}" for k, v in skipped.items()))
    print(f"manifests: {args.output_dir}")


if __name__ == "__main__":
    main()
