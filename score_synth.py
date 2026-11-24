"""
score_synth.py -- automatic choice of ONE designated-subject target per main, and calibration
against hand labels.
Last modified: 2026-11-24 (main kinds: object mains only by default, attribute mains optional with
               a presence check; defaults from the calibration on the hand-picked v2.1 data:
               --copy-max 0.88 --dino-copy-max 0.94 --lift-min -0.03 --min-main 0 --min-fusion 0)

Selection step of the automatic target selection; replaces the manual choice in select_targets.py.
Works on the outputs of synthesize_sweep.py and of synthesize_v2_1.py (any folder with a
synth_meta.json), so the hand-picked v2.1 selections can be used to calibrate it.

Per candidate target T of main k (one cell = one t, strength, seed)
-------------------------------------------------------------------
Measurements (cached in <synth dir>/target_measure.json), CLIP = image encoder of the Kandinsky
prior (as in synthesize_v2_1.py), DINO = DINOv2 (as in score_sweep.py), all cosine similarities:
    cos_main   CLIP(T, input_k)
    dino_main  DINO(T, input_k)        structure / shape of the main (CLIP is mostly category)
    lift       cos_main - CLIP(input_k, recon)
    margin     cos_main - max_{i != k} CLIP(T, input_i)
VLM (Qwen3-VL as in score_sweep.py; cached in <synth dir>/target_vlm.json):
    concepts   the inventory of the source image (score_sweep.py _sources/<stem>.json if present)
    own        the main subject of every INPUT: the same multiple-choice question as score_sweep.py
               (main_choice.json of the decomposition folder is reused if it fits)
    p_main     object main:    P(main subject of T = own concept of input k), same multiple-choice
                               question on T
               attribute main: P(yes | "Does this image show the <type> <c>?") for the own concept c
                               of input k (a color / pattern / ... is never "the main subject" of an
                               image that shows an object, so the choice question does not apply)
    fusion     mean over the other inputs' own concepts of P(yes | "Is there a <c> in this image?")

Main kinds (from the own concept of the main input)
    object     own concept of type "object"
    attribute  own concept of another type (material, color, shape, pattern, style, scene)
    unknown    no own concept ("none of these")
By default only object mains can get a pick. --attr-mains also admits attribute mains whose
concept is clearly present in the target (p_main >= --min-attr).

Gates (a cell is a candidate only if it passes all)
    copy           cos_main > --copy-max or dino_main > --dino-copy-max    near copy of the main input
    no_effect      lift < --lift-min                                       designation had no effect
    main_unknown   the main input itself has no own concept ("none of these")
    attr_main      attribute main without --attr-mains (not considered)
    main_not_top   object main: the own concept of input k is not the most likely main subject of T
    main_weak      object main: p_main < --min-main (off by default: same cells as main_not_top)
    attr_weak      attribute main: p_main < --min-attr (presence of the main concept)
    no_fusion      max presence of the other inputs' concepts < --min-fusion (off by default)
    not_distinct   distinct < --distinct-min, see below
The VLM is asked only about cells that pass copy / no_effect (all cells with --vlm-all or --labels),
and never about attribute mains without --attr-mains.

Score and pick
    score    = p_main + w_struct * dino_main + w_fuse * fusion + w_dist * distinct
    distinct = cos_main - max_{j != k} CLIP(pick of main j, input_k)
               how much more T resembles its main than the targets picked for the other mains do
    Per main the best-scoring candidate is picked. distinct depends on the picks of the other
    mains, so the picks are refined a few rounds until they no longer change.
    With --no-vlm: score = margin + w_struct * dino_main (+ w_dist * distinct), VLM gates skipped.
    <synth-root>/overrides.csv (columns folder,main,cell) replaces automatic picks: cell = a cell
    name (e.g. t0.30_s0.55_seed0), "none" for no pick, empty for automatic.

Output
------
    <synth-root>/_review/<folder>.jpeg         every cell of every main: pick (green), passed (blue),
                                               failed (red, with the reasons)
    <synth-root>/_review/overview_XX.jpeg      one row per main: main input, pick, recon, inputs
    <synth-root>/_review/cells.csv, picks.csv, summary.json
    --export-review DIR (e.g. data/selection/review): select_targets.py-style review folders with the
        pick in main<k>/keep/, read by build_dataset.py unchanged. Folders written by this script are
        listed in DIR/_exported_synth.json and rewritten on every export; other folders (e.g. made
        by select_targets.py) are never touched. Target files in the synth folders are never deleted.

Calibration (--labels)
----------------------
--labels points to review folders chosen by hand with select_targets.py: the files in
main<k>/keep/ are the hand label (keep), every other cell is dropped; a main<k>/ or a whole
review folder deleted by hand (listed in _created.txt) means nothing was kept. Only labelled
folders are scored, and the VLM is asked about every cell. Written to <synth-root>/_calibration/:
    calibration.json   gate x label counts (how many hand-kept cells each gate rejects), metric
                       quantiles of kept vs dropped cells, main-level agreement (auto pick in the
                       hand-kept set, false accepts, false rejects), overall and per main kind,
                       keep rate by main concept type
    cells_labeled.csv, mains_labeled.csv
    weights.csv        (--fit-weights) agreement for a grid of w_struct, w_fuse, w_dist; the
                       weights are reported, not applied: pass the chosen ones on the command line

Usage
-----
    # automatic picks on synthesize_sweep.py outputs, then export for build_dataset.py
    python score_synth.py --synth-root data/synth
    python score_synth.py --synth-root data/synth --export-review data/selection/review

    # calibration on the hand-picked v2.1 data (old decomposition folders by name from --decomp-root)
    python score_synth.py --synth-root data/synth_handpicked --decomp-root generate_source_handpicked \\
        --labels data/selection_handpicked/review --fit-weights

    # also attribute mains (color / pattern / ...), if their concept is clearly present in the target
    python score_synth.py --synth-root data/synth --attr-mains --min-attr 0.8

All measurements and VLM answers are cached: changing a threshold or weight takes seconds.

Decomposition folders: synth_meta.json stores the path used at synthesis time. With --decomp-root,
the folder of the same name under these roots is used FIRST (needed when the original folder was
renamed and a different folder of the same name may now be at the old path); otherwise the stored
path.
"""

import argparse
import csv
import itertools
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from score_sweep import CHOICE_PROMPT, PROMPT_VERSION, VLM, Encoders, choice_table, load_json, source_inventory

MEASURE_VERSION = 1
PRESENCE_VERSION = 1  # increase when a presence question below changes
PRESENCE_OBJECT = 'Is there a "{name}" in this image?'
PRESENCE_OTHER = 'Does this image show the {type} "{name}"?'
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
METRICS = ("cos_main", "dino_main", "lift", "margin", "p_main", "fusion", "distinct", "score")
KINDS = ("object", "attribute", "unknown")
GREEN, BLUE, RED, GRAY, BLACK = (0, 140, 0), (40, 90, 220), (200, 0, 0), (130, 130, 130), (0, 0, 0)
EXPORT_MANIFEST = "_exported_synth.json"


def parse_args():
    p = argparse.ArgumentParser(description="Pick one designated-subject target per main; calibrate on hand labels")
    p.add_argument("--synth-root", type=Path, default=Path("data/synth"),
                   help="Searched recursively for synth_meta.json (synthesize_sweep.py or synthesize_v2_1.py)")
    p.add_argument("--decomp-root", type=Path, nargs="*", default=[],
                   help="Look up decomposition folders by name here first (e.g. generate_source_handpicked)")
    p.add_argument("--source-dir", type=Path, default=Path("samples/sources"),
                   help="Fallback location of source images (<stem>.<ext>) if groups.json has no valid path")
    p.add_argument("--inventory-dir", type=Path, default=Path("output/decompose/sweep/_sources"),
                   help="Source inventories of score_sweep.py, reused if they fit the VLM model and prompt")
    p.add_argument("--labels", type=Path, default=None, help="Hand-picked review folders (select_targets.py)")
    p.add_argument("--fit-weights", action="store_true", help="With --labels: grid search of the score weights")
    p.add_argument("--export-review", type=Path, default=None, help="Write review folders with the picks here")
    p.add_argument("--limit", type=int, default=None, help="Only the first LIMIT synth folders")
    # VLM and encoders
    p.add_argument("--no-vlm", action="store_true", help="CLIP / DINOv2 only")
    p.add_argument("--vlm-all", action="store_true", help="Ask the VLM about every cell, also those failing copy / "
                                                          "no_effect (always on with --labels)")
    p.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--vlm-size", type=int, default=448)
    p.add_argument("--max-concepts", type=int, default=8, help="Concepts in a new source inventory (<= 9)")
    p.add_argument("--dino-model", default="facebook/dinov2-base")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8, help="Images per CLIP / DINOv2 batch")
    p.add_argument("--remeasure", action="store_true", help="Recompute the CLIP / DINOv2 measurements")
    p.add_argument("--reask", action="store_true", help="Ask the VLM again")
    # gates (defaults from the calibration on the hand-picked v2.1 data)
    p.add_argument("--copy-max", type=float, default=0.88, help="copy if cos_main > this")
    p.add_argument("--dino-copy-max", type=float, default=0.94, help="copy if dino_main > this")
    p.add_argument("--lift-min", type=float, default=-0.03, help="no_effect if lift < this")
    p.add_argument("--min-main", type=float, default=0.0,
                   help="Object mains: main_weak if p_main < this (0 = off; main_not_top already applies)")
    p.add_argument("--min-fusion", type=float, default=0.0,
                   help="no_fusion if every other concept has presence < this (0 = off)")
    p.add_argument("--distinct-min", type=float, default=0.0, help="not_distinct if distinct < this")
    p.add_argument("--attr-mains", action="store_true",
                   help="Also admit attribute mains (color, pattern, material, shape, style, scene); "
                        "default: object mains only")
    p.add_argument("--min-attr", type=float, default=0.8,
                   help="With --attr-mains: attr_weak if the presence of the main concept in the target < this")
    # score weights
    p.add_argument("--w-struct", type=float, default=1.0, help="Weight of dino_main")
    p.add_argument("--w-fuse", type=float, default=1.0, help="Weight of fusion")
    p.add_argument("--w-dist", type=float, default=1.0, help="Weight of distinct")
    # review images
    p.add_argument("--tile", type=int, default=180)
    p.add_argument("--sheet-cols", type=int, default=10, help="Tiles per row in the per-folder sheets")
    p.add_argument("--overview-rows", type=int, default=16, help="Rows (mains) per overview page")
    args = p.parse_args()
    args.max_concepts = min(args.max_concepts, 9)
    if args.labels:
        args.vlm_all = True
    return args


# --------------------------------------------------------------------------------------
# folder records
# --------------------------------------------------------------------------------------
def list_inputs(folder: Path):
    paths = []
    while (folder / f"input_{len(paths)}.jpeg").exists():
        paths.append(folder / f"input_{len(paths)}.jpeg")
    return paths


def resolve_folder(stored: str, roots):
    """Decomposition folder: same name under --decomp-root first, then the path stored in synth_meta.json."""
    name = Path(stored).name
    for root in roots:
        if (root / name).is_dir():
            return root / name
        hits = [p for p in root.rglob(name) if p.is_dir()]
        if hits:
            return hits[0]
    return Path(stored) if Path(stored).is_dir() else None


def find_source(folder: Path, args):
    """Source image of a decomposition folder: groups.json "image", else <source-dir>/<stem>.*"""
    g = folder / "groups.json"
    if g.exists():
        path = json.loads(g.read_text()).get("image")
        if path and Path(path).is_file():
            return Path(path)
    stem = folder.name.split("_topk")[0]
    if args.source_dir.is_dir():
        for p in sorted(args.source_dir.glob(stem + ".*")):
            if p.suffix.lower() in IMAGE_EXTS:
                return p
    return None


def build_rec(meta_path: Path, args):
    meta = json.loads(meta_path.read_text())
    synth_dir = meta_path.parent
    rel = synth_dir.relative_to(args.synth_root)
    name = "__".join(rel.parts) or synth_dir.name  # same naming as select_targets.py review folders
    folder = resolve_folder(meta["folder"], args.decomp_root)
    if folder is None:
        print(f"[skip] {name}: decomposition folder {meta['folder']} not found (use --decomp-root)")
        return None
    inputs = list_inputs(folder)
    if len(inputs) != meta["n_inputs"]:
        print(f"[skip] {name}: {len(inputs)} inputs in {folder}, synth_meta.json says {meta['n_inputs']}")
        return None
    recon = next((p for p in (folder / "recon.jpeg", synth_dir / "recon.jpeg") if p.exists()), None)
    source = find_source(folder, args)
    mains = {}
    for r in meta["results"]:
        cells = []
        for c in r["cells"]:
            if c.get("strength") is None or not (synth_dir / c["file"]).exists():
                continue  # --embed-only reference column, or file missing
            cells.append({"main": r["main"], "name": Path(c["file"]).stem, "file": c["file"],
                          "path": str(synth_dir / c["file"]), "t": c["t"], "s": c["strength"],
                          "seed": c.get("seed", meta.get("seed"))})
        if cells:
            mains[r["main"]] = {"class": r.get("class"), "cells": cells, "pick": None, "by_hand": False}
    if not mains:
        print(f"[skip] {name}: no target images")
        return None
    return {"name": name, "synth_dir": str(synth_dir), "folder": str(folder), "inputs": [str(p) for p in inputs],
            "n": len(inputs), "recon": str(recon) if recon else None, "source": str(source) if source else None,
            "stem": source.stem if source else folder.name.split("_topk")[0],
            "meta_closeness": meta.get("closeness"), "mains": mains, "own": None, "status": "ok"}


# --------------------------------------------------------------------------------------
# CLIP / DINOv2 measurements
# --------------------------------------------------------------------------------------
def encode_paths(enc: Encoders, paths, bs: int):
    cs, ds = [], []
    for i in range(0, len(paths), bs):
        c, d = enc.encode([Image.open(p).convert("RGB") for p in paths[i:i + bs]])
        cs.append(c)
        ds.append(d)
    return torch.cat(cs), torch.cat(ds)


def measure_rec(rec, enc: Encoders, args) -> int:
    """Attach clip / dino similarities to every cell; returns the number of newly encoded cells."""
    path = Path(rec["synth_dir"]) / "target_measure.json"
    fresh = lambda d: (not args.remeasure and d.get("version") == MEASURE_VERSION
                       and d.get("dino_model") == args.dino_model and d.get("inputs") == rec["inputs"]
                       and d.get("recon") == rec["recon"])
    cache = load_json(path, fresh) or {"version": MEASURE_VERSION, "dino_model": args.dino_model,
                                       "inputs": rec["inputs"], "recon": rec["recon"], "ref": None, "cells": {}}
    cells = [c for m in rec["mains"].values() for c in m["cells"]]
    missing = [c for c in cells if c["file"] not in cache["cells"]]
    r4 = lambda t: np.round(t.cpu().numpy(), 4).tolist()
    if missing or cache["ref"] is None:
        ci, di = encode_paths(enc, rec["inputs"], args.batch_size)
        ref = {"clip_pair": r4(ci @ ci.T), "dino_pair": r4(di @ di.T)}
        cr = dr = None
        if rec["recon"]:
            cr, dr = encode_paths(enc, [rec["recon"]], 1)
            ref.update(clip_recon=r4(ci @ cr[0]), dino_recon=r4(di @ dr[0]))
        else:  # no recon: closeness from synthesis time
            ref.update(clip_recon=rec["meta_closeness"], dino_recon=None)
        cache["ref"] = ref
        if missing:
            ct, dt = encode_paths(enc, [c["path"] for c in missing], args.batch_size)
            for j, c in enumerate(missing):
                cache["cells"][c["file"]] = {
                    "clip": r4(ct[j] @ ci.T), "dino": r4(dt[j] @ di.T),
                    "clip_recon": float(ct[j] @ cr[0]) if cr is not None else None,
                    "dino_recon": float(dt[j] @ dr[0]) if dr is not None else None}
        path.write_text(json.dumps(cache))
    rec["closeness"] = cache["ref"]["clip_recon"]
    for c in cells:
        c.update(cache["cells"][c["file"]])
    return len(missing)


def cheap_fields(rec, args):
    """Cheap metrics and gates of every cell."""
    n = rec["n"]
    for k, m in rec["mains"].items():
        for c in m["cells"]:
            clip, dino = c["clip"], c["dino"]
            c["cos_main"], c["dino_main"] = clip[k], dino[k]
            c["lift"] = clip[k] - rec["closeness"][k]
            c["margin"] = clip[k] - max(clip[i] for i in range(n) if i != k)
            c["cheap"] = []
            if clip[k] > args.copy_max or dino[k] > args.dino_copy_max:
                c["cheap"].append("copy")
            if c["lift"] < args.lift_min:
                c["cheap"].append("no_effect")


# --------------------------------------------------------------------------------------
# VLM
# --------------------------------------------------------------------------------------
def presence_question(name: str, kind: str) -> str:
    if kind == "object":
        return PRESENCE_OBJECT.format(name=name)
    return PRESENCE_OTHER.format(name=name, type=kind if kind in ("material", "color", "shape", "pattern",
                                                                  "style", "scene") else "concept")


def main_kind(own) -> str:
    """object / attribute / unknown, from the own concept of the main input."""
    if own is None or own["index"] == 0:
        return "unknown"
    return "object" if own["type"] == "object" else "attribute"


def get_inventory(rec, get_vlm, args, counts):
    """Source inventory: <synth-root>/_sources/<stem>.json, else score_sweep.py's, else asked."""
    fresh = lambda d: (not args.reask and d.get("model") == args.vlm_model
                       and d.get("prompt_version") == PROMPT_VERSION)
    own_path = args.synth_root / "_sources" / f"{rec['stem']}.json"
    inv = load_json(own_path, fresh)
    if inv is None and args.inventory_dir:
        inv = load_json(args.inventory_dir / f"{rec['stem']}.json", fresh)
    if inv is None:
        image = rec["source"] or rec["recon"]
        if image is None:
            return None
        if rec["source"] is None:
            print(f"[note] {rec['name']}: source image not found, inventory of recon.jpeg instead")
        inv = source_inventory(get_vlm(), image, args.max_concepts)
        counts["inventories asked"] += 1
    own_path.parent.mkdir(parents=True, exist_ok=True)
    own_path.write_text(json.dumps(inv, indent=1))
    return inv if inv.get("ok") else None


def vlm_rec(rec, get_vlm, args, counts):
    """Own concept of every input, then p_main / presence of the cells that need them."""
    inv = get_inventory(rec, get_vlm, args, counts)
    if inv is None:
        rec["status"] = "no source inventory"
        return
    concepts = inv["concepts"]
    names = [c["name"] for c in concepts]
    labels = ["none of these"] + names
    types = [None] + [c["type"] for c in concepts]
    fresh = lambda d: (not args.reask and d.get("model") == args.vlm_model
                       and d.get("prompt_version") == PROMPT_VERSION and d.get("concepts") == names)

    # main subject of every input: score_sweep.py's main_choice.json if it fits, else asked here
    synth_dir = Path(rec["synth_dir"])
    choice = load_json(Path(rec["folder"]) / "main_choice.json", fresh) or load_json(synth_dir / "input_choice.json", fresh)
    if choice is None or len(choice["p"]) != rec["n"]:
        choice = choice_table(get_vlm(), rec["inputs"], concepts)
        (synth_dir / "input_choice.json").write_text(json.dumps(choice, indent=1))
        counts["input tables asked"] += 1
    rec["own"] = []
    for row in choice["p"]:
        i = int(np.argmax(row))
        rec["own"].append({"index": i, "name": labels[i], "type": types[i], "p": float(row[i])})

    # target answers
    path = synth_dir / "target_vlm.json"
    cache = load_json(path, lambda d: fresh(d) and d.get("presence_version") == PRESENCE_VERSION) or {
        "model": args.vlm_model, "prompt_version": PROMPT_VERSION, "presence_version": PRESENCE_VERSION,
        "concepts": names, "cells": {}}
    prompt = CHOICE_PROMPT.format(options="\n".join(f"{i + 1}. {n}" for i, n in enumerate(names)))
    changed = False
    for k, m in rec["mains"].items():
        own = rec["own"][k]
        m["kind"] = main_kind(own)
        others = []
        for j, o in enumerate(rec["own"]):
            if j != k and o["index"] > 0 and o["index"] != own["index"] and o["name"] not in [x[0] for x in others]:
                others.append((o["name"], o["type"]))
        m["others"] = others
        if m["kind"] == "unknown" or (m["kind"] == "attribute" and not args.attr_mains):
            continue  # these mains fail main_unknown / attr_main without any VLM answer on the targets
        # object mains: choice question; attribute mains: presence of their own concept instead
        needed = others + ([(own["name"], own["type"])] if m["kind"] == "attribute" else [])
        for c in m["cells"]:
            if not args.vlm_all and c["cheap"]:
                continue
            entry = cache["cells"].setdefault(c["file"], {})
            todo_choice = m["kind"] == "object" and "choice" not in entry
            todo_pres = [(n, t) for n, t in needed if n not in entry.get("presence", {})]
            if todo_choice or todo_pres:
                vlm = get_vlm()
                image = vlm.load(c["path"])
                if todo_choice:
                    entry["choice"] = vlm.choice_probs(image, prompt, len(names) + 1)
                pres = entry.setdefault("presence", {})
                for n, t in todo_pres:
                    pres[n] = vlm.yes_prob(image, presence_question(n, t))
                counts["cells asked"] += 1
                changed = True
            c["vlm"] = entry
    if changed:
        path.write_text(json.dumps(cache))


# --------------------------------------------------------------------------------------
# gates, score, picks
# --------------------------------------------------------------------------------------
def gate_fields(rec, args):
    """p_main, fusion and every gate except not_distinct (which depends on the picks)."""
    for k, m in rec["mains"].items():
        own = rec["own"][k] if rec["own"] else None
        m["own"] = own
        m.setdefault("kind", main_kind(own) if own is not None else None)
        for c in m["cells"]:
            reasons = list(c["cheap"])
            c["p_main"] = c["fusion"] = None
            if not args.no_vlm:
                v = c.get("vlm")
                if own is None:
                    reasons.append("vlm_failed")
                elif m["kind"] == "unknown":
                    reasons.append("main_unknown")
                elif m["kind"] == "attribute" and not args.attr_mains:
                    reasons.append("attr_main")
                elif v is None:
                    reasons.append("vlm_not_asked")  # only cells that already failed copy / no_effect
                else:
                    pres_all = v.get("presence", {})
                    if m["kind"] == "object":
                        ch = v["choice"]
                        c["p_main"] = ch[own["index"]]
                        if int(np.argmax(ch)) != own["index"]:
                            reasons.append("main_not_top")
                        if c["p_main"] < args.min_main:
                            reasons.append("main_weak")
                    else:  # attribute main: presence of its own concept in the target
                        c["p_main"] = pres_all.get(own["name"], 0.0)
                        if c["p_main"] < args.min_attr:
                            reasons.append("attr_weak")
                    pres = [pres_all[n] for n, _ in m.get("others", []) if n in pres_all]
                    c["fusion"] = float(np.mean(pres)) if pres else 0.0
                    if pres and max(pres) < args.min_fusion:
                        reasons.append("no_fusion")
            c["pre"] = reasons
            c["ok"] = not reasons


def base_score(c, w, no_vlm: bool) -> float:
    if no_vlm:
        return c["margin"] + w["struct"] * c["dino_main"]
    return c["p_main"] + w["struct"] * c["dino_main"] + w["fuse"] * c["fusion"]


def select_rec(rec, w, args, overrides=None, rounds: int = 5) -> dict:
    """Pick per main (sets distinct, score, reasons, verdict of every cell). Returns {main: cell name or None}."""
    mains = sorted(rec["mains"])
    cells = {k: rec["mains"][k]["cells"] for k in mains}
    pick = {}
    for k in mains:
        ok = [c for c in cells[k] if c["ok"]]
        pick[k] = max(ok, key=lambda c: base_score(c, w, args.no_vlm)) if ok else None

    def total(c):
        return base_score(c, w, args.no_vlm) + w["dist"] * (c["distinct"] or 0.0)

    for _ in range(rounds):
        changed = False
        for k in mains:
            others = [pick[j] for j in mains if j != k and pick[j] is not None]
            best = None
            for c in cells[k]:
                c["distinct"] = c["clip"][k] - max(o["clip"][k] for o in others) if others else None
                if not c["ok"] or (c["distinct"] is not None and c["distinct"] < args.distinct_min):
                    continue
                if best is None or total(c) > total(best):
                    best = c
            if best is not pick[k]:
                pick[k], changed = best, True
        if not changed:
            break

    for k in mains:
        m = rec["mains"][k]
        m["by_hand"] = False
        value = (overrides or {}).get((rec["name"], k), "")
        if value:
            m["by_hand"] = True
            pick[k] = None if value.lower() == "none" else next((c for c in cells[k] if c["name"] == value), None)
            if value.lower() != "none" and pick[k] is None:
                print(f"[warn] overrides.csv: no cell {value} in {rec['name']} main{k}")
        for c in cells[k]:
            c["reasons"] = list(c["pre"])
            if c["ok"] and c["distinct"] is not None and c["distinct"] < args.distinct_min:
                c["reasons"].append("not_distinct")
            c["score"] = round(total(c), 4) if c["ok"] else None
            if c is pick[k]:
                c["verdict"] = "pick (by hand)" if m["by_hand"] else "pick"
            elif not c["reasons"]:
                c["verdict"] = "passed, lower score"
            else:
                c["verdict"] = "failed: " + ",".join(c["reasons"])
        m["pick"] = pick[k]["name"] if pick[k] is not None else None
        if pick[k] is None:
            why = Counter(r for c in cells[k] for r in c["reasons"]).most_common(3)
            m["no_pick_reason"] = "set by hand" if m["by_hand"] else ", ".join(f"{r} {n}" for r, n in why)
    return {k: rec["mains"][k]["pick"] for k in mains}


def read_overrides(path: Path) -> dict:
    """overrides.csv -> {(folder, main): cell name / "none"}."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {(row["folder"].strip(), int(row["main"])): (row.get("cell") or "").strip()
                for row in csv.DictReader(f) if row.get("folder") and row.get("main")}


# --------------------------------------------------------------------------------------
# review images and tables
# --------------------------------------------------------------------------------------
def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def fmt(x, signed=False):
    if x is None:
        return "-"
    return f"{x:+.2f}" if signed else f"{x:.2f}"


def draw_rows(rows, path: Path, tile: int, title: str = "", n_lines: int = 5):
    """rows: list of rows (None = separator); a row is a list of cells (image path / None, [(text, color)], frame)."""
    f, line, gap, sep = font(12), 14, 8, 10
    cap = n_lines * line + 4
    n_cols = max(len(r) for r in rows if r is not None)
    title_h = (title.count("\n") + 1) * (line + 2) + gap if title else 0
    H = title_h + gap + sum(sep if r is None else tile + cap + gap for r in rows)
    W = gap + n_cols * (tile + gap)
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.multiline_text((gap, gap // 2), title, fill=BLACK, font=font(13), spacing=2)
    y = title_h + gap // 2
    for row in rows:
        if row is None:
            draw.line([(gap, y + sep // 2), (W - gap, y + sep // 2)], fill=GRAY, width=1)
            y += sep
            continue
        for i, cell in enumerate(row):
            if cell is None:
                continue
            src, lines, frame = cell
            x = gap + i * (tile + gap)
            img = Image.open(src).convert("RGB").resize((tile, tile)) if src and Path(src).exists() \
                else Image.new("RGB", (tile, tile), (225, 225, 225))
            canvas.paste(img, (x, y))
            if frame is not None:
                draw.rectangle([x - 2, y - 2, x + tile + 1, y + tile + 1], outline=frame, width=4)
            for j, (text, color) in enumerate(lines[:n_lines]):
                draw.text((x + 1, y + tile + 3 + j * line), text, fill=color, font=f)
        y += tile + cap + gap
    canvas.save(path, quality=88)


def input_lines(rec, i):
    own = rec["own"][i] if rec["own"] else None
    tag = " MAIN" if i in rec["mains"] else ""
    lines = [(f"input_{i}{tag}", BLACK)]
    if own:
        lines.append((f"{own['name'][:22]}", BLACK))
        lines.append((f"{own['type'] or ''} p={own['p']:.2f}", GRAY))
    lines.append((f"close {fmt(rec['closeness'][i])}", GRAY))
    return lines


def cell_order(m):
    """Pick first, then passed cells by score, then failed cells in grid order."""
    cells = m["cells"]
    head = [c for c in cells if c["verdict"].startswith("pick")]
    passed = sorted((c for c in cells if c["verdict"] == "passed, lower score"), key=lambda c: -c["score"])
    failed = [c for c in cells if c["verdict"].startswith("failed")]
    return head + passed + failed


def draw_sheet(rec, path: Path, args):
    rows = [[(rec["recon"], [("recon", BLACK)], None)]
            + [(p, input_lines(rec, i), None) for i, p in enumerate(rec["inputs"])]]
    for k in sorted(rec["mains"]):
        m = rec["mains"][k]
        own = m.get("own")
        head = (rec["inputs"][k], [(f"main = input_{k}", BLACK), ((own or {}).get("name", "?")[:22], BLACK),
                                   (f"{m.get('kind') or '?'} main, {m['class'] or ''}", GRAY),
                                   (f"pick: {m['pick'] or 'none'}", GREEN if m["pick"] else RED)], None)
        cells = []
        for c in cell_order(m):
            color = GREEN if c["verdict"].startswith("pick") else BLUE if c["verdict"].startswith("passed") else RED
            verdict = c["verdict"].replace("failed: ", "x ")
            cells.append((c["path"], [(c["name"], BLACK),
                                      (f"p {fmt(c['p_main'])} fuse {fmt(c['fusion'])}", BLACK),
                                      (f"dino {fmt(c['dino_main'])} lift {fmt(c['lift'], True)}", BLACK),
                                      (f"dist {fmt(c['distinct'], True)} sc {fmt(c['score'])}", BLACK),
                                      (verdict[:30], color)], color))
        per_row = max(1, args.sheet_cols - 1)
        rows.append(None)
        for i in range(0, max(len(cells), 1), per_row):
            rows.append([head if i == 0 else None] + cells[i:i + per_row])
    title = (f"{rec['name']}\np = P(main concept is the subject) (object main) / presence of the main concept "
             "(attribute main), fuse = presence of the other inputs' concepts,\n"
             "dino = DINO(T, main), lift = CLIP(T, main) - CLIP(main, recon), dist = distinct, sc = score")
    draw_rows(rows, path, args.tile, title)


def write_review(recs, args) -> dict:
    review = args.synth_root / "_review"
    review.mkdir(parents=True, exist_ok=True)
    for rec in tqdm(recs, desc="review sheets"):
        draw_sheet(rec, review / f"{rec['name']}.jpeg", args)

    # overview: one row per main
    rows = []
    for rec in recs:
        for k in sorted(rec["mains"]):
            m = rec["mains"][k]
            c = next((c for c in m["cells"] if c["name"] == m["pick"]), None)
            own = (m.get("own") or {}).get("name", "?")
            main_cell = (rec["inputs"][k], [(rec["name"][:26], BLACK),
                                            (f"main{k} ({m.get('kind') or '?'}): {own}"[:30], BLACK)], None)
            if c is not None:
                pick_cell = (c["path"], [(c["name"] + (" (hand)" if m["by_hand"] else ""), GREEN),
                                         (f"p {fmt(c['p_main'])} fuse {fmt(c['fusion'])} sc {fmt(c['score'])}", BLACK)],
                             GREEN)
            else:
                pick_cell = (None, [("no pick", RED), ((m.get("no_pick_reason") or "")[:30], RED)], RED)
            others = [(p, [(f"input_{i}", GRAY)], None) for i, p in enumerate(rec["inputs"]) if i != k]
            rows.append([main_cell, pick_cell, (rec["recon"], [("recon", GRAY)], None)] + others)
    for old in review.glob("overview_*.jpeg"):
        old.unlink()
    for page, i in enumerate(range(0, len(rows), args.overview_rows)):
        draw_rows(rows[i:i + args.overview_rows], review / f"overview_{page:02d}.jpeg", args.tile,
                  "main input | pick | recon | other inputs", n_lines=2)

    # tables
    cell_rows, pick_rows = [], []
    for rec in recs:
        for k in sorted(rec["mains"]):
            m = rec["mains"][k]
            own = m.get("own") or {}
            for c in m["cells"]:
                cell_rows.append({"folder": rec["name"], "main": k, "class": m["class"], "cell": c["name"],
                                  "t": c["t"], "s": c["s"], "seed": c["seed"], "verdict": c["verdict"],
                                  "reasons": " ".join(c["reasons"]),
                                  **{x: "" if c.get(x) is None else round(c[x], 4) for x in METRICS},
                                  "path": c["path"]})
            pick_rows.append({"folder": rec["name"], "main": k, "kind": m.get("kind"), "own": own.get("name"),
                              "own_type": own.get("type"),
                              "own_p": own.get("p"), "class": m["class"], "pick": m["pick"] or "",
                              "by_hand": m["by_hand"], "no_pick_reason": "" if m["pick"] else m.get("no_pick_reason", ""),
                              "cells": len(m["cells"]), "passed": sum(1 for c in m["cells"] if not c["reasons"])})
    write_csv(review / "cells.csv", cell_rows)
    write_csv(review / "picks.csv", pick_rows)

    reasons = Counter(r for row in cell_rows for r in row["reasons"].split())
    picks = [r for r in pick_rows if r["pick"]]
    summary = {
        "folders": len(recs), "mains": len(pick_rows), "mains_with_pick": len(picks),
        "mains_by_hand": sum(r["by_hand"] for r in pick_rows),
        "folders_with_pick": len({r["folder"] for r in picks}),
        "picks_per_folder": dict(Counter(Counter(r["folder"] for r in picks).values())),
        "cells": len(cell_rows), "cells_passed": sum(1 for r in cell_rows if not r["reasons"]),
        "fail_reasons": dict(reasons.most_common()),
        "picked_cells": {"t": dict(Counter(r["t"] for r in cell_rows if r["verdict"].startswith("pick"))),
                         "s": dict(Counter(r["s"] for r in cell_rows if r["verdict"].startswith("pick"))),
                         "seed": dict(Counter(r["seed"] for r in cell_rows if r["verdict"].startswith("pick")))},
        "mains_by_kind": {str(kd): {"mains": sum(r["kind"] == kd for r in pick_rows),
                                    "with_pick": sum(r["kind"] == kd and bool(r["pick"]) for r in pick_rows)}
                          for kd in sorted({r["kind"] for r in pick_rows}, key=str)},
        "no_pick_mains_by_own_type": dict(Counter(str(r["own_type"]) for r in pick_rows if not r["pick"])),
        "settings": settings_dict(args)}
    (review / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_csv(path: Path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def settings_dict(args) -> dict:
    keys = ("copy_max", "dino_copy_max", "lift_min", "min_main", "min_fusion", "distinct_min", "attr_mains",
            "min_attr", "w_struct", "w_fuse", "w_dist", "no_vlm", "vlm_model", "dino_model")
    return {k: getattr(args, k) for k in keys}


# --------------------------------------------------------------------------------------
# export: review folders for build_dataset.py
# --------------------------------------------------------------------------------------
def export_review(recs, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    manifest_path = dst / EXPORT_MANIFEST
    manifest = set(json.loads(manifest_path.read_text())) if manifest_path.exists() else set()
    written = removed = skipped = 0
    for rec in recs:
        d = dst / rec["name"]
        picked = {k: m for k, m in rec["mains"].items() if m["pick"]}
        if d.exists() and rec["name"] not in manifest:
            print(f"[export] {d} exists and was not written by this script: left unchanged")
            skipped += 1
            continue
        if d.exists():
            shutil.rmtree(d)
        if not picked:
            if rec["name"] in manifest:
                manifest.discard(rec["name"])
                removed += 1
            continue
        d.mkdir(parents=True)
        info = {"decomposition_folder": rec["folder"], "synth_dir": rec["synth_dir"], "inputs": rec["inputs"],
                "cells": {f"main{k}": {c["name"]: c["path"] for c in m["cells"]} for k, m in rec["mains"].items()},
                "picked_by": "score_synth.py",
                "picks": {f"main{k}": m["pick"] for k, m in rec["mains"].items()}}
        (d / "info.json").write_text(json.dumps(info, indent=2))
        for k, m in picked.items():
            keep = d / f"main{k}" / "keep"
            keep.mkdir(parents=True)
            shutil.copy2(rec["inputs"][k], d / f"main{k}" / "_main.jpeg")
            c = next(c for c in m["cells"] if c["name"] == m["pick"])
            shutil.copy2(c["path"], keep / f"{c['name']}.jpeg")
        manifest.add(rec["name"])
        written += 1
    manifest_path.write_text(json.dumps(sorted(manifest), indent=1))
    print(f"export -> {dst}: {written} review folder(s) written, {removed} removed (no pick any more), "
          f"{skipped} left unchanged (not ours)")


# --------------------------------------------------------------------------------------
# calibration against hand labels
# --------------------------------------------------------------------------------------
def read_labels(label_dir: Path) -> dict:
    """{review folder name: {"removed": bool, "mains": {k: set of kept file stems}}}"""
    labels = {}
    for d in sorted(label_dir.iterdir()):
        if not (d.is_dir() and (d / "info.json").exists()):
            continue
        mains = {}
        for md in d.iterdir():
            hit = re.fullmatch(r"main(\d+)", md.name)
            if hit and md.is_dir():
                keep = md / "keep"
                mains[int(hit.group(1))] = {x.stem.split("__")[0] for x in keep.iterdir()
                                            if x.suffix.lower() in IMAGE_EXTS} if keep.is_dir() else set()
        labels[d.name] = {"removed": False, "mains": mains}
    created = label_dir / "_created.txt"
    if created.exists():
        for name in (x.strip() for x in created.read_text().splitlines()):
            if name and name not in labels:
                labels[name] = {"removed": True, "mains": {}}
    return labels


def label_of(rec, labels):
    return labels.get(rec["name"]) or labels.get(Path(rec["synth_dir"]).name)


def hand_sets(rec, lab) -> dict:
    """{main: set of hand-kept cell names of the grid} (empty set = nothing kept); hand-added images are ignored."""
    out = {}
    for k, m in rec["mains"].items():
        names = {c["name"] for c in m["cells"]}
        kept = set() if lab["removed"] else lab["mains"].get(k, set())
        out[k] = kept & names
        m["hand_added"] = len(kept - names)
    return out


def agreement(labeled, picks) -> dict:
    """Main-level comparison of automatic picks with the hand labels."""
    c = Counter()
    for rec, hand in labeled:
        for k, kept in hand.items():
            auto = picks[(rec["name"], k)]
            if auto is None and not kept:
                c["both_none"] += 1
            elif auto is None:
                c["false_reject"] += 1
            elif not kept:
                c["false_accept"] += 1
            elif auto in kept:
                c["top1_match"] += 1
            else:
                c["pick_not_kept"] += 1
    both = c["top1_match"] + c["pick_not_kept"]
    total = sum(c.values())
    return {**{k: c[k] for k in ("top1_match", "pick_not_kept", "false_accept", "false_reject", "both_none")},
            "mains": total, "top1_rate": round(c["top1_match"] / both, 3) if both else None,
            "accept_reject_agreement": round((both + c["both_none"]) / total, 3) if total else None}


def by_kind_subset(labeled, kind):
    """The labelled (rec, hand) pairs restricted to the mains of one kind."""
    out = []
    for rec, hand in labeled:
        sub = {k: v for k, v in hand.items() if rec["mains"][k].get("kind") == kind}
        if sub:
            out.append((rec, sub))
    return out


def quantiles(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if not len(v):
        return None
    q = np.percentile(v, [0, 10, 50, 90, 100])
    return {"n": int(len(v)), "min": round(q[0], 3), "p10": round(q[1], 3), "median": round(q[2], 3),
            "p90": round(q[3], 3), "max": round(q[4], 3)}


def calibrate(recs, labels, args, weights):
    out = args.synth_root / "_calibration"
    out.mkdir(parents=True, exist_ok=True)
    labeled = [(rec, hand_sets(rec, label_of(rec, labels))) for rec in recs]

    # weight grid (gates fixed): agreement of the automatic picks for every combination
    fit_rows = []
    if args.fit_weights:
        grid = (0.0, 0.5, 1.0, 2.0)
        for ws, wf, wd in tqdm(list(itertools.product(grid, grid, grid)), desc="weight grid"):
            w = {"struct": ws, "fuse": wf, "dist": wd}
            picks = {(rec["name"], k): v for rec, _ in labeled for k, v in select_rec(rec, w, args).items()}
            fit_rows.append({"w_struct": ws, "w_fuse": wf, "w_dist": wd, **agreement(labeled, picks)})
        fit_rows.sort(key=lambda r: (-(r["top1_rate"] or 0), r["false_accept"]))
        write_csv(out / "weights.csv", fit_rows)

    # final picks with the weights of the command line (overrides are not applied in calibration)
    picks = {(rec["name"], k): v for rec, _ in labeled for k, v in select_rec(rec, weights, args).items()}
    cell_rows, main_rows = [], []
    kept_cells, dropped_cells = [], []
    for rec, hand in labeled:
        for k, m in rec["mains"].items():
            own = m.get("own") or {}
            for c in m["cells"]:
                is_kept = c["name"] in hand[k]
                (kept_cells if is_kept else dropped_cells).append(c)
                cell_rows.append({"folder": rec["name"], "main": k, "cell": c["name"], "hand": "keep" if is_kept else "drop",
                                  "auto": c["verdict"], "reasons": " ".join(c["reasons"]),
                                  **{x: "" if c.get(x) is None else round(c[x], 4) for x in METRICS}})
            main_rows.append({"folder": rec["name"], "main": k, "kind": m.get("kind"), "own": own.get("name"),
                              "own_type": own.get("type"),
                              "own_p": own.get("p"), "class": m["class"], "hand_kept": " ".join(sorted(hand[k])),
                              "hand_added_images": m.get("hand_added", 0), "auto_pick": m["pick"] or "",
                              "match": bool(m["pick"]) and m["pick"] in hand[k]})
    write_csv(out / "cells_labeled.csv", cell_rows)
    write_csv(out / "mains_labeled.csv", main_rows)

    # how many hand-kept / dropped cells each gate rejects
    gates = sorted({r for c in kept_cells + dropped_cells for r in c["reasons"]})
    gate_table = {g: {"kept_by_hand": sum(g in c["reasons"] for c in kept_cells),
                      "dropped_by_hand": sum(g in c["reasons"] for c in dropped_cells)} for g in gates}
    by_type = defaultdict(lambda: Counter())
    for r in main_rows:
        by_type[str(r["own_type"])]["mains"] += 1
        by_type[str(r["own_type"])]["kept_by_hand"] += bool(r["hand_kept"])
    report = {
        "folders": len(labeled), "mains": len(main_rows),
        "cells": {"kept_by_hand": len(kept_cells), "dropped_by_hand": len(dropped_cells),
                  "kept_passing_all_gates": sum(not c["reasons"] for c in kept_cells),
                  "dropped_passing_all_gates": sum(not c["reasons"] for c in dropped_cells)},
        "gates": gate_table,
        "metrics_kept": {x: quantiles(c.get(x) for c in kept_cells) for x in METRICS},
        "metrics_dropped": {x: quantiles(c.get(x) for c in dropped_cells) for x in METRICS},
        "agreement": agreement(labeled, picks),
        "agreement_by_kind": {kd: agreement(by_kind_subset(labeled, kd), picks) for kd in KINDS},
        "keep_rate_by_main_type": {t: dict(v) for t, v in by_type.items()},
        "hand_added_images_ignored": sum(r["hand_added_images"] for r in main_rows),
        "weights_used": weights, "best_weights": fit_rows[:5], "settings": settings_dict(args)}
    (out / "calibration.json").write_text(json.dumps(report, indent=2))

    a, cc = report["agreement"], report["cells"]
    print(f"\ncalibration on {report['folders']} folder(s), {report['mains']} main(s):")
    print(f"  cells kept by hand {cc['kept_by_hand']}, passing all gates {cc['kept_passing_all_gates']}; "
          f"dropped by hand {cc['dropped_by_hand']}, passing all gates {cc['dropped_passing_all_gates']}")
    print("  gate: rejects kept / dropped cells: "
          + ", ".join(f"{g} {v['kept_by_hand']}/{v['dropped_by_hand']}" for g, v in gate_table.items()))
    print(f"  mains: top-1 match {a['top1_match']}, pick not in hand set {a['pick_not_kept']}, false accept "
          f"{a['false_accept']}, false reject {a['false_reject']}, both none {a['both_none']}; "
          f"top-1 rate {a['top1_rate']}, accept/reject agreement {a['accept_reject_agreement']}")
    for kd, ak in report["agreement_by_kind"].items():
        if ak["mains"]:
            print(f"    {kd} mains {ak['mains']}: top-1 match {ak['top1_match']}, pick not in hand set "
                  f"{ak['pick_not_kept']}, false accept {ak['false_accept']}, false reject {ak['false_reject']}, "
                  f"both none {ak['both_none']}; top-1 rate {ak['top1_rate']}")
    for x in ("cos_main", "dino_main", "lift", "p_main", "fusion"):
        k_, d_ = report["metrics_kept"][x], report["metrics_dropped"][x]
        if k_ and d_:
            print(f"  {x:9s} kept p10/med/p90 {k_['p10']}/{k_['median']}/{k_['p90']}   "
                  f"dropped {d_['p10']}/{d_['median']}/{d_['p90']}")
    if fit_rows:
        print("  best weights (w_struct, w_fuse, w_dist -> top-1 rate, false accepts): "
              + "; ".join(f"{r['w_struct']:g},{r['w_fuse']:g},{r['w_dist']:g} -> {r['top1_rate']}, {r['false_accept']}"
                          for r in fit_rows[:5]))
    print(f"  details: {out}/calibration.json, cells_labeled.csv, mains_labeled.csv"
          + (", weights.csv" if fit_rows else ""))


# --------------------------------------------------------------------------------------
def main():
    args = parse_args()
    t0, counts = time.time(), Counter()
    metas = sorted(p for p in args.synth_root.rglob("synth_meta.json")
                   if not any(part.startswith("_") for part in p.relative_to(args.synth_root).parts))
    labels = read_labels(args.labels) if args.labels else None
    recs = []
    for mp in metas:
        rec = build_rec(mp, args)
        if rec is None:
            continue
        if labels is not None and label_of(rec, labels) is None:
            continue  # calibration: only labelled folders
        recs.append(rec)
    recs = recs[: args.limit]
    if not recs:
        raise SystemExit(f"no usable synth folders under {args.synth_root}" + (" with labels" if labels else ""))
    print(f"{len(recs)} synth folder(s), {sum(len(r['mains']) for r in recs)} main(s), "
          f"{sum(len(m['cells']) for r in recs for m in r['mains'].values())} cell(s)")

    # CLIP / DINOv2
    enc = Encoders(args.dino_model, args.device)
    for rec in tqdm(recs, desc="CLIP / DINOv2"):
        counts["cells encoded"] += measure_rec(rec, enc, args)
        cheap_fields(rec, args)
    del enc
    torch.cuda.empty_cache()

    # VLM
    vlm = None

    def get_vlm():
        nonlocal vlm
        vlm = vlm or VLM(args.vlm_model, args.device, args.vlm_size)
        return vlm

    if not args.no_vlm:
        for rec in tqdm(recs, desc="VLM"):
            vlm_rec(rec, get_vlm, args, counts)
    for rec in recs:
        gate_fields(rec, args)

    weights = {"struct": args.w_struct, "fuse": args.w_fuse, "dist": args.w_dist}
    if labels is not None:
        calibrate(recs, labels, args, weights)
    overrides = read_overrides(args.synth_root / "overrides.csv")
    for rec in recs:
        select_rec(rec, weights, args, overrides)
    summary = write_review(recs, args)
    if args.export_review:
        export_review(recs, args.export_review)

    print(f"\nscored in {time.time() - t0:.0f} s; new work: "
          + (", ".join(f"{k} {v}" for k, v in counts.items() if v) or "none (all cached)"))
    print(f"mains with a pick: {summary['mains_with_pick']} / {summary['mains']} "
          f"({summary['mains_by_hand']} set by hand); folders with a pick: {summary['folders_with_pick']} / "
          f"{summary['folders']}; picks per folder: {summary['picks_per_folder']}")
    print("fail reasons (cells): " + ", ".join(f"{k} {v}" for k, v in summary["fail_reasons"].items()))
    print(f"review: {args.synth_root / '_review'} (<folder>.jpeg, overview_*.jpeg, cells.csv, picks.csv, summary.json); "
          f"change picks in {args.synth_root / 'overrides.csv'} (folder,main,cell)")


if __name__ == "__main__":
    main()
