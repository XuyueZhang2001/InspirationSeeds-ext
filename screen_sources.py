"""
screen_sources.py -- automatic first pass: which decomposition folders of data/no_main can serve as
sources of designated-main targets (synthesize_v2_1.py). Candidates are copied to
generate_source/screening/ for manual review.
Last modified: 2026-11-19

Data layout
    output/decompose/<run>/     raw decompose_v7_2.py runs (one per setting)
    data/no_main/               hand-picked decompositions of all settings (unspecified pool of
                                build_dataset.py, and the input of this script)
    generate_source/screening/  candidates of this script + _report/ (table, contact sheets)
    generate_source/            folders you move here from screening/ after review
                                (synthesize_v2_1.py skips screening/)
    data/synth/, data/selection/   synthesize_v2_1.py and select_targets.py

A folder is useful for designated targets if at least two of its inputs show clearly DIFFERENT,
recognizable subjects (e.g. a woman, flowers, a statue), not variants of one object (shape /
material / color of the same vase) and not content unrelated to the source (hallucinated
objects). This script scores that with image-level CLIP features of the Kandinsky 2.2 prior
(the encoder synthesize_v2_1.py uses for its closeness) and copies the candidates.
It is a lenient first pass: you prune by hand afterwards.

Measures per folder (inputs re-encoded from input_k.jpeg; source = groups.json "image")
    noun_k        top-1 noun of input k (CLIP retrieval over decompose_v7_2.BUILTIN_NOUNS,
                  "a photo of a {noun}"), with its similarity
    rel_k         cos(input_k, source): how much input k belongs to the source image.
                  Inputs with rel_k < --min-relevance are flagged "low" (often hallucinated
                  content) and not counted as subjects
    pair_cos      cosine between inputs (image level)
    n_subjects    distinct subjects among the relevant inputs: two inputs are the same
                  subject if they share the top-1 noun or pair_cos >= --same-cos
    diversity     1 - mean pair_cos among the relevant inputs
    score         n_subjects + diversity   (sorting key)
    candidate     >= 2 inputs, decomposable (groups.json), n_subjects >= --min-subjects

Copying (skipped with --dry-run)
    Candidates are copied (whole folder) to --screen-dir (generate_source/screening/) unless a
    folder of that name exists in --screen-dir or directly in --generate-source (already
    reviewed). Copied names are appended to <screen-dir>/_screened.txt; a folder listed there is
    never copied again, so candidates you delete stay deleted (--recopy ignores the list).
    Existing folders are never touched.

Manual review
    keep:    move <screen-dir>/<folder>/ to generate_source/<folder>/
    reject:  delete <screen-dir>/<folder>/
    Then run synthesize_v2_1.py on generate_source (screening/ is excluded by default).

Report (--report-dir, default <screen-dir>/_report)
    screen.csv                 one line per folder: measures, candidate, reason, copied
    candidates_XX.jpeg         contact sheets of the candidates (best score first)
    rejected_XX.jpeg           contact sheets of the other folders (to spot misses)
    summary.json               counts, thresholds, how the folders already in generate_source/
                               (your earlier manual choice) score: use it to adjust the
                               thresholds; quantiles of rel and max_pair_cos
Contact sheet row: source, then the inputs with "noun rel"; a red caption marks a low-
relevance input; the row title shows the score and the reason.

Usage
-----
    # report only (no copying), to check the thresholds first
    python screen_sources.py --dry-run

    # copy the candidates to generate_source/screening/
    python screen_sources.py

    # stricter
    python screen_sources.py --same-cos 0.75 --min-relevance 0.5
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

from decompose_v7_2 import BUILTIN_NOUNS, IMAGE_EXTS

PRIOR = "kandinsky-community/kandinsky-2-2-prior"
RECORD = "_screened.txt"


# --------------------------------------------------------------------------------------
# CLIP (image and text encoders of the Kandinsky 2.2 prior)
# --------------------------------------------------------------------------------------
class Clip:
    def __init__(self, device):
        self.device = device
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.proc = CLIPImageProcessor.from_pretrained(PRIOR, subfolder="image_processor")
        self.vision = CLIPVisionModelWithProjection.from_pretrained(PRIOR, subfolder="image_encoder",
                                                                    torch_dtype=dtype).to(device).eval()
        self.tok = CLIPTokenizer.from_pretrained(PRIOR, subfolder="tokenizer")
        self.text = CLIPTextModelWithProjection.from_pretrained(PRIOR, subfolder="text_encoder",
                                                                torch_dtype=dtype).to(device).eval()
        self.dtype = dtype

    @torch.inference_mode()
    def images(self, images):
        px = self.proc(images, return_tensors="pt").pixel_values.to(self.device, self.dtype)
        return torch.nn.functional.normalize(self.vision(px).image_embeds.float(), dim=-1)

    @torch.inference_mode()
    def texts(self, texts, batch=64):
        out = []
        for i in range(0, len(texts), batch):
            t = self.tok(texts[i:i + batch], padding="max_length", max_length=self.tok.model_max_length,
                         truncation=True, return_tensors="pt")
            o = self.text(input_ids=t.input_ids.to(self.device), attention_mask=t.attention_mask.to(self.device))
            out.append(o.text_embeds.float())
        return torch.nn.functional.normalize(torch.cat(out), dim=-1)


# --------------------------------------------------------------------------------------
# Folders
# --------------------------------------------------------------------------------------
def list_inputs(folder: Path):
    paths, i = [], 0
    while (folder / f"input_{i}.jpeg").exists():
        paths.append(folder / f"input_{i}.jpeg")
        i += 1
    return paths


def source_image(folder: Path, info: dict, source_dir: Path):
    """groups.json "image" if it exists, else <stem>.* in source_dir."""
    p = info.get("image")
    if p and Path(p).exists():
        return Path(p)
    stem = folder.name.split("_topk")[0]
    for q in sorted(source_dir.glob(f"{stem}.*")):
        if q.suffix.lower() in IMAGE_EXTS:
            return q
    return None


def same_subject_groups(idx, nouns, pair, same_cos):
    """Number of groups when inputs sharing the top-1 noun or with pair cos >= same_cos are merged."""
    parent = {i: i for i in idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in idx:
        for b in idx:
            if a < b and (nouns[a] == nouns[b] or pair[a][b] >= same_cos):
                parent[find(a)] = find(b)
    return len({find(i) for i in idx})


def screen_folder(folder: Path, clip: Clip, noun_bank, args):
    rec = {"folder": folder.name, "n_inputs": 0, "candidate": False, "reason": "", "score": 0.0}
    inputs = list_inputs(folder)
    rec["n_inputs"] = len(inputs)
    info = {}
    if (folder / "groups.json").exists():
        with open(folder / "groups.json") as f:
            info = json.load(f)
    rec["decomposable"] = info.get("decomposable", True)
    if len(inputs) < 2:
        rec["reason"] = "fewer than 2 inputs"
        return rec, None
    src = source_image(folder, info, args.source_dir)
    if src is None:
        rec["reason"] = "source image not found"
        return rec, None

    imgs = [Image.open(p).convert("RGB") for p in inputs]
    src_img = Image.open(src).convert("RGB")
    f = clip.images(imgs + [src_img])
    f_in, f_src = f[:-1], f[-1]
    n = len(inputs)

    sims = f_in @ noun_bank.T
    top = sims.argmax(dim=1).tolist()
    nouns = [BUILTIN_NOUNS[j] for j in top]
    noun_sim = [float(sims[i, j]) for i, j in enumerate(top)]
    rel = (f_in @ f_src).tolist()
    pair = (f_in @ f_in.T).tolist()

    relevant = [i for i in range(n) if rel[i] >= args.min_relevance]
    n_subjects = same_subject_groups(relevant, nouns, pair, args.same_cos) if relevant else 0
    pcs = [pair[a][b] for a in relevant for b in relevant if a < b]
    diversity = 1.0 - sum(pcs) / len(pcs) if pcs else 0.0
    max_pair = max((pair[a][b] for a in range(n) for b in range(n) if a < b), default=None)

    rec.update({
        "source": str(src),
        "nouns": nouns,
        "noun_sim": [round(v, 3) for v in noun_sim],
        "rel": [round(v, 3) for v in rel],
        "low_relevance": [i for i in range(n) if rel[i] < args.min_relevance],
        "max_pair_cos": round(max_pair, 3) if max_pair is not None else None,
        "n_subjects": n_subjects,
        "diversity": round(diversity, 3),
        "score": round(n_subjects + diversity, 3),
        "subject_noun": (info.get("subject") or {}).get("noun"),
    })
    if not rec["decomposable"]:
        rec["reason"] = "not decomposable (groups.json)"
    elif n_subjects < args.min_subjects:
        rec["reason"] = (f"{n_subjects} distinct subject(s) among {len(relevant)} relevant input(s)"
                         if relevant else "no relevant input")
    else:
        rec["candidate"] = True
        rec["reason"] = f"{n_subjects} distinct subjects"
    return rec, (src_img, imgs)


# --------------------------------------------------------------------------------------
# Contact sheets
# --------------------------------------------------------------------------------------
def contact_sheets(entries, prefix, out_dir: Path, per_sheet: int, tile: int):
    """entries: list of (rec, (source, inputs)); one row per folder."""
    try:
        font = ImageFont.load_default(size=12)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    gap, title_h, cap_h = 6, 18, 16
    n_cols = 1 + max((len(imgs) for _, (_, imgs) in entries), default=1)
    paths = []
    for page in range(0, len(entries), per_sheet):
        chunk = entries[page:page + per_sheet]
        row_h = title_h + tile + cap_h + gap
        W = gap + n_cols * (tile + gap)
        canvas = Image.new("RGB", (W, gap + len(chunk) * row_h), "white")
        draw = ImageDraw.Draw(canvas)
        for r, (rec, (src, imgs)) in enumerate(chunk):
            y = gap + r * row_h
            color = (0, 120, 0) if rec["candidate"] else (90, 90, 90)
            draw.text((gap, y + 2), f"{rec['folder']}   score {rec['score']:.2f}   {rec['reason']}", fill=color, font=font)
            y += title_h
            canvas.paste(src.resize((tile, tile)), (gap, y))
            draw.text((gap + 2, y + tile + 2), "source", fill="black", font=font)
            for i, im in enumerate(imgs):
                x = gap + (i + 1) * (tile + gap)
                canvas.paste(im.resize((tile, tile)), (x, y))
                low = i in rec["low_relevance"]
                draw.text((x + 2, y + tile + 2), f"{i}: {rec['nouns'][i]}  rel {rec['rel'][i]:.2f}",
                          fill=(200, 0, 0) if low else "black", font=font)
        path = out_dir / f"{prefix}_{page // per_sheet:02d}.jpeg"
        canvas.save(path, quality=88)
        paths.append(path)
    return paths


def quantiles(values):
    """min / 10% / 50% / 90% / max, to calibrate the thresholds."""
    if not values:
        return None
    v = sorted(values)
    q = lambda f: round(v[min(len(v) - 1, int(f * (len(v) - 1) + 0.5))], 3)
    return {"min": q(0.0), "p10": q(0.1), "p50": q(0.5), "p90": q(0.9), "max": q(1.0)}


# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Screen decomposition folders for designated-target synthesis")
    p.add_argument("--decomp-root", type=Path, default=Path("data/no_main"),
                   help="Decomposition folders to screen (one level, folders with input_0.jpeg)")
    p.add_argument("--generate-source", type=Path, default=Path("generate_source"),
                   help="Reviewed folders; a candidate with a folder of the same name here is not copied")
    p.add_argument("--screen-dir", type=Path, default=None,
                   help="Where the candidates are copied (default: <generate-source>/screening)")
    p.add_argument("--source-dir", type=Path, default=Path("samples/sources"),
                   help="Fallback for source images whose groups.json path does not exist")
    p.add_argument("--report-dir", type=Path, default=None, help="Default: <screen-dir>/_report")
    p.add_argument("--min-subjects", type=int, default=2, help="Distinct subjects needed for a candidate")
    p.add_argument("--same-cos", type=float, default=0.8,
                   help="Inputs with image-level cosine >= this value count as the same subject")
    p.add_argument("--min-relevance", type=float, default=0.4,
                   help="Inputs with cos(input, source) below this value are not counted as subjects")
    p.add_argument("--dry-run", action="store_true", help="Write the table and sheets, do not copy")
    p.add_argument("--recopy", action="store_true", help=f"Ignore {RECORD} (copy deleted candidates again)")
    p.add_argument("--per-sheet", type=int, default=20, help="Folders per contact sheet")
    p.add_argument("--tile", type=int, default=160, help="Tile size of the contact sheets")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    folders = sorted(d for d in args.decomp_root.iterdir() if d.is_dir() and (d / "input_0.jpeg").exists())
    if not folders:
        raise ValueError(f"No decomposition folders (with input_0.jpeg) in {args.decomp_root}")
    if args.screen_dir is None:
        args.screen_dir = args.generate_source / "screening"
    if args.report_dir is None:
        args.report_dir = args.screen_dir / "_report"
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.screen_dir.mkdir(parents=True, exist_ok=True)

    def subdirs(d: Path):
        return {x.name for x in d.iterdir() if x.is_dir()} if d.exists() else set()

    # reviewed: directly in generate_source/ (the screening dir itself is not a decomposition folder)
    reviewed = subdirs(args.generate_source) - {args.screen_dir.name}
    pending = subdirs(args.screen_dir) - {args.report_dir.name}
    record_path = args.screen_dir / RECORD
    record = set(record_path.read_text().split()) if record_path.exists() and not args.recopy else set()
    print(f"{len(folders)} folder(s) in {args.decomp_root}; {len(reviewed)} in {args.generate_source}, "
          f"{len(pending)} waiting in {args.screen_dir}")

    clip = Clip(args.device)
    noun_bank = clip.texts([f"a photo of a {n.replace('-', ' ')}" for n in BUILTIN_NOUNS])

    rows, cand, rej = [], [], []
    for k, folder in enumerate(folders):
        rec, imgs = screen_folder(folder, clip, noun_bank, args)
        rec["in_generate_source"] = folder.name in reviewed
        rec["in_screening"] = folder.name in pending
        rec["copied"] = False
        if (rec["candidate"] and not args.dry_run and folder.name not in reviewed
                and folder.name not in pending and folder.name not in record):
            shutil.copytree(folder, args.screen_dir / folder.name)
            with open(record_path, "a") as f:
                f.write(folder.name + "\n")
            rec["copied"] = True
        rows.append(rec)
        if imgs is not None:
            (cand if rec["candidate"] else rej).append((rec, imgs))
        if (k + 1) % 20 == 0 or k + 1 == len(folders):
            print(f"  {k + 1} / {len(folders)}")

    cand.sort(key=lambda e: -e[0]["score"])
    rej.sort(key=lambda e: -e[0]["score"])
    sheets = (contact_sheets(cand, "candidates", args.report_dir, args.per_sheet, args.tile)
              + contact_sheets(rej, "rejected", args.report_dir, args.per_sheet, args.tile))

    fields = ["folder", "candidate", "score", "n_subjects", "diversity", "max_pair_cos", "n_inputs", "nouns", "rel",
              "noun_sim", "low_relevance", "subject_noun", "decomposable", "reason", "in_generate_source", "in_screening", "copied",
              "source"]
    with open(args.report_dir / "screen.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (-r["candidate"], -r["score"])):
            w.writerow({k: (" ".join(map(str, r[k])) if isinstance(r.get(k), list) else r.get(k)) for k in fields})

    chosen = [r for r in rows if r["in_generate_source"]]
    summary = {
        "decomp_root": str(args.decomp_root), "n_folders": len(rows),
        "n_candidates": sum(r["candidate"] for r in rows), "n_copied": sum(r["copied"] for r in rows),
        "dry_run": args.dry_run,
        "thresholds": {"min_subjects": args.min_subjects, "same_cos": args.same_cos, "min_relevance": args.min_relevance},
        "already_in_generate_source": {
            "n": len(chosen), "n_candidates": sum(r["candidate"] for r in chosen),
            "not_candidates": [{"folder": r["folder"], "reason": r["reason"]} for r in chosen if not r["candidate"]],
        },
        "distributions": {"rel": quantiles([v for r in rows for v in r.get("rel", [])]),
                          "max_pair_cos": quantiles([r["max_pair_cos"] for r in rows
                                                     if r.get("max_pair_cos") is not None])},
        "sheets": [str(s) for s in sheets],
    }
    with open(args.report_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\ncandidates: {summary['n_candidates']} / {len(rows)}"
          + ("" if args.dry_run else f"; copied {summary['n_copied']} to {args.screen_dir}"))
    if chosen:
        print(f"folders already in {args.generate_source} (earlier manual choice): "
              f"{summary['already_in_generate_source']['n_candidates']} / {len(chosen)} pass the screening")
        for r in summary["already_in_generate_source"]["not_candidates"]:
            print(f"  would be rejected: {r['folder']} ({r['reason']})")
    print(f"table: {args.report_dir / 'screen.csv'}; sheets: {len(sheets)} in {args.report_dir}")


if __name__ == "__main__":
    main()
