"""
score_sweep.py -- score the decompose_v7_2.py folders of a sweep and pick, per source image, the folder for
the no-main pool (level 1) and the folder for the main pool (level 2, replaces screen_sources.py).
Last modified: 2026-11-23

Reads every folder with a groups.json under --sweep-root (run_sweep.sh writes
<sweep-root>/topk64_c0.3_s1/<decompose_v7_2.py folder>/). decompose_v7_2.py is not changed. The source
image and the settings are read from groups.json, never from folder names.

What a decomposition is supposed to show is not known in advance (the SAE groups have no names and a source
can have any number of concepts). So the checks test what the inputs ACTUALLY show, with the concepts of
the source as the vocabulary:

  source inventory   once per source image, the VLM lists the source's visible concepts, each ONE thing (never a
                     combination like "boy flying a kite"), with a type (object / material / color / shape /
                     pattern / style / scene), e.g. boy (object), kite (object), smiling cloud (object),
                     meadow (scene) -> <sweep-root>/_sources/<stem>.json
  main-choice table  per folder, ONE multiple-choice question per input: "Which ONE of these is the main subject
                     of this image? 0 = none of these, 1 = boy, 2 = kite, ...". The probabilities of the answer
                     numbers are read directly (no text is generated), so every input gets a distribution over
                     the source concepts that sums to 1. (A yes/no "is X shown?" question is answered yes for
                     anything visible, also a supporting part like the platform under a figure, and saturates.)
                     -> <folder>/main_choice.json
  own concept        per input: the concept with the highest probability; own_p = that probability

Level 1 (no-main): every input is different, comes from the source and carries its own concept
  cheap checks on every folder (CLIP image encoder of the Kandinsky prior + DINOv2 -> <folder>/measure.json)
    unrelated        rel_clip = cos(input, source) < --min-rel
    source_copy      rel_clip > --max-rel
    duplicate        check 0: two inputs look the same (pair_clip >= --dup-clip or pair_dino >= --dup-dino);
                     the input less related to the source is marked. Any duplicate fails the whole folder.
  VLM checks (only on the folders that pass the cheap checks, best cheap score first, see "cascade")
    not_in_source    "none of these" is the most likely answer: broken image or content not in the source
    unclear          the own concept wins with own_p < --min-own: the input does not clearly show one concept
    shared_concept   two inputs have the same own concept (the one with the lower own_p is marked), or an input
                     gives the OBJECT that is another input's own concept a main-subject probability >= --leak
                     (it competes as the subject, e.g. a kite input in which the boy is as much the subject).
                     Scenes, colors, styles etc. are not checked for leaks: a meadow behind every input is fine.
  An input gets the first matching label of RULES, otherwise "unique".
  no_main_ok = decomposable, >= 2 inputs, all inputs unique.

Level 2 (main): at least --min-objects inputs each carry a different, prominent object
    objecthood       the own concept's inventory type is "object"
    prominence       own_p >= --prominent (the object clearly wins the main-subject question of its input)
    different kinds  VLM (text only) "Are <a> and <b> the same kind of object?" < 0.5 (bird / parrot -> same)
  n_objects = number of different kinds among the unique, prominent object inputs
  main_ok   = no_main_ok and n_objects >= --min-objects

Score and picks (per source image)
    score         sum over the unique inputs of own_p  +  diversity (1 - mean pair_clip between inputs)
                  Every input adds how clearly it shows its own concept (at most 1), so a 4th input only helps
                  if it is clear. Before the VLM own_p is unknown and counted as 1 (cheap score), so the cheap
                  score is an upper bound of the final score.
    no-main pick  best-scoring no_main_ok folder;  main pick  best-scoring main_ok folder
  cascade: the VLM checks the folders that pass the cheap checks in the order of their cheap score, and skips a
  folder once its cheap score (its best possible score) cannot beat the picks already found; at most
  --vlm-top folders per image (0 = no limit). Skipped folders are "not checked". With --no-vlm only the
  cheap checks run: no_main_ok then means "passes the cheap checks", and there are no main picks.
  overrides     <sweep-root>/overrides.csv (stem, no_main, main) replaces the automatic picks of an image: a
                folder name as in folders.csv "name" (e.g. topk64_c0.3_s0/14_topk64_s0_c0.3_a0.7_obj0.5),
                "none" for no pick, empty = automatic. Rerun the scorer (seconds) to apply it.

Results are kept per folder and add up over runs (score one image, a batch or everything; the picks of one
image never depend on other images):
    <folder>/measure.json, main_choice.json   cached measurements and VLM answers
    <folder>/score.json                        labels, flags and picks of the last scoring of this folder
    <sweep-root>/_review/                      a view of ALL scored folders, rebuilt from the score.json files:
      <stem>.jpeg                              every folder of one source image, picks first (redrawn for the
                                               images scored in this run)
      overview_00.jpeg, ...                    one row per pick of every image (images without a pick: their
                                               best folder, in red): the whole dataset at a glance
      folders.csv, inputs.csv, summary.json    tables of every folder that has a score.json
    --export-no-main data/no_main      the no-main picks AND the main picks (a main pick must also be in the
                                       no-main pool, so build_dataset.py finds byte-identical inputs)
    --export-main generate_source      the main picks (synthesize_v2_1.py reads them)
  Export keeps the target in sync with the picks of the images scored in this run: new picks are copied
  (unchanged, under their own name); a folder this script exported earlier that is no longer a pick of its
  image is removed. Folders it did not export (listed in <target>/_exported.json) are never touched.
  Every measurement and VLM answer is cached; changing a threshold and rerunning takes seconds (it may ask
  the VLM about a few more folders if the picks change). check_scoring.py compares the result with your labels.

Usage
    python score_sweep.py                                   # all folders under output/decompose/sweep
    python score_sweep.py --image samples/sources/14.jpeg   # only the folders of one source image
    python score_sweep.py --leak 0.7 --min-objects 3        # other thresholds
    python score_sweep.py --export-no-main data/no_main --export-main generate_source
"""

import argparse
import csv
import itertools
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from decompose_v7_2 import IMAGE_EXTS

PRIOR = "kandinsky-community/kandinsky-2-2-prior"
CONCEPT_TYPES = ("object", "material", "color", "shape", "pattern", "style", "scene")
PROMPT_VERSION = 3  # increase when a prompt below changes: cached VLM answers are then asked again
SCORE_VERSION = 4   # increase when the score.json format changes: older score.json files are left out of _review
INVENTORY_PROMPT = """List the distinct visual concepts in this image: the objects, and notable materials, colors,
patterns, shapes, styles or scenes. At most {k} concepts, the most salient first.
Each concept is ONE single thing with a short name (1-3 words): never a combination or an action
("boy" and "kite", not "boy flying a kite"). The names must be clearly different from each other.
Answer with JSON only, no other text:
{{"concepts": [{{"name": "...", "type": "object"}}]}}
where type is one of "object", "material", "color", "shape", "pattern", "style", "scene"."""
CHOICE_PROMPT = """Which ONE of these is the main subject of this image?
0. none of these (the image shows something else, or nothing recognizable)
{options}
Answer with the number only."""
SAME_KIND_QUESTION = 'Are "{a}" and "{b}" the same kind of object?'

# label rules for one input x (dict from label_inputs), checked in order; the first match wins
RULES = [
    ("unrelated",      lambda x, a: x["rel_clip"] < a.min_rel),
    ("source_copy",    lambda x, a: x["rel_clip"] > a.max_rel),
    ("duplicate",      lambda x, a: x["dup_of"] is not None),
    ("not_in_source",  lambda x, a: x["own"] == "none of these"),
    ("unclear",        lambda x, a: x["own_p"] is not None and x["own_p"] < a.min_own),
    ("shared_concept", lambda x, a: x["shares_with"] is not None),
]
GREEN, ORANGE, RED = (0, 130, 0), (215, 120, 0), (200, 0, 0)
LABEL_COLORS = {"unique": GREEN, "duplicate": ORANGE, "shared_concept": ORANGE}


def parse_args():
    p = argparse.ArgumentParser(description="Score decompose_v7_2.py sweep folders: no-main and main picks")
    p.add_argument("--sweep-root", type=Path, default=Path("output/decompose/sweep"))
    src = p.add_mutually_exclusive_group()
    src.add_argument("--image", help="Only the folders of this source image")
    src.add_argument("--image-dir", help="Only the folders of the source images in this folder")
    p.add_argument("--export-no-main", type=Path, default=None, help="Copy no-main and main picks here (data/no_main)")
    p.add_argument("--export-main", type=Path, default=None, help="Copy main picks here (generate_source)")
    p.add_argument("--no-vlm", action="store_true", help="Cheap checks only (no VLM, no main picks)")
    p.add_argument("--vlm-top", type=int, default=4, help="VLM-check at most this many folders per image (0 = all)")
    p.add_argument("--remeasure", action="store_true", help="Recompute measure.json of every folder")
    p.add_argument("--reask", action="store_true", help="Ask the VLM again (inventories and main-choice tables)")
    p.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--vlm-size", type=int, default=448, help="Images are resized to this size for the VLM")
    p.add_argument("--max-concepts", type=int, default=8, help="At most this many concepts in a source inventory (<= 9)")
    p.add_argument("--overview-rows", type=int, default=20, help="Rows per overview page")
    p.add_argument("--dino-model", default="facebook/dinov2-base")
    p.add_argument("--device", default="cuda")
    # thresholds
    p.add_argument("--min-rel", type=float, default=0.4, help="unrelated if rel_clip < this")
    p.add_argument("--max-rel", type=float, default=0.92, help="source_copy if rel_clip > this")
    p.add_argument("--dup-clip", type=float, default=0.8, help="duplicate pair if pair_clip >= this")
    p.add_argument("--dup-dino", type=float, default=0.75, help="duplicate pair if pair_dino >= this")
    p.add_argument("--min-own", type=float, default=0.4, help="unclear if the own concept wins with own_p < this")
    p.add_argument("--leak", type=float, default=0.3, help="shared_concept if another input's own OBJECT gets a "
                                                         "main-subject probability >= this in an input")
    p.add_argument("--prominent", type=float, default=0.5, help="an object counts for main if its own_p >= this")
    p.add_argument("--min-objects", type=int, default=2, help="main_ok needs this many different prominent objects")
    args = p.parse_args()
    args.max_concepts = min(args.max_concepts, 9)  # answers are the digits 0-9
    return args


# --------------------------------------------------------------------------------------
# cheap measurements (CLIP, DINOv2)
# --------------------------------------------------------------------------------------
class Encoders:
    """CLIP image encoder of the Kandinsky 2.2 prior (as in screen_sources.py) and DINOv2."""

    def __init__(self, dino_model, device):
        from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPVisionModelWithProjection
        dt = torch.float16 if device.startswith("cuda") else torch.float32
        self.device, self.dtype = device, dt
        self.clip_proc = CLIPImageProcessor.from_pretrained(PRIOR, subfolder="image_processor")
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(
            PRIOR, subfolder="image_encoder", torch_dtype=dt).to(device).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained(dino_model)
        self.dino_model = AutoModel.from_pretrained(dino_model, torch_dtype=dt).to(device).eval()
        self.sources = {}  # source path -> (clip, dino) features: every source is encoded once per run

    @torch.inference_mode()
    def encode(self, images):
        px = self.clip_proc(images, return_tensors="pt").pixel_values.to(self.device, self.dtype)
        c = F.normalize(self.clip_model(px).image_embeds.float(), dim=-1)
        px = self.dino_proc(images=images, return_tensors="pt").pixel_values.to(self.device, self.dtype)
        d = F.normalize(self.dino_model(pixel_values=px).pooler_output.float(), dim=-1)
        return c, d

    def source(self, path):
        if path not in self.sources:
            c, d = self.encode([Image.open(path).convert("RGB")])
            self.sources[path] = (c[0], d[0])
        return self.sources[path]


def list_inputs(folder: Path):
    """input_0.jpeg, input_1.jpeg, ... (consecutive)."""
    paths = []
    while (folder / f"input_{len(paths)}.jpeg").exists():
        paths.append(folder / f"input_{len(paths)}.jpeg")
    return paths


def measure(folder: Path, info: dict, enc: Encoders) -> dict:
    inputs = list_inputs(folder)
    source = Path(info.get("image", ""))
    m = {"n_inputs": len(inputs), "source": str(source) if source.is_file() else None}
    if len(inputs) < 2 or m["source"] is None:
        return m
    c, d = enc.encode([Image.open(p).convert("RGB") for p in inputs])
    sc, sd = enc.source(m["source"])
    r = lambda t: np.round(t.cpu().numpy(), 4).tolist()
    m.update({"rel_clip": r(c @ sc), "rel_dino": r(d @ sd), "pair_clip": r(c @ c.T), "pair_dino": r(d @ d.T)})
    return m


# --------------------------------------------------------------------------------------
# VLM: source inventory, yes-probabilities
# --------------------------------------------------------------------------------------
class VLM:
    def __init__(self, model_name, device, size):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.name, self.size = model_name, size
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.bfloat16).to(device).eval()
        tok = self.processor.tokenizer
        first = lambda words: sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})
        self.yes_ids, self.no_ids = first(["Yes", "yes"]), first(["No", "no"])
        self.digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]

    def load(self, path):
        return Image.open(path).convert("RGB").resize((self.size, self.size))

    def _inputs(self, image, text):
        content = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": text}]
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.model.device)

    @torch.inference_mode()
    def yes_prob(self, image, question) -> float:
        """P(yes) of the first answer token, from one forward pass (image may be None: text-only question)."""
        logits = self.model(**self._inputs(image, question + " Answer yes or no.")).logits[0, -1].float()
        yes, no = torch.logsumexp(logits[self.yes_ids], 0), torch.logsumexp(logits[self.no_ids], 0)
        return round(float(torch.sigmoid(yes - no)), 4)

    @torch.inference_mode()
    def choice_probs(self, image, prompt, n_options) -> list:
        """Probabilities of the answers 0 .. n_options-1, from the first answer token (one forward pass)."""
        logits = self.model(**self._inputs(image, prompt)).logits[0, -1].float()
        return [round(float(v), 4) for v in logits[self.digit_ids[:n_options]].softmax(0)]

    @torch.inference_mode()
    def generate(self, image, prompt, sample=False) -> str:
        inputs = self._inputs(image, prompt)
        extra = {"do_sample": True, "temperature": 0.3} if sample else {"do_sample": False}
        out = self.model.generate(**inputs, max_new_tokens=400, **extra)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def parse_inventory(raw: str, k: int):
    """[{"name", "type"}, ...] from the VLM answer, or None if it is not usable."""
    try:
        data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        out, seen = [], set()
        for c in data["concepts"]:
            name, kind = str(c["name"]).strip(), str(c.get("type", "")).lower().strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                out.append({"name": name, "type": kind if kind in CONCEPT_TYPES else "other"})
        return out[:k] or None
    except (ValueError, KeyError, TypeError):
        return None


def source_inventory(vlm: VLM, source: str, k: int) -> dict:
    image = vlm.load(source)
    result = {"model": vlm.name, "prompt_version": PROMPT_VERSION, "ok": False, "raw": []}
    for attempt in range(2):  # second attempt sampled
        raw = vlm.generate(image, INVENTORY_PROMPT.format(k=k), sample=attempt > 0)
        result["raw"].append(raw)
        concepts = parse_inventory(raw, k)
        if concepts:
            result.update(ok=True, concepts=concepts)
            break
    return result


def choice_table(vlm: VLM, inputs, concepts) -> dict:
    """Per input: probabilities of [none of these] + the inventory concepts being the main subject."""
    options = "\n".join(f"{i + 1}. {c['name']}" for i, c in enumerate(concepts))
    prompt = CHOICE_PROMPT.format(options=options)
    p = [vlm.choice_probs(vlm.load(path), prompt, len(concepts) + 1) for path in inputs]
    return {"model": vlm.name, "prompt_version": PROMPT_VERSION, "concepts": [c["name"] for c in concepts], "p": p}


# --------------------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------------------
def label_inputs(m: dict, args, choice=None, concepts=None) -> list:
    """Label every input; choice / concepts are None before the VLM step."""
    n = m["n_inputs"]
    others = lambda k: [j for j in range(n) if j != k]
    inputs = [{"index": k, "rel_clip": m["rel_clip"][k], "rel_dino": m["rel_dino"][k],
               "max_pair_clip": max(m["pair_clip"][k][j] for j in others(k)),
               "max_pair_dino": max(m["pair_dino"][k][j] for j in others(k)),
               "dup_of": None, "own": None, "own_type": None, "own_p": None, "none_p": None,
               "shares_with": None, "leak": None} for k in range(n)]
    less_related = lambda i, j: (i, j) if m["rel_clip"][i] < m["rel_clip"][j] else (j, i)

    # check 0: inputs that look the same
    for i, j in itertools.combinations(range(n), 2):
        if m["pair_clip"][i][j] >= args.dup_clip or m["pair_dino"][i][j] >= args.dup_dino:
            weaker, other = less_related(i, j)
            inputs[weaker]["dup_of"] = other

    if choice is not None:
        P = np.array(choice["p"])  # inputs x (1 + concepts); column 0 = "none of these"
        names = ["none of these"] + [c["name"] for c in concepts]
        types = [None] + [c["type"] for c in concepts]
        for x, row in zip(inputs, P):
            own = int(row.argmax())
            x.update(own=names[own], own_type=types[own], own_p=float(row[own]), none_p=float(row[0]), _own=own)
        real = [x for x in inputs if x["_own"] > 0]
        # two inputs with the same own concept: the one with the lower probability is marked
        for a, b in itertools.combinations(real, 2):
            if a["_own"] == b["_own"]:
                weaker, other = (a, b) if a["own_p"] < b["own_p"] else (b, a)
                if weaker["shares_with"] is None:
                    weaker["shares_with"] = other["index"]
        # an input in which the own OBJECT of another input competes as the main subject
        for a in real:
            if a["own_type"] != "object":
                continue
            for b in inputs:
                p_leak = float(P[b["index"], a["_own"]])
                if b is not a and b["_own"] != a["_own"] and p_leak >= args.leak and b["shares_with"] is None:
                    b["shares_with"], b["leak"] = a["index"], round(p_leak, 4)
        for x in inputs:
            x.pop("_own")

    for x in inputs:
        x["label"] = next((name for name, test in RULES if test(x, args)), "unique")
    return inputs


# --------------------------------------------------------------------------------------
# level 2: prominent objects of different kinds
# --------------------------------------------------------------------------------------
def count_objects(inputs, get_vlm, kinds: dict, args) -> int:
    """Number of different kinds among the unique, prominent object inputs (get_vlm() loads the VLM only when a
    same-kind answer is not cached)."""
    objects = [x["own"] for x in inputs
               if x["label"] == "unique" and x["own_type"] == "object" and x["own_p"] >= args.prominent]
    groups = []  # objects of the same kind (bird / parrot) form one group
    for name in objects:
        for g in groups:
            key = "|".join(sorted([name.lower(), g[0].lower()]))
            if key not in kinds:
                kinds[key] = get_vlm().yes_prob(None, SAME_KIND_QUESTION.format(a=name, b=g[0]))
            if kinds[key] >= 0.5:
                g.append(name)
                break
        else:
            groups.append([name])
    return len(groups)


# --------------------------------------------------------------------------------------
# folder records, picks
# --------------------------------------------------------------------------------------
def set_labels(rec: dict, inputs: list, m: dict):
    """inputs, score, no_main_ok and reason of a folder record from its labelled inputs."""
    bad = [f"{x['index']} {x['label']}" for x in inputs if x["label"] != "unique"]
    pairs = [m["pair_clip"][i][j] for i, j in itertools.combinations(range(m["n_inputs"]), 2)]
    clarity = sum(1.0 if x["own_p"] is None else x["own_p"] for x in inputs if x["label"] == "unique")
    rec.update(inputs=inputs, no_main_ok=not bad, score=round(clarity + 1 - float(np.mean(pairs)), 3),
               reason="not unique: " + ", ".join(bad) if bad else "all inputs unique")


def select(recs, overrides: dict) -> dict:
    """Per source image: no-main pick = best no_main_ok folder, main pick = best main_ok folder; then the
    hand overrides of overrides.csv (stem -> {"no_main": name / "none" / "", "main": ...})."""
    by_stem = defaultdict(list)
    for r in recs:
        by_stem[r["stem"]].append(r)
    for stem, rs in by_stem.items():
        rs.sort(key=lambda r: -(r["score"] if r["score"] is not None else -9))
        pick = {"no_main": next((r for r in rs if r["no_main_ok"]), None),
                "main": next((r for r in rs if r["main_ok"]), None)}
        by_hand = set()
        for kind, value in (overrides.get(stem) or {}).items():
            if value:
                pick[kind] = None if value.lower() == "none" else next((r for r in rs if r["name"] == value), None)
                by_hand.add(kind)
        for r in rs:
            r["no_main_pick"], r["main_pick"] = r is pick["no_main"], r is pick["main"]
            r["verdict"] = "selected" if r["no_main_pick"] or r["main_pick"] else "discarded"
            for kind in by_hand:
                if r is pick[kind]:
                    r["reason"] += f"; {kind.replace('_', '-')} pick set by hand"
            if r["no_main_ok"] and r["verdict"] == "discarded" and pick["no_main"] is not None:
                r["reason"] += f"; lower score than {(pick['main'] if r['main_ok'] and pick['main'] else pick['no_main'])['name']}"
        rs.sort(key=lambda r: r["verdict"] != "selected")  # picks first, then by score (stable sort)
    return by_stem


def read_overrides(path: Path) -> dict:
    """overrides.csv -> {stem: {"no_main": value, "main": value}}; value = folder name, "none" or "" (automatic)."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {row["stem"].strip(): {k: (row.get(k) or "").strip() for k in ("no_main", "main")}
                for row in csv.DictReader(f)}


def picks(r):
    return " + ".join(name for name, on in (("no-main", r["no_main_pick"]), ("main", r["main_pick"])) if on)


# --------------------------------------------------------------------------------------
# review images and tables
# --------------------------------------------------------------------------------------
def draw_sheet(recs, path: Path, tile=150):
    """One row per folder: verdict and reason, then source, recon and the inputs with their labels."""
    try:
        font = ImageFont.load_default(size=12)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    gap, line = 6, 15
    row_h = 2 * line + tile + 4 * line + 2 * gap
    n_cols = 2 + max(len(list_inputs(Path(r["path"]))) for r in recs)
    canvas = Image.new("RGB", (max(gap + n_cols * (tile + gap), 760), len(recs) * row_h), "white")
    draw = ImageDraw.Draw(canvas)

    def tile_at(x, y, img_path, captions):
        ok = img_path and Path(img_path).exists()
        img = Image.open(img_path).convert("RGB") if ok else Image.new("RGB", (8, 8), "gray")
        canvas.paste(img.resize((tile, tile)), (x, y))
        for i, (text, color) in enumerate(captions):
            draw.text((x + 2, y + tile + 2 + i * line), text[:24], fill=color, font=font)

    for row, r in enumerate(recs):
        y = row * row_h + gap
        score = "-" if r["score"] is None else f"{r['score']:.2f}"
        head = f"SELECTED [{picks(r)}]" if r["verdict"] == "selected" else "DISCARDED"
        draw.text((gap, y), f"{head}  score {score}  {r['name']}", font=font,
                  fill=GREEN if r["verdict"] == "selected" else RED)
        draw.text((gap, y + line), r["reason"], fill="black", font=font)
        y += 2 * line
        folder = Path(r["path"])
        tile_at(gap, y, r["source"], [("source", "black")])
        tile_at(gap + tile + gap, y, folder / "recon.jpeg", [("recon", "black")])
        for x in r["inputs"]:
            label = x["label"]
            if label == "duplicate":
                label += f" of {x['dup_of']}"
            elif label == "shared_concept":
                label += f" ({x['shares_with']}" + (f" p {x['leak']:.2f})" if x["leak"] is not None else ")")
            own = "-" if x["own"] is None else f"{x['own']} {x['own_p']:.2f}"
            kind = "" if x["own"] is None else f"{x['own_type'] or ''}  none {x['none_p']:.2f}"
            tile_at(gap + (2 + x["index"]) * (tile + gap), y, folder / f"input_{x['index']}.jpeg", [
                (f"{x['index']}: {own}", "black"), (label, LABEL_COLORS.get(x["label"], RED)), (kind, "black"),
                (f"rel {x['rel_clip']:.2f}/{x['rel_dino']:.2f}", "black")])
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=88)


def write_review(scored, all_recs, review: Path, args):
    """Sheets of the images scored in this run; overview pages, tables and summary of ALL scored folders."""
    review.mkdir(parents=True, exist_ok=True)
    scored_by_stem = defaultdict(list)
    for r in scored:
        scored_by_stem[r["stem"]].append(r)
    for stem, rs in scored_by_stem.items():
        draw_sheet(rs, review / f"{stem}.jpeg")

    by_stem = defaultdict(list)
    for r in all_recs:
        by_stem[r["stem"]].append(r)
    # overview: the picks of every image (an image without a pick: its best folder, drawn in red)
    rows = []
    for stem, rs in sorted(by_stem.items()):
        chosen = [r for r in rs if r["verdict"] == "selected"]
        rows += chosen or sorted(rs, key=lambda r: -(r["score"] if r["score"] is not None else -9))[:1]
    for old in review.glob("overview_*.jpeg"):
        old.unlink()
    for page in range(0, len(rows), args.overview_rows):
        draw_sheet(rows[page:page + args.overview_rows], review / f"overview_{page // args.overview_rows:02d}.jpeg")
    inventories = {}
    for stem in by_stem:
        inv = load_json(args.sweep_root / "_sources" / f"{stem}.json", lambda d: True)
        inventories[stem] = (inv or {}).get("concepts") or []
    cols = ["name", "stem", "n_inputs", "top_k", "clean_frac", "seed", "verdict", "no_main_pick", "main_pick", "score",
            "no_main_ok", "main_ok", "n_objects", "vlm", "reason"]
    with open(review / "folders.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["labels", "own_concepts", "inventory", "path"])
        for stem, rs in sorted(by_stem.items()):
            inventory = " | ".join(f"{c['name']} ({c['type']})" for c in inventories[stem])
            for r in rs:
                w.writerow([r.get(c, len(r["inputs"]) if c == "n_inputs" else None) for c in cols] + [" ".join(x["label"] for x in r["inputs"]),
                                                       " | ".join(str(x["own"]) for x in r["inputs"]), inventory, r["path"]])
    icols = ["index", "label", "own", "own_type", "own_p", "none_p", "shares_with", "leak", "dup_of",
             "rel_clip", "rel_dino", "max_pair_clip", "max_pair_dino"]
    with open(review / "inputs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "stem", "verdict", *icols])
        for r in all_recs:
            for x in r["inputs"]:
                w.writerow([r["name"], r["stem"], r["verdict"], *(x.get(c) for c in icols)])

    picked = [x for r in all_recs if r["verdict"] == "selected" for x in r["inputs"]]
    per_setting = defaultdict(Counter)
    for r in all_recs:
        key = f"topk{r['top_k']}_c{r['clean_frac']}_s{r['seed']}"
        per_setting[key]["folders"] += 1
        per_setting[key]["no_main_ok"] += bool(r["no_main_ok"])
        per_setting[key]["main_ok"] += bool(r["main_ok"])
        per_setting[key]["picked"] += r["verdict"] == "selected"
    by_n = {}
    for n in sorted({r.get("n_inputs", len(r["inputs"])) for r in all_recs}):
        rs = [r for r in all_recs if r.get("n_inputs", len(r["inputs"])) == n]
        cheap = [r for r in rs if r["vlm"] in ("ok", "not checked", "inventory failed")]
        checked = [r for r in rs if r["vlm"] == "ok"]
        labels = Counter(x["label"] for r in checked for x in r["inputs"])
        n_in = sum(labels.values())
        by_n[str(n)] = {
            "folders": len(rs), "pass_cheap": len(cheap), "vlm_checked": len(checked),
            "no_main_ok": sum(r["no_main_ok"] for r in checked), "main_ok": sum(r["main_ok"] for r in checked),
            "no_main_picks": sum(r["no_main_pick"] for r in rs), "main_picks": sum(r["main_pick"] for r in rs),
            # per input, among VLM-checked folders: the share of inputs with each label (compare across n)
            "input_label_rate": {k: round(v / n_in, 3) for k, v in labels.most_common()} if n_in else {},
        }
    summary = {
        "sweep_root": str(args.sweep_root),
        "by_n_inputs": by_n, "images": len(by_stem), "folders": len(all_recs),
        "no_main_picks": sum(r["no_main_pick"] for r in all_recs), "main_picks": sum(r["main_pick"] for r in all_recs),
        "images_without_no_main_pick": sorted(s for s, rs in by_stem.items() if not any(r["no_main_pick"] for r in rs)),
        "images_without_main_pick": sorted(s for s, rs in by_stem.items() if not any(r["main_pick"] for r in rs)),
        "vlm": dict(Counter(r["vlm"] for r in all_recs)),
        "input_labels_all_folders": dict(Counter(x["label"] for r in all_recs for x in r["inputs"])),
        "picked_own_concept_types": dict(Counter(x["own_type"] for x in picked if x["own_type"]).most_common()),
        "picked_objects": dict(Counter(x["own"] for x in picked if x["own_type"] == "object").most_common(40)),
        "per_setting": {k: dict(v) for k, v in sorted(per_setting.items())},
        "thresholds_last_run": {k: getattr(args, k) for k in (
            "min_rel", "max_rel", "dup_clip", "dup_dino", "min_own", "leak", "prominent", "min_objects", "vlm_top",
            "vlm_model", "dino_model")},
    }
    (review / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def export(recs, dst_root: Path, which: str):
    """Sync dst_root with the picks among recs (the images scored in this run).

    New picks are copied; folders this script exported earlier for these images that are no longer picks are
    removed. dst_root/_exported.json lists what this script exported; other folders are never touched."""
    dst_root.mkdir(parents=True, exist_ok=True)
    manifest_path = dst_root / "_exported.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    stems = {r["stem"] for r in recs}
    wanted = {Path(r["path"]).name: r for r in recs
              if r["main_pick"] or (which == "no_main" and r["no_main_pick"])}
    added = removed = kept = 0
    for name, stem in list(manifest.items()):
        if stem in stems and name not in wanted:
            shutil.rmtree(dst_root / name, ignore_errors=True)
            del manifest[name]
            removed += 1
    for name, r in wanted.items():
        if (dst_root / name).exists():
            kept += 1
        else:
            shutil.copytree(r["path"], dst_root / name)
            manifest[name] = r["stem"]
            added += 1
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"export {which} -> {dst_root}: {added} added, {removed} removed (no longer picks), {kept} already there")


# --------------------------------------------------------------------------------------
def find_folders(root: Path, stems=None):
    """(folder, groups.json) of every decomposition folder under root, except under _review / _logs / _sources."""
    out = []
    for g in sorted(root.rglob("groups.json")):
        if any(part.startswith("_") for part in g.relative_to(root).parts):
            continue
        info = json.loads(g.read_text())
        if stems is None or Path(info.get("image", "")).stem in stems:
            out.append((g.parent, info))
    return out


def load_json(path: Path, fresh):
    """The cached JSON at path if it exists and fresh(data) is true, else None."""
    if path.exists():
        data = json.loads(path.read_text())
        if fresh(data):
            return data
    return None


def main():
    args = parse_args()
    stems = None
    if args.image:
        stems = {Path(args.image).stem}
    elif args.image_dir:
        stems = {q.stem for q in Path(args.image_dir).iterdir() if q.suffix.lower() in IMAGE_EXTS}
    folders = find_folders(args.sweep_root, stems)
    if not folders:
        raise SystemExit(f"no decomposition folders (groups.json) under {args.sweep_root}")
    t0, counts = time.time(), Counter()
    enc = vlm = None
    vlm_fresh = lambda d: args.reask is False and d.get("model") == args.vlm_model and \
        d.get("prompt_version") == PROMPT_VERSION
    kinds_path = args.sweep_root / "_sources" / "same_kind.json"
    kinds = load_json(kinds_path, lambda d: d.get("model") == args.vlm_model and not args.reask) or {"model": args.vlm_model}

    def get_vlm():
        nonlocal vlm
        vlm = vlm or VLM(args.vlm_model, args.device, args.vlm_size)
        return vlm

    # ---------------- cheap checks on every folder ----------------
    recs, measures = [], {}
    for folder, info in tqdm(folders, desc="cheap checks"):
        rec = {"name": str(folder.relative_to(args.sweep_root)), "path": str(folder),
               "stem": Path(info.get("image", folder.name)).stem, "top_k": info.get("top_k"),
               "clean_frac": info.get("clean_frac"), "seed": info.get("seed"), "source": None, "inputs": [],
               "score": None, "no_main_ok": False, "main_ok": False, "n_objects": None, "vlm": "not asked",
               "version": SCORE_VERSION}
        recs.append(rec)
        m = None if args.remeasure else load_json(folder / "measure.json", lambda d: "pair_dino" in d or "rel_clip" not in d)
        if m is None or (m["source"] is None and Path(info.get("image", "")).is_file()):
            enc = enc or Encoders(args.dino_model, args.device)
            m = measure(folder, info, enc)
            (folder / "measure.json").write_text(json.dumps(m, indent=1))
            counts["measured"] += 1
        measures[rec["path"]] = m
        rec["source"], rec["n_inputs"] = m["source"], m["n_inputs"]
        if not info.get("decomposable", True):
            rec["reason"] = "not decomposable"
        elif m["n_inputs"] < 2:
            rec["reason"] = f"{m['n_inputs']} input(s)"
        elif m["source"] is None:
            rec["reason"] = "source image not found"
        else:
            set_labels(rec, label_inputs(m, args), m)
    if enc is not None:
        del enc
        torch.cuda.empty_cache()

    # ---------------- VLM checks: per image, best cheap score first, until both picks are found ----------------
    by_stem = defaultdict(list)
    for r in recs:
        by_stem[r["stem"]].append(r)
    for stem, rs in tqdm(sorted(by_stem.items()), desc="VLM checks", disable=args.no_vlm):
        candidates = sorted((r for r in rs if r["no_main_ok"]), key=lambda r: -r["score"])
        if args.no_vlm or not candidates:
            continue
        inv_path = args.sweep_root / "_sources" / f"{stem}.json"
        inv = load_json(inv_path, vlm_fresh)
        if inv is None:
            inv = source_inventory(get_vlm(), candidates[0]["source"], args.max_concepts)
            inv_path.parent.mkdir(parents=True, exist_ok=True)
            inv_path.write_text(json.dumps(inv, indent=1))
            counts["inventories asked"] += 1
        best = {"no_main": -1e9, "main": -1e9}  # best final scores found so far
        for i, r in enumerate(candidates):
            # the cheap score is the best score r can still reach: skip it if it cannot beat both picks
            if not inv["ok"] or (args.vlm_top and i >= args.vlm_top) or r["score"] <= min(best.values()):
                r.update(no_main_ok=False, vlm="not checked" if inv["ok"] else "inventory failed",
                         reason="not checked by the VLM (cannot beat the picks found)" if inv["ok"] else
                         "VLM could not list the source concepts")
                continue
            folder, m = Path(r["path"]), measures[r["path"]]
            choice_path = folder / "main_choice.json"
            names = [c["name"] for c in inv["concepts"]]
            choice = load_json(choice_path, lambda d: vlm_fresh(d) and d.get("concepts") == names)
            if choice is None:
                choice = choice_table(get_vlm(), list_inputs(folder), inv["concepts"])
                choice_path.write_text(json.dumps(choice, indent=1))
                counts["folders asked"] += 1
            set_labels(r, label_inputs(m, args, choice, inv["concepts"]), m)
            r["vlm"] = "ok"
            if r["no_main_ok"]:
                n_kinds = len(kinds)
                r["n_objects"] = count_objects(r["inputs"], get_vlm, kinds, args)
                if len(kinds) > n_kinds:
                    counts["same-kind questions asked"] += len(kinds) - n_kinds
                r["main_ok"] = r["n_objects"] >= args.min_objects
                r["reason"] += f", {r['n_objects']} different prominent object(s)"
            if r["no_main_ok"]:
                best["no_main"] = max(best["no_main"], r["score"])
            if r["main_ok"]:
                best["main"] = max(best["main"], r["score"])
    kinds_path.parent.mkdir(parents=True, exist_ok=True)
    kinds_path.write_text(json.dumps(kinds, indent=1))

    # ---------------- picks, review, export ----------------
    select(recs, read_overrides(args.sweep_root / "overrides.csv"))
    for r in recs:
        (Path(r["path"]) / "score.json").write_text(json.dumps(r, indent=1))
    # the review covers every folder scored so far; score.json files of older script versions are left out
    saved = [json.loads((f / "score.json").read_text()) for f, _ in find_folders(args.sweep_root)
             if (f / "score.json").exists()]
    all_recs = [r for r in saved if r.get("version") == SCORE_VERSION]
    outdated = sorted({r.get("stem", "?") for r in saved if r.get("version") != SCORE_VERSION})
    review = args.sweep_root / "_review"
    summary = write_review(recs, all_recs, review, args)
    if args.export_no_main:
        export(recs, args.export_no_main, "no_main")
    if args.export_main:
        export(recs, args.export_main, "main")

    n_images = len({r["stem"] for r in recs})
    print(f"scored {len(recs)} folder(s) of {n_images} image(s) in {time.time() - t0:.0f} s; new work: "
          + (", ".join(f"{k} {v}" for k, v in counts.items()) or "none (all cached)"))
    print(f"all scored folders: {summary['folders']} of {summary['images']} image(s); "
          f"no-main picks {summary['no_main_picks']}, main picks {summary['main_picks']}; VLM {summary['vlm']}")
    print("input labels: " + ", ".join(f"{k} {v}" for k, v in summary["input_labels_all_folders"].items()))
    for n, b in summary["by_n_inputs"].items():
        pct = lambda a, c: f"{a}/{c}" + (f" ({a / c:.0%})" if c else "")
        bad = round(1 - b["input_label_rate"].get("unique", 1.0), 3) if b["input_label_rate"] else "-"
        print(f"  {n} inputs: {b['folders']} folders, pass cheap {pct(b['pass_cheap'], b['folders'])}, "
              f"pass no-main {pct(b['no_main_ok'], b['vlm_checked'])} of VLM-checked, main {b['main_ok']}, "
              f"picks {b['no_main_picks']} / {b['main_picks']}; inputs failing the VLM checks {bad}")
    for key in ("images_without_no_main_pick", "images_without_main_pick"):
        if summary[key]:
            print(f"{key.replace('_', ' ')}: {len(summary[key])} ({', '.join(summary[key][:8])})")
    if outdated:
        print(f"[note] {len(outdated)} image(s) have score.json files from an older version of this script and are "
              f"not in the review: {', '.join(outdated[:10])}{' ...' if len(outdated) > 10 else ''}. "
              f"Score them again (python score_sweep.py, or --image) to include them.")
    print(f"review: {review}/<stem>.jpeg (per image), overview_*.jpeg (all picks), folders.csv, inputs.csv, "
          f"summary.json; change picks in {args.sweep_root / 'overrides.csv'}")


if __name__ == "__main__":
    main()