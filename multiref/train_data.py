"""Training data for multi-reference fine-tuning: cached latents, pool sampling and the pool schedule.

Data comes from build_dataset.py (train.jsonl / val.jsonl) and cache_features.py
(<cache-dir>/latents/*.pt, prompts.pt, config.json).

Pool schedule
    "0:1.0,0.3:1.0,0.4:0.65" = probability of drawing from the unspecified pool as a function
    of training progress (step / max_steps): 1.0 until 30 %, linear to 0.65 at 40 %, then 0.65.
    A single value ("0.5") is a constant probability.

Sampling inside a pool
    unspecified  uniform over decomposition folders
    designated   uniform over folders, then over the mains of the folder, then over the kept
                 targets of the main (every source image counts the same, however many mains
                 and targets it has)

Reference augmentation (2026-11-26)
    cache_features.py --ref-augs N stores N augmented versions of every input (horizontal flip
    and / or a random square crop, see there) as <cache-dir>/latents/<key>_aug<v>.pt. With
    aug_prob > 0, make_example replaces each reference independently with probability aug_prob
    by one of these versions (uniform over v). The target is never augmented, so copying the
    main reference pixel by pixel no longer matches the target. aug_on = "main" restricts this
    to the main reference (comparison only: then "augmented" correlates with "main").
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from .data import cache_key, cache_file, read_jsonl
from .prompts import arrange, prompt_key


# --------------------------------------------------------------------------------------
# Schedule
# --------------------------------------------------------------------------------------
class PoolSchedule:
    """Piecewise-linear probability of the unspecified pool over training progress in [0, 1]."""

    def __init__(self, spec: str):
        points = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                x, y = part.split(":")
                points.append((float(x), float(y)))
            else:
                points.append((0.0, float(part)))
        if not points:
            raise ValueError(f"Empty schedule {spec!r}")
        for _, y in points:
            if not 0.0 <= y <= 1.0:
                raise ValueError(f"Probabilities must be in [0, 1]: {spec!r}")
        self.points = sorted(points)
        self.spec = spec

    def __call__(self, progress: float) -> float:
        pts = self.points
        if progress <= pts[0][0]:
            return pts[0][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if progress <= x1:
                return y0 if x1 == x0 else y0 + (y1 - y0) * (progress - x0) / (x1 - x0)
        return pts[-1][1]


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------
def aug_cache_file(cache_dir, path, size: int, aug: int) -> Path:
    """Cached latent of the augmented version aug (0 .. N-1) of one input image."""
    return Path(cache_dir) / "latents" / f"{cache_key(path, size)}_aug{int(aug)}.pt"


class LatentStore:
    """Cached VAE latents ([16, h, w], bfloat16) and prompt embeddings, optionally kept in memory.

    ref_augs: number of augmented versions per input in the cache (0 = none; config.json "ref_augs").
    """

    def __init__(self, cache_dir, preload_paths=None):
        self.cache_dir = Path(cache_dir)
        with open(self.cache_dir / "config.json") as f:
            self.config = json.load(f)
        self.ref_size = self.config["ref_size"]
        self.target_size = self.config["target_size"]
        self.ref_augs = int(self.config.get("ref_augs", 0))
        self.prompts = torch.load(self.cache_dir / "prompts.pt", map_location="cpu")
        self._mem = {}
        if preload_paths:
            for item in preload_paths:
                path, size, aug = (tuple(item) + (None,))[:3]   # (path, size) or (path, size, aug)
                self._mem[(str(Path(path).resolve()), size, aug)] = self._load(path, size, aug)

    def _load(self, path, size, aug=None) -> torch.Tensor:
        if aug is None:
            f = cache_file(self.cache_dir, path, size)
        else:
            f = aug_cache_file(self.cache_dir, path, size, aug)
        if not f.exists():
            what = "" if aug is None else f" (augmented version {aug})"
            raise FileNotFoundError(f"No cached latent for {path} at {size} px{what} ({f}); run cache_features.py"
                                    + ("" if aug is None else " with --ref-augs"))
        return torch.load(f, map_location="cpu")

    def latent(self, path, size, aug=None) -> torch.Tensor:
        key = (str(Path(path).resolve()), size, aug)
        if key in self._mem:
            return self._mem[key]
        return self._load(path, size, aug)

    def ref(self, path, aug=None):
        """Reference latent; aug = None for the original, else the augmented version aug."""
        return self.latent(path, self.ref_size, aug)

    def target(self, path):
        return self.latent(path, self.target_size)

    def prompt(self, key):
        if key not in self.prompts:
            raise KeyError(f"Prompt template {key!r} not in the cache; rerun cache_features.py")
        return self.prompts[key]


def all_latent_paths(rows, store_ref_size, store_target_size, ref_augs: int = 0):
    """Every (path, size, aug) needed by the rows (for preloading); aug = None for the originals.

    ref_augs > 0 also lists the augmented versions 0 .. ref_augs-1 of every input.
    """
    out = set()
    for r in rows:
        for x in r["inputs"]:
            out.add((x, store_ref_size, None))
            out.update((x, store_ref_size, v) for v in range(ref_augs))
        out.update((x, store_target_size, None) for x in r["targets"])
    return sorted(out, key=lambda t: (str(t[0]), t[1], -1 if t[2] is None else t[2]))


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------
class PoolSampler:
    """Draws a (row, target path) pair from the unspecified or designated pool."""

    def __init__(self, rows, seed: int = 0):
        self.rng = random.Random(seed)
        self.unspec = defaultdict(list)                        # folder -> rows
        self.desig = defaultdict(lambda: defaultdict(list))    # folder -> main -> rows
        for r in rows:
            if r["pool"] == "unspecified":
                self.unspec[r["folder"]].append(r)
            else:
                self.desig[r["folder"]][r["main"]].append(r)
        self.unspec_folders = sorted(self.unspec)
        self.desig_folders = sorted(self.desig)
        if not self.unspec_folders and not self.desig_folders:
            raise ValueError("No training samples")

    def counts(self):
        return {"unspecified_folders": len(self.unspec_folders),
                "designated_folders": len(self.desig_folders),
                "designated_mains": sum(len(m) for m in self.desig.values())}

    def sample(self, p_unspec: float):
        use_unspec = self.rng.random() < p_unspec
        if use_unspec and not self.unspec_folders:
            use_unspec = False
        if not use_unspec and not self.desig_folders:
            use_unspec = True
        if use_unspec:
            row = self.rng.choice(self.unspec[self.rng.choice(self.unspec_folders)])
        else:
            mains = self.desig[self.rng.choice(self.desig_folders)]
            row = self.rng.choice(mains[self.rng.choice(sorted(mains))])
        return row, self.rng.choice(row["targets"])


def make_example(row, target_path, store: LatentStore, role_mode: str, rng: random.Random, shuffle: bool = True,
                 aug_prob: float = 0.0, aug_on: str = "all"):
    """One training example: reference latents in model order, target latent and prompt embeddings.

    aug_prob > 0: every reference (aug_on "all") or only the main reference (aug_on "main") is
    replaced with probability aug_prob by a random augmented version from the cache. With
    aug_prob = 0 no extra random numbers are drawn, so the example is the same as before.
    "augs" in the result: augmented version per reference in model order (None = original).
    """
    order, main_pos, _ = arrange(len(row["inputs"]), row["main"], role_mode, shuffle=shuffle, rng=rng)
    augs = [None] * len(order)
    if aug_prob > 0:
        if store.ref_augs <= 0:
            raise ValueError("aug_prob > 0 but the cache has no augmented references (cache_features.py --ref-augs)")
        for j, i in enumerate(order):
            if aug_on == "main" and (row.get("main") is None or i != row["main"]):
                continue
            if rng.random() < aug_prob:
                augs[j] = rng.randrange(store.ref_augs)
    refs = [store.ref(row["inputs"][i], a) for i, a in zip(order, augs)]
    key = prompt_key(main_pos, role_mode)
    return {"refs": refs, "target": store.target(target_path), "main_pos": main_pos, "prompt_key": key,
            "prompt": store.prompt(key), "pool": row["pool"], "id": row["id"], "order": order, "augs": augs}


def load_rows(data_dir, split: str):
    path = Path(data_dir) / f"{split}.jsonl"
    return read_jsonl(path) if path.exists() else []


def pick_overfit_rows(rows, n: int, seed: int = 0, folders=None):
    """Small fixed subset for the overfitting test.

    Takes whole designated folders (all their mains, so counterfactual pairs are included) with
    the most mains first, until about half of n; fills the rest with unspecified samples, first
    those of the same folders. Every designated row keeps only its first target, so the
    mapping inputs -> target is deterministic.

    folders: optional list of folder names (last path component, e.g. "104_topk64_nmax4").
    Then every designated row of these folders is taken (in the given order, without the n / 2
    limit), followed by the unspecified rows of these folders, and the rest is filled with
    random unspecified rows of other folders up to n. Used to repeat a test on the same folders
    after the dataset changed (rebuild, dropped targets), which would change the automatic choice.
    """
    rng = random.Random(seed)
    desig = defaultdict(list)
    unspec = {}
    for r in rows:
        if r["pool"] == "designated":
            desig[r["folder"]].append(r)
        else:
            unspec[r["folder"]] = r
    picked, chosen = [], []
    if folders:
        by_name = {Path(f).name: f for f in set(desig) | set(unspec)}
        missing = [name for name in folders if name not in by_name]
        if missing:
            raise ValueError(f"--overfit-folders not found in the data: {', '.join(missing)}")
        chosen = [by_name[name] for name in folders]
        for folder in chosen:
            for r in sorted(desig.get(folder, []), key=lambda r: r["main"]):
                picked.append(dict(r, targets=r["targets"][:1]))
        rest = [unspec[f] for f in chosen if f in unspec]
        others = [r for f, r in sorted(unspec.items()) if f not in chosen]
        rng.shuffle(others)
        return picked + (rest + others)[: max(0, n - len(picked))]
    folders = chosen
    for folder in sorted(desig, key=lambda f: (-len(desig[f]), f)):
        if len(picked) >= n // 2:
            break
        for r in sorted(desig[folder], key=lambda r: r["main"]):
            picked.append(dict(r, targets=r["targets"][:1]))
        folders.append(folder)
    rest = [unspec[f] for f in folders if f in unspec]
    others = [r for f, r in sorted(unspec.items()) if f not in folders]
    rng.shuffle(others)
    picked += (rest + others)[: max(0, n - len(picked))]
    return picked
