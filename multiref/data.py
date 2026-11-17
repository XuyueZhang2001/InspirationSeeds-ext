"""Manifest and cache helpers shared by the data scripts and the training script (no torch).

Paths are stored as given on the command line (usually relative to the repository root);
all scripts are meant to be run from the repository root.
"""

import hashlib
import json
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def stem_from_folder(folder) -> str:
    """Source image stem of a decomposition folder ('<stem>_topk64_nmax4' -> '<stem>')."""
    return Path(folder).name.split("_topk")[0]


def list_inputs(folder):
    """input_0.jpeg, input_1.jpeg, ... of a decomposition folder (consecutive indices)."""
    folder = Path(folder)
    paths, i = [], 0
    while (folder / f"input_{i}.jpeg").exists():
        paths.append(folder / f"input_{i}.jpeg")
        i += 1
    return paths


def cache_key(path, size: int) -> str:
    """Key of the cached VAE latent of one image at one resolution."""
    return hashlib.sha1(f"{Path(path).resolve()}|{int(size)}".encode()).hexdigest()[:20]


def cache_file(cache_dir, path, size: int) -> Path:
    return Path(cache_dir) / "latents" / f"{cache_key(path, size)}.pt"
