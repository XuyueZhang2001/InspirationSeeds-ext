# fetch_hf_samples.py
import json
import re
from pathlib import Path

from datasets import load_dataset

N = 20
OUT = Path("hf_samples")
(OUT / "sources").mkdir(parents=True, exist_ok=True)
(OUT / "paper_inputs").mkdir(parents=True, exist_ok=True)

# Streaming: only downloads the rows that are actually read, not the whole dataset
ds = load_dataset("kfirgold99/Inspiration-Seeds-Training", split="train", streaming=True)
ds = ds.shuffle(seed=0, buffer_size=300)

records, seen = [], set()
for row in ds:
    sample_id = row["sample_id"]
    uid = sample_id.split("__")[0]
    if uid in seen:            # the same source may appear with several topk/frac settings
        continue
    seen.add(uid)

    # parts = dict(p.split("_", 1) for p in sample_id.split("__")[1:])   # {"topk": "32", "frac": "0.7"}
    m_topk = re.search(r"topk_(\d+)", sample_id)
    m_frac = re.search(r"frac_([\d.]+)", sample_id)
    parts = {
        "topk": m_topk.group(1) if m_topk else "0",
        "frac": m_frac.group(1) if m_frac else "0",
    }
    src = row["output_image"].convert("RGB")
    grid = row["input_image"].convert("RGB")
    w, h = grid.size
    img_a = grid.crop((0, 0, w // 2, h // 2))          # top-left
    img_b = grid.crop((w // 2, h // 2, w, h))          # bottom-right

    src.save(OUT / "sources" / f"{uid}.jpeg")
    img_a.save(OUT / "paper_inputs" / f"{uid}_A.jpeg")
    img_b.save(OUT / "paper_inputs" / f"{uid}_B.jpeg")
    records.append({"uid": uid, "sample_id": sample_id,
                    "topk": int(parts["topk"]) or None, "frac": float(parts["frac"]) or None})

    if len(records) >= N:
        break

with open(OUT / "samples.json", "w") as f:
    json.dump(records, f, indent=2)
print(f"Saved {len(records)} samples to {OUT}/")