from datasets import load_dataset
from pathlib import Path

print("Loading dataset from Hugging Face...")
ds = load_dataset("kfirgold99/Inspiration-Seeds-Training", split="train")

out_dir = Path("./dataset_samples")
out_dir.mkdir(exist_ok=True)

print("Downloading and saving 20 images...")
for i in range(20):
    sample = ds[i]
    filepath = out_dir / f"{sample['sample_id']}.png"
    sample["input_image"].save(filepath)

print(f"Done! 20 images are saved in {out_dir}")