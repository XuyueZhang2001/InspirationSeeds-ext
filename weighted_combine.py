# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "diffusers",
#     "transformers",
#     "huggingface-hub",
#     "pillow",
# ]
# ///
import argparse
from pathlib import Path
from typing import List

import torch
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.utils import load_image
from transformers import CLIPVisionModelWithProjection

DEVICE = "cuda"

@torch.inference_mode()
def embed_image(item, prior):
    """Extract CLIP image embedding using Kandinsky prior."""
    return prior.interpolate([item], [1]).image_embeds[0]

@torch.inference_mode()
def generate_one(image_embeds, decoder, negative_emb, output_path):
    """Generate a single image from CLIP embeddings."""
    generator = torch.Generator("cuda")
    generator.manual_seed(42) # Use a fixed seed for reproducibility
    
    images = decoder(
        image_embeds=image_embeds.unsqueeze(0),
        negative_image_embeds=negative_emb,
        height=512,
        width=512,
        num_inference_steps=18,
        num_images_per_prompt=1,
        generator=generator,
    ).images
    
    images[0].save(output_path)
    print(f"  -> Saved target: {output_path.name}")

@torch.inference_mode()
def load_kandinsky():
    """Load Kandinsky prior and decoder models."""
    image_encoder = (
        CLIPVisionModelWithProjection.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", subfolder="image_encoder"
        )
        .half()
        .to(DEVICE)
    )
    unet = (
        UNet2DConditionModel.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", subfolder="unet")
        .half()
        .to(DEVICE)
    )
    prior = KandinskyV22PriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior", image_encoder=image_encoder, torch_dtype=torch.float16
    ).to(DEVICE)
    decoder = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder", unet=unet, torch_dtype=torch.float16
    ).to(DEVICE)
    zero_embed = prior.get_zero_embed()
    return prior, decoder, zero_embed

@torch.inference_mode()
def process_single_folder(
    folder_path: Path, 
    prior, 
    decoder, 
    negative_emb, 
    weights: List[float], 
    lam: float, 
    output_name: str
):
    """Process a single subfolder containing input_0.jpeg ... input_K.jpeg"""
    print(f"Processing: {folder_path.name}")
    
    # Normalize weights
    weight_sum = sum(weights)
    if weight_sum == 0:
        print("  [Error] Sum of weights is zero. Skipping.")
        return
    norm_weights = [w / weight_sum for w in weights]
    K = len(weights)

    # Load images and extract embeddings
    e_list = []
    for i in range(K):
        img_path = folder_path / f"input_{i}.jpeg"
        if not img_path.exists():
            print(f"  [Error] Missing {img_path.name}. Skipping folder.")
            return
        
        img = load_image(str(img_path))
        e_i = embed_image(img, prior)
        e_list.append(e_i.float())

    # Calculate mean and directions
    e_stack = torch.stack(e_list)
    e_mean = e_stack.mean(dim=0)
    
    # d_i = e_i - e_mean
    d_list = [e_i - e_mean for e_i in e_list]

    # e_target = e_mean + lambda * sum(w_i * d_i)
    weighted_d_sum = torch.zeros_like(e_mean)
    for w, d in zip(norm_weights, d_list):
        weighted_d_sum += w * d
        
    e_target = e_mean + lam * weighted_d_sum

    # Generate and save
    output_path = folder_path / output_name
    generate_one(e_target.to(torch.float16), decoder, negative_emb, output_path)


if __name__ == "__main__":
    def parse_float_list(value: str) -> list[float]:
        return [float(x.strip()) for x in value.split(",")]

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Batch combine decomposed input images using specified weights."
        )
        parser.add_argument(
            "--root-dir",
            type=str,
            required=True,
            help="Root directory to search for subfolders containing input_0.jpeg.",
        )
        parser.add_argument(
            "--weights",
            type=parse_float_list,
            required=True,
            help="Comma-separated weights for each image, e.g., '0.6,0.3,0.1'",
        )
        parser.add_argument(
            "--lambda-val",
            type=float,
            default=1.0,
            help="Scaling factor for the weighted direction. Default is 1.0.",
        )
        parser.add_argument(
            "--output-name",
            type=str,
            default="target_weighted.jpeg",
            help="Name of the output file. Default is target_weighted.jpeg.",
        )
        return parser.parse_args()

    args = parse_args()
    
    root_path = Path(args.root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory not found: {root_path}")

    # Automatically find all subfolders that contain 'input_0.jpeg'
    print(f"Searching for valid folders in {root_path}...")
    valid_folders = [p.parent for p in root_path.rglob("input_0.jpeg")]
    # Filter out folders that already have the target image generated (optional)
    valid_folders = [p for p in valid_folders if not (p / args.output_name).exists()]
    
    if not valid_folders:
        print("No valid folders found (or all targets already generated). Exiting.")
        exit()

    print(f"Found {len(valid_folders)} folders to process.")

    # Load models once to save time
    prior, decoder, negative_emb = load_kandinsky()

    # Loop through all folders
    for idx, folder in enumerate(valid_folders):
        print(f"[{idx+1}/{len(valid_folders)}] ", end="")
        process_single_folder(
            folder_path=folder,
            prior=prior,
            decoder=decoder,
            negative_emb=negative_emb,
            weights=args.weights,
            lam=args.lambda_val,
            output_name=args.output_name,
        )

    print("All done!")