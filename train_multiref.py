"""
train_multiref.py -- LoRA (+ main-image role embedding) fine-tuning of FLUX.1 Kontext dev with 2-4 references.
Last modified: 2026-11-26 (--ref-aug-prob / --ref-aug-on: reference augmentation from the cache)
               2026-11-25 (several GPUs: data parallel with torchrun or --gpus; gradients of the trainable
               parameters averaged over the GPUs after every optimizer step)
               2026-11-22 (--overfit-folders: overfitting test on named folders; default --main-t 2 -> 3)
               2026-11-20 (default --role embed -> tid)
               2026-11-19 (role mode "tid": main image marked by position id t = --main-t)

Data: manifests from build_dataset.py (train.jsonl, val.jsonl) and the cache from
cache_features.py (VAE latents + prompt embeddings). Only the transformer (and, for the sample
images, the VAE) is loaded; no text encoder is needed.

One training step (batch 1, --grad-accum micro steps per optimizer step), following the
diffusers example train_dreambooth_lora_flux_kontext.py where possible:
    1. pool:   unspecified with probability --unspec-schedule(progress), else designated
    2. sample: unspecified -> uniform folder; designated -> uniform folder, main, kept target
    3. order:  references shuffled (multiref.prompts.arrange); prompt template of the main
               position; cached latents (references [16, 64, 64], target [16, 128, 128])
    4. noise:  sigma from the scheduler's training table (--weighting-scheme: uniform or
               logit_normal), x_t = (1 - sigma) x_0 + sigma * noise, packed target tokens (t = 0).
               Designated samples can use their own density (--designated-sigma, see below)
    5. refs:   reference tokens and ids (multiref.conditioning.build_reference_tokens) appended
               after the target tokens; the main reference gets t = --main-t instead of 1
               (--role tid) or the E_main hook (--role embed)
    6. loss:   transformer(timestep = sigma, guidance = --guidance-scale); MSE between the
               output of the target tokens and (noise - x_0)

Role modes (--role; how the main image is designated, see multiref/prompts.py)
    tid    (default) main image keeps its position, its tokens get position id t = --main-t
           (3; 2 before 2026-11-22) instead of 1; without a main image all references have
           t = 1 (canvas layout only). The base model does not react to the marker, so the LoRA
           learns it from scratch, but there is no position shortcut. t = 3 gave an about 3x
           larger designation gap than t = 2 in the overfitting tests
    embed  learnable E_main added to the main image's tokens; order unchanged (E_main stayed
           too small to be used in the first overfitting test)
    slot   main image moved to the first position (learns "first input = main image" even
           without a designation: check with eval_multiref.py)
    index  main image named in the prompt ("image k")

Sigma sampling of the designated samples
    Which input is the main image decides the composition, and the composition is decided at
    high noise; at low sigma the target is mostly visible in x_t and the main image hardly
    matters. --designated-sigma logit_normal therefore draws the sigma of designated samples
    directly as sigma = sigmoid(N(--designated-sigma-mean, --designated-sigma-std)), clamped to
    the range of the sigma table (mean > 0 -> more HIGH-noise samples; mean 1, std 1: median
    sigma 0.73). Unspecified samples keep --weighting-scheme. The loss weighting is unchanged.
    Note: --weighting-scheme logit_normal indexes the DESCENDING sigma table with u, so there a
    positive --logit-mean favours LOW sigma (as in the diffusers example).

Reference augmentation (--ref-aug-prob, default 0 = off)
    The targets of the designated pool are synthesized from the main input (img2img), so they share
    its layout pixel by pixel; a model can lower the loss by redrawing the marked reference, which
    the copy-rate evaluation of runs/v2_auto showed (copy 0.89 at step 12000). With
    --ref-aug-prob p every reference of a TRAINING sample is replaced independently with
    probability p by one of the augmented versions from the cache (horizontal flip / random square
    crop, made by cache_features.py --ref-augs N); the target is unchanged. Copying the main
    reference then no longer matches the target, so the model has to carry over what the main
    image shows, not where its pixels are. --ref-aug-on main augments only the main reference
    (for comparison; "augmented" then correlates with "main", which does not hold at inference).
    Validation loss, designation gap and sample images always use the original references, so
    runs with and without augmentation are directly comparable. Logged: ref_aug_share (share of
    augmented references in the training samples).

Trainable parameters: LoRA on the attention / feed-forward layers INSIDE the transformer
blocks (the final output projection, also named proj_out, is excluded) and E_main.
LoRA parameters are kept in float32, the forward pass runs in bfloat16 autocast.

Validation
    * every --val-every steps: loss on a fixed set of validation samples (fixed order, target,
      noise and sigmas 0.25 / 0.5 / 0.75), separately for the two pools
    * every --gap-every steps (and at step 0): designation gap on fixed designated samples
      (whole folders with the most mains first, at most --gap-samples). Per sample (main k,
      target_k) and per sigma in --gap-sigmas, with the same noise and target:
          L_right    main = k          L_wrong_j  main = j (every other input)
          L_none     no main (base prompt, no E_main)
      logged as  gap      = mean_j L_wrong_j - L_right   (> 0: the model uses the designation)
                 gap_acc  = share with L_right < min_j L_wrong_j   (chance: 1 / n inputs)
                 gap_none = L_none - L_right
      overall and per sigma (gap_s0.9, gap_acc_s0.9, ...). A wrong main is designated the same
      way as a right one: E_main on input j (embed), input j moved first (slot), "image j" in
      the prompt (index), t = --main-t on input j (tid). With --role embed the gap is exactly 0 at step 0 (E_main = 0 and the
      prompt does not depend on the main), which checks the metric itself.
    * every --image-every steps (and at step 0): sample images for a few validation folders,
      preferably with >= 2 designated mains: no main, then each main, next to the targets
      -> <output-dir>/samples/step_XXXXXX.jpeg

Training log (log.jsonl, every --log-every steps): loss per pool, grad_norm, p_unspec, lr,
    sec_per_step, designated_share, mean sigma per pool (sigma_designated, sigma_unspecified)
    and, with --role embed, e_main_norm, ref_token_norm (mean per-token norm of the reference
    tokens) and e_main_ratio = e_main_norm / ref_token_norm.

Checkpoints (every --save-every steps and at the end): <output-dir>/checkpoint-XXXXXX/
    lora/                      diffusers LoRA format (generate_multi.py --lora)
    role_embedding.safetensors E_main (generate_multi.py --role-embedding)
    train_state.pt             optimizer, scheduler, step and RNG states (--resume)

Overfitting test
    --overfit N trains on a fixed subset of about N training samples (whole designated folders
    with their counterfactual mains, plus unspecified samples), one target per main, and uses
    the same subset for validation. --overfit-folders names the folders explicitly (from the
    train or val split), so that a test can be repeated on the same folders after the dataset
    changed. The flow-matching loss drops only slowly even when samples
    are memorized; judge by the sample images (they should approach the targets) and by the
    designation gap (it should grow, mainly at high sigma).

Usage
-----
    # overfitting test (default role tid)
    python train_multiref.py --data-dir data/multiref --output-dir runs/overfit \\
        --overfit 8 --max-steps 1500 --warmup-steps 50 --grad-accum 1 \\
        --val-every 100 --image-every 250 --save-every 500

    # overfitting test, main image moved first, more high-noise designated samples
    python train_multiref.py --data-dir data/multiref --output-dir runs/overfit_slot_hs \\
        --overfit 8 --max-steps 1500 --warmup-steps 50 --grad-accum 1 --role slot \\
        --designated-sigma logit_normal --val-every 100 --image-every 250 --save-every 500

    # overfitting test, main image marked by position id t = 3 (setting of runs/overfit_tid_t3)
    python train_multiref.py --data-dir data/multiref --output-dir runs/overfit_tid_t3 \\
        --role tid --main-t 3 \\
        --overfit 8 --max-steps 5000 --warmup-steps 50 --grad-accum 1 --lr 3e-4 \\
        --unspec-schedule 0.3 --designated-sigma logit_normal \\
        --val-every 250 --image-every 500 --save-every 1000

    # full run
    python train_multiref.py --data-dir data/multiref --output-dir runs/v1 --max-steps 10000

    # with reference augmentation (first: cache_features.py --ref-augs 4)
    python train_multiref.py --data-dir data/multiref --output-dir runs/v3_aug --ref-aug-prob 0.5

Several GPUs (data parallel)
    python train_multiref.py --gpus 0,1,2,3 ...            # restarts itself with torchrun on GPUs 0-3
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 train_multiref.py ...
    Every GPU holds the full model and draws its own samples (own sampler, order, sigma and noise
    seeds); after the --grad-accum micro steps of an optimizer step the gradients of the trainable
    parameters (LoRA, E_main) are averaged over the GPUs, so every GPU makes the same update and the
    parameters stay identical. Samples per optimizer step = --grad-accum x number of GPUs: with
    4 GPUs, --grad-accum 1 gives the 4 samples per step of a single-GPU run with --grad-accum 4,
    about 4 times faster. Validation, designation gap, sample images, checkpoints and the log are
    made by the first GPU only; the others wait. Resuming needs the same number of GPUs to restore
    the per-GPU random states exactly (otherwise they are reseeded from the step).
"""

import argparse
import builtins
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxKontextPipeline, FluxTransformer2DModel
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image, ImageDraw, ImageFont

from multiref.conditioning import (
    LAYOUTS,
    MAIN_T,
    MainRoleEmbedding,
    MultiRefKontextPipeline,
    build_reference_tokens,
    marker_pos,
    pack_latents,
)
from multiref.prompts import ROLE_MODES, arrange, prompt_key
from multiref.train_data import (
    LatentStore,
    PoolSampler,
    PoolSchedule,
    all_latent_paths,
    load_rows,
    make_example,
    pick_overfit_rows,
)

DEVICE = "cuda"  # with several GPUs: the GPU of this process (torch.cuda.set_device in setup_distributed)
DTYPE = torch.bfloat16
RANK, WORLD = 0, 1  # set by setup_distributed
DEFAULT_SCHEDULE = "0:1.0,0.3:1.0,0.4:0.65"
OVERFIT_SCHEDULE = "0.5"
# LoRA layers (suffixes of module names inside transformer_blocks / single_transformer_blocks)
LORA_SUFFIXES = (
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",                        # image stream attention
    "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",    # text stream attention
    "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",      # double-block feed-forward
    "proj_mlp", "proj_out",                                                        # single-block MLP / output
)


# --------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Multi-reference LoRA fine-tuning of FLUX.1 Kontext")
    g = p.add_argument_group("data")
    g.add_argument("--data-dir", type=Path, default=Path("data/multiref"), help="train.jsonl / val.jsonl")
    g.add_argument("--cache-dir", type=Path, default=None, help="Default: <data-dir>/cache")
    g.add_argument("--output-dir", type=Path, default=Path("runs/multiref"))
    g.add_argument("--unspec-schedule", default=None,
                   help=f"Probability of the unspecified pool over training progress (default {DEFAULT_SCHEDULE}; "
                        f"with --overfit {OVERFIT_SCHEDULE})")
    g.add_argument("--overfit", type=int, default=0, help="Overfitting test on about N fixed training samples")
    g.add_argument("--overfit-folders", default=None,
                   help="Comma-separated folder names for --overfit (all their mains and unspecified samples, "
                        "train and val split), filled up with random unspecified samples to N")
    g.add_argument("--no-shuffle", action="store_true", help="Keep the input order fixed (debugging)")
    g.add_argument("--preload", action="store_true", help="Keep all cached latents in memory")
    g.add_argument("--ref-aug-prob", type=float, default=0.0,
                   help="Probability that a reference of a training sample is replaced by an augmented "
                        "version from the cache (cache_features.py --ref-augs; 0 = off)")
    g.add_argument("--ref-aug-on", choices=["all", "main"], default="all",
                   help="Augment every reference (default) or only the main reference")

    g = p.add_argument_group("model")
    g.add_argument("--base-model", default="black-forest-labs/FLUX.1-Kontext-dev")
    g.add_argument("--role", choices=ROLE_MODES, default="tid", help="How the main image is designated")
    g.add_argument("--layout", choices=LAYOUTS, default="canvas", help="Position ids of the references")
    g.add_argument("--main-t", type=float, default=MAIN_T,
                   help="Position id t of the main reference with --role tid (the other references have t = 1)")
    g.add_argument("--lora-rank", type=int, default=32)
    g.add_argument("--lora-alpha", type=int, default=None, help="Default: equal to the rank")
    g.add_argument("--lora-layers", default=",".join(LORA_SUFFIXES),
                   help="Comma-separated module-name suffixes inside the transformer blocks")
    g.add_argument("--init-lora", default=None, help="Initialize the LoRA from these weights (same rank and layers)")
    g.add_argument("--guidance-scale", type=float, default=3.5, help="Guidance embedding value (use it at inference)")

    g = p.add_argument_group("optimization")
    g.add_argument("--max-steps", type=int, default=10000, help="Optimizer steps")
    g.add_argument("--grad-accum", type=int, default=4, help="Micro steps (samples) per optimizer step")
    g.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate")
    g.add_argument("--role-lr", type=float, default=1e-3, help="E_main learning rate")
    g.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay of the LoRA parameters")
    g.add_argument("--warmup-steps", type=int, default=500, help="Linear warmup, then constant")
    g.add_argument("--max-grad-norm", type=float, default=1.0)
    g.add_argument("--weighting-scheme", choices=["none", "logit_normal", "mode", "cosmap", "sigma_sqrt"],
                   default="none", help="Timestep density / loss weighting (none = uniform sigma)")
    g.add_argument("--logit-mean", type=float, default=0.0,
                   help="logit_normal mean of u; u indexes the DESCENDING sigma table, so > 0 favours LOW sigma")
    g.add_argument("--logit-std", type=float, default=1.0)
    g.add_argument("--mode-scale", type=float, default=1.29)
    g.add_argument("--designated-sigma", choices=["same", "logit_normal"], default="same",
                   help="Sigma density of designated samples: same as unspecified, or "
                        "sigma = sigmoid(N(mean, std)) (mean > 0 favours HIGH sigma)")
    g.add_argument("--designated-sigma-mean", type=float, default=1.0)
    g.add_argument("--designated-sigma-std", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("validation and checkpoints")
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--val-every", type=int, default=500, help="Validation loss interval (0 = off)")
    g.add_argument("--val-samples", type=int, default=16, help="Validation samples used for the loss")
    g.add_argument("--gap-every", type=int, default=None,
                   help="Designation-gap interval (default: --val-every; 0 = off)")
    g.add_argument("--gap-samples", type=int, default=8, help="Designated samples used for the gap")
    g.add_argument("--gap-sigmas", default="0.9,0.75,0.5,0.25", help="Sigmas of the designation gap")
    g.add_argument("--gap-noises", type=int, default=1, help="Noise draws per sigma for the gap")
    g.add_argument("--image-every", type=int, default=1000, help="Sample-image interval (0 = off)")
    g.add_argument("--image-folders", type=int, default=3, help="Validation folders shown in the sample images")
    g.add_argument("--image-steps", type=int, default=28, help="Denoising steps of the sample images")
    g.add_argument("--no-images-at-start", action="store_true", help="Skip the sample images at step 0")
    g.add_argument("--save-every", type=int, default=1000)
    g.add_argument("--resume", type=Path, default=None, help="checkpoint-XXXXXX folder to resume from")
    g.add_argument("--tensorboard", action="store_true", help="Also log to TensorBoard (<output-dir>/tb)")

    g = p.add_argument_group("GPUs")
    g.add_argument("--gpus", default=None,
                   help="GPUs to train on: '3' = GPU 3; '0,1,2,3' or '0-3' = data parallel on these GPUs "
                        "(the script restarts itself with torchrun). Default: one process on the default GPU, "
                        "or as started by torchrun")
    args = p.parse_args()
    if args.gap_every is None:
        args.gap_every = args.val_every
    return args


# --------------------------------------------------------------------------------------
# Several GPUs
# --------------------------------------------------------------------------------------
def parse_gpus(text: str):
    """'3' -> [3]; '0,1' -> [0, 1]; '0-3' -> [0, 1, 2, 3]; '0,2-3' -> [0, 2, 3]."""
    gpus = []
    for part in (x.strip() for x in text.split(",")):
        if "-" in part:
            a, b = part.split("-")
            gpus.extend(range(int(a), int(b) + 1))
        elif part:
            gpus.append(int(part))
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError(f"--gpus {text!r}: give distinct GPU indices, e.g. 3, 0,1 or 0-3")
    return gpus


def strip_option(argv, name):
    """Remove an option and its value from an argument list."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a == name:
            skip = True
        elif not a.startswith(name + "="):
            out.append(a)
    return out


def launch_with_gpus(args):
    """Handle --gpus before anything touches CUDA. Returns True if this process should continue."""
    if args.gpus is None or "LOCAL_RANK" in os.environ:
        return True
    gpus = parse_gpus(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    if len(gpus) == 1:
        return True  # one GPU: this process
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(gpus)}",
           os.path.abspath(__file__)] + strip_option(sys.argv[1:], "--gpus")
    print(f"data parallel on GPUs {gpus}: {' '.join(cmd)}")
    sys.exit(subprocess.run(cmd, env=os.environ.copy()).returncode)


def setup_distributed():
    """Initialize torch.distributed when started by torchrun; only the first process prints."""
    global RANK, WORLD
    if "LOCAL_RANK" not in os.environ:
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)  # DEVICE = "cuda" now means this GPU
    # long timeout: the other processes wait while the first one validates or draws sample images
    dist.init_process_group("nccl", timeout=timedelta(hours=2), device_id=torch.device("cuda", local_rank))
    RANK, WORLD = dist.get_rank(), dist.get_world_size()
    if RANK != 0:
        builtins.print = lambda *a, **k: None


def barrier():
    if WORLD > 1:
        dist.barrier()


@torch.no_grad()
def average_gradients(params):
    """Average the gradients of the trainable parameters over the GPUs (one all-reduce per dtype)."""
    if WORLD == 1:
        return
    by_dtype = defaultdict(list)
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        by_dtype[p.grad.dtype].append(p)
    for group in by_dtype.values():
        flat = torch.cat([p.grad.reshape(-1) for p in group])
        dist.all_reduce(flat)
        flat /= WORLD
        offset = 0
        for p in group:
            n = p.grad.numel()
            p.grad.copy_(flat[offset:offset + n].view_as(p.grad))
            offset += n


@torch.no_grad()
def broadcast_parameters(params):
    """Copy the parameters of the first GPU to the others (same start, guard against drift)."""
    if WORLD == 1:
        return
    for p in params:
        dist.broadcast(p.data, src=0)


def gather_window(window):
    """Merge the logging windows of all GPUs (every GPU must call this)."""
    if WORLD == 1:
        return window
    parts = [None] * WORLD
    dist.all_gather_object(parts, dict(window))
    merged = defaultdict(list)
    for part in parts:
        for k, v in part.items():
            merged[k].extend(v)
    return merged


def rng_state(sampler, data_rng, sigma_gen):
    return {"sampler_rng": sampler.rng.getstate(), "data_rng": data_rng.getstate(),
            "sigma_gen": sigma_gen.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state()}


def gather_rng_states(sampler, data_rng, sigma_gen):
    """Random states of every GPU, in rank order (every GPU must call this)."""
    state = rng_state(sampler, data_rng, sigma_gen)
    if WORLD == 1:
        return [state]
    parts = [None] * WORLD
    dist.all_gather_object(parts, state)
    return parts


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def lora_target_names(transformer, suffixes):
    """Full names of the Linear layers inside the transformer blocks that end with one of the suffixes."""
    names = []
    for name, module in transformer.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not (name.startswith("transformer_blocks.") or name.startswith("single_transformer_blocks.")):
            continue
        if any(name == s or name.endswith("." + s) for s in suffixes):
            names.append(name)
    if not names:
        raise ValueError(f"No LoRA target layers found for suffixes {suffixes}")
    return names


def build_transformer(args):
    transformer = FluxTransformer2DModel.from_pretrained(args.base_model, subfolder="transformer", torch_dtype=DTYPE)
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    targets = lora_target_names(transformer, [s.strip() for s in args.lora_layers.split(",") if s.strip()])
    config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha or args.lora_rank,
                        init_lora_weights="gaussian", target_modules=targets)
    transformer.add_adapter(config)
    transformer.to(DEVICE)
    cast_training_params(transformer, dtype=torch.float32)  # LoRA weights in float32
    if args.init_lora:
        load_lora_into(transformer, args.init_lora)
    n_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"LoRA: rank {args.lora_rank}, {len(targets)} layers, {n_params / 1e6:.1f} M trainable parameters")
    return transformer


def load_lora_into(transformer, path):
    """Load diffusers-format LoRA weights into the existing adapter (warm start)."""
    state = FluxKontextPipeline.lora_state_dict(path)
    state = {k[len("transformer."):]: v for k, v in state.items() if k.startswith("transformer.")}
    state = convert_unet_state_dict_to_peft(state)
    result = set_peft_model_state_dict(transformer, state, adapter_name="default")
    unexpected = getattr(result, "unexpected_keys", None) or []
    print(f"init LoRA {path}: {len(state)} tensors, {len(unexpected)} unexpected keys")


def sigma_table(args):
    """Training sigmas of the scheduler (same table as the diffusers example), descending."""
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    return sched.sigmas.clone().float()


def sample_sigma(args, sigmas, generator, pool="unspecified"):
    """Training sigma ([1] tensor) of one sample of the given pool."""
    if pool == "designated" and args.designated_sigma == "logit_normal":
        # drawn directly in sigma space: a positive mean means more high-noise samples
        z = torch.randn(1, generator=generator) * args.designated_sigma_std + args.designated_sigma_mean
        return torch.sigmoid(z).clamp(float(sigmas.min()), float(sigmas.max()))
    u = compute_density_for_timestep_sampling(weighting_scheme=args.weighting_scheme, batch_size=1,
                                              logit_mean=args.logit_mean, logit_std=args.logit_std,
                                              mode_scale=args.mode_scale, generator=generator)
    idx = (u * len(sigmas)).long().clamp(0, len(sigmas) - 1)
    return sigmas[idx]


# --------------------------------------------------------------------------------------
# Forward / loss
# --------------------------------------------------------------------------------------
def forward_loss(transformer, role, ex, sigma, noise, args):
    """Flow-matching loss of one example. sigma: [1] tensor; noise: like the target latent or None."""
    x0 = ex["target"].unsqueeze(0).to(DEVICE, DTYPE)                      # [1, 16, H, W]
    _, _, h, w = x0.shape
    if noise is None:
        noise = torch.randn_like(x0)
    noise = noise.to(DEVICE, DTYPE)
    s = sigma.to(DEVICE, torch.float32)
    xt = ((1.0 - s.view(1, 1, 1, 1)) * x0.float() + s.view(1, 1, 1, 1) * noise.float()).to(DTYPE)

    tgt_tokens = pack_latents(xt)
    tgt_ids = FluxKontextPipeline._prepare_latent_image_ids(1, h // 2, w // 2, DEVICE, DTYPE)
    refs = [r.unsqueeze(0).to(DEVICE, DTYPE) for r in ex["refs"]]
    ref_tokens, ref_ids, spans = build_reference_tokens(refs, args.layout, (h // 2, w // 2),
                                                        main_pos=marker_pos(args.role, ex["main_pos"]),
                                                        main_t=getattr(args, "main_t", MAIN_T))
    hidden = torch.cat([tgt_tokens, ref_tokens.to(DEVICE, DTYPE)], dim=1)
    img_ids = torch.cat([tgt_ids, ref_ids.to(DEVICE, DTYPE)], dim=0)

    pr = ex["prompt"]
    if role is not None:
        role.set_main(spans, ex["main_pos"])
    try:
        with torch.autocast("cuda", dtype=DTYPE):
            pred = transformer(
                hidden_states=hidden,
                encoder_hidden_states=pr["prompt_embeds"].unsqueeze(0).to(DEVICE, DTYPE),
                pooled_projections=pr["pooled_prompt_embeds"].unsqueeze(0).to(DEVICE, DTYPE),
                timestep=s,                                    # the transformer multiplies by 1000
                img_ids=img_ids,
                txt_ids=pr["text_ids"].to(DEVICE, DTYPE),
                guidance=torch.full((1,), args.guidance_scale, device=DEVICE),
                return_dict=False,
            )[0]
    finally:
        if role is not None:
            role.clear()
    pred = pred[:, : tgt_tokens.shape[1]].float()
    target = pack_latents((noise.float() - x0.float()))       # packing is a permutation of the latent
    weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas=s).view(1, 1, 1).float()
    return (weighting * (pred - target) ** 2).mean()


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
def fixed_val_set(rows, store, args):
    """Fixed validation examples: order, target, noise and sigmas do not change between evaluations."""
    rows = sorted(rows, key=lambda r: r["id"])
    rng = random.Random(1234)
    rng.shuffle(rows)
    out = []
    for i, row in enumerate(rows[: args.val_samples]):
        ex = make_example(row, row["targets"][0], store, args.role, random.Random(i), shuffle=not args.no_shuffle)
        gen = torch.Generator().manual_seed(10_000 + i)
        noises = [torch.randn(ex["target"].unsqueeze(0).shape, generator=gen) for _ in range(3)]
        out.append((ex, noises))
    return out


@torch.no_grad()
def validation_loss(transformer, role, val_set, args):
    transformer.eval()
    losses = defaultdict(list)
    for ex, noises in val_set:
        for sig, noise in zip((0.25, 0.5, 0.75), noises):
            loss = forward_loss(transformer, role, ex, torch.tensor([sig]), noise, args)
            losses[ex["pool"]].append(loss.item())
    transformer.train()
    out = {f"val_loss_{k}": sum(v) / len(v) for k, v in losses.items()}
    all_losses = [x for v in losses.values() for x in v]
    out["val_loss"] = sum(all_losses) / max(1, len(all_losses))
    return out


def parse_sigmas(text):
    return [float(x) for x in text.split(",") if x.strip()]


def designation_gap_set(rows, store, args):
    """Fixed designated samples for the designation gap, each with all its designations.

    Whole folders (all their mains) first, most mains first, at most --gap-samples rows. For each
    row the references are shuffled once (fixed seed) and the same order is used for every
    designation, so only the designation differs (slot mode still moves the designated input
    first). Noise is drawn once per row and sigma.
    """
    by_folder = defaultdict(list)
    for r in rows:
        if r["pool"] == "designated":
            by_folder[r["folder"]].append(r)
    picked = []
    for folder in sorted(by_folder, key=lambda f: (-len(by_folder[f]), f)):
        picked += sorted(by_folder[folder], key=lambda r: r["main"])
        if len(picked) >= args.gap_samples:
            break
    picked = picked[: args.gap_samples]

    sigmas = parse_sigmas(args.gap_sigmas)
    out = []
    for i, row in enumerate(picked):
        target = row["targets"][0]

        def example(main):
            # a fresh RNG with the same seed -> the same shuffle for every designation
            return make_example(dict(row, main=main), target, store, args.role, random.Random(30_000 + i),
                                shuffle=not args.no_shuffle)

        right = example(row["main"])
        wrong = [example(j) for j in range(len(row["inputs"])) if j != row["main"]]
        gen = torch.Generator().manual_seed(40_000 + i)
        shape = right["target"].unsqueeze(0).shape
        noises = {sig: [torch.randn(shape, generator=gen) for _ in range(args.gap_noises)] for sig in sigmas}
        out.append({"id": row["id"], "right": right, "wrong": wrong, "none": example(None), "noises": noises})
    return out


@torch.no_grad()
def designation_gap(transformer, role, gap_set, args):
    """Loss differences between right, wrong and no designation (see the module docstring)."""
    transformer.eval()
    per_sigma = defaultdict(lambda: defaultdict(list))
    for item in gap_set:
        for sig, noises in item["noises"].items():
            s = torch.tensor([sig])
            for noise in noises:
                l_right = forward_loss(transformer, role, item["right"], s, noise, args).item()
                l_wrong = [forward_loss(transformer, role, ex, s, noise, args).item() for ex in item["wrong"]]
                l_none = forward_loss(transformer, role, item["none"], s, noise, args).item()
                per_sigma[sig]["gap"].append(sum(l_wrong) / len(l_wrong) - l_right)
                per_sigma[sig]["gap_acc"].append(float(l_right < min(l_wrong)))
                per_sigma[sig]["gap_none"].append(l_none - l_right)
    transformer.train()
    out = {}
    for metric in ("gap", "gap_acc", "gap_none"):
        values = [v for sig in per_sigma for v in per_sigma[sig][metric]]
        out[metric] = sum(values) / max(1, len(values))
        for sig in per_sigma:
            v = per_sigma[sig][metric]
            out[f"{metric}_s{sig:g}"] = sum(v) / len(v)
    return out


def image_folders(rows, n):
    """Validation folders for the sample images: most designated mains first."""
    by_folder = defaultdict(list)
    for r in rows:
        by_folder[r["folder"]].append(r)
    ranked = sorted(by_folder, key=lambda f: (-sum(r["pool"] == "designated" for r in by_folder[f]), f))
    out = []
    for f in ranked[:n]:
        rs = by_folder[f]
        out.append(sorted(rs, key=lambda r: (-1 if r["main"] is None else r["main"])))
    return out


@torch.no_grad()
def sample_images(pipe, role, folders, store, args, step):
    pipe.transformer.eval()
    size = store.target_size
    blocks = []
    for rows in folders:
        inputs = [Image.open(p).convert("RGB") for p in rows[0]["inputs"]]
        results = []
        for row in rows:
            order, main_pos, _ = arrange(len(row["inputs"]), row["main"], args.role, shuffle=False)
            refs = [store.ref(row["inputs"][i]).unsqueeze(0).to(DEVICE, DTYPE) for i in order]
            pr = store.prompt(prompt_key(main_pos, args.role))
            with torch.autocast("cuda", dtype=DTYPE):
                out = pipe.generate_from_latents(
                    refs, main_pos, args.role, args.layout, role, height=size, width=size, main_t=args.main_t,
                    prompt_embeds=pr["prompt_embeds"].unsqueeze(0).to(DEVICE, DTYPE),
                    pooled_prompt_embeds=pr["pooled_prompt_embeds"].unsqueeze(0).to(DEVICE, DTYPE),
                    num_inference_steps=args.image_steps, guidance_scale=args.guidance_scale,
                    generator=torch.Generator(device=DEVICE).manual_seed(args.seed))
            label = "no main" if row["main"] is None else f"main = input_{row['main']}"
            results.append((label, out.images[0], Image.open(row["targets"][0]).convert("RGB")))
        blocks.append((Path(rows[0]["folder"]).name, inputs, results))
    pipe.transformer.train()
    path = args.output_dir / "samples" / f"step_{step:06d}.jpeg"
    save_sample_grid(blocks, path, step)
    return path


def save_sample_grid(blocks, path, step, tile=256):
    """Per folder: inputs | generated (one per condition) | targets (same columns)."""
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    gap, cap, head = 8, 20, 24
    n_cols = max(max(len(inp), len(res)) for _, inp, res in blocks)
    W = gap + n_cols * (tile + gap)
    block_h = head + 3 * (tile + cap + gap)
    canvas = Image.new("RGB", (W, 30 + len(blocks) * block_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 6), f"step {step}: row 1 inputs, row 2 generated, row 3 targets", fill="black", font=font)
    y0 = 30
    for name, inputs, results in blocks:
        draw.text((gap, y0 + 4), name, fill="black", font=font)
        rows = [[(im, f"input_{i}") for i, im in enumerate(inputs)],
                [(gen, label) for label, gen, _ in results],
                [(tgt, "target") for _, _, tgt in results]]
        for r, cells in enumerate(rows):
            y = y0 + head + r * (tile + cap + gap)
            for c, (im, text) in enumerate(cells):
                x = gap + c * (tile + gap)
                canvas.paste(im.resize((tile, tile)), (x, y))
                draw.text((x + 2, y + tile + 2), text, fill="black", font=font)
        y0 += block_h
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)


# --------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------
def save_checkpoint(out_dir: Path, transformer, role, optimizer, lr_sched, step, rank_states, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    lora_state = get_peft_model_state_dict(transformer)
    FluxKontextPipeline.save_lora_weights(
        out_dir / "lora",
        transformer_lora_layers={k: v.detach().to(DTYPE).cpu() for k, v in lora_state.items()},
        transformer_lora_adapter_metadata=transformer.peft_config["default"].to_dict(),
    )
    if role is not None:
        role.save(out_dir / "role_embedding.safetensors")
    torch.save({
        "step": step,
        "lora": {k: v.detach().cpu() for k, v in lora_state.items()},
        "e_main": role.e_main.detach().cpu() if role is not None else None,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_sched.state_dict(),
        **rank_states[0],                  # random states of the first GPU (single-GPU format)
        "rank_states": rank_states,        # random states of every GPU
        "world_size": len(rank_states),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }, out_dir / "train_state.pt")
    print(f"saved {out_dir}")


def load_checkpoint(path: Path, transformer, role, optimizer, lr_sched, sampler, data_rng, sigma_gen, args):
    state = torch.load(path / "train_state.pt", map_location="cpu", weights_only=False)
    set_peft_model_state_dict(transformer, state["lora"], adapter_name="default")
    cast_training_params(transformer, dtype=torch.float32)
    if role is not None and state["e_main"] is not None:
        with torch.no_grad():
            role.e_main.copy_(state["e_main"].to(role.e_main.device))
    optimizer.load_state_dict(state["optimizer"])
    lr_sched.load_state_dict(state["lr_scheduler"])
    states = state.get("rank_states") or [{k: state[k] for k in ("sampler_rng", "data_rng", "torch_rng", "cuda_rng")}]
    if len(states) == WORLD:
        st = states[RANK]
        sampler.rng.setstate(st["sampler_rng"])
        data_rng.setstate(st["data_rng"])
        if "sigma_gen" in st:
            sigma_gen.set_state(st["sigma_gen"])
        torch.set_rng_state(st["torch_rng"])
        torch.cuda.set_rng_state(st["cuda_rng"])
    else:
        # other number of GPUs: new, distinct random states per GPU, derived from the step
        seed = args.seed + 7919 * state["step"] + 1000 * RANK
        sampler.rng.seed(seed + 2)
        data_rng.seed(seed + 1)
        sigma_gen.manual_seed(seed + 3)
        torch.manual_seed(seed)
        print(f"[warn] checkpoint saved with {len(states)} GPU(s), resuming on {WORLD}: random states reseeded")
    print(f"resumed from {path} at step {state['step']}")
    return state["step"]


# --------------------------------------------------------------------------------------
def main():
    args = parse_args()
    launch_with_gpus(args)   # --gpus with several GPUs: restarts with torchrun and exits here
    setup_distributed()
    args.world_size = WORLD
    torch.manual_seed(args.seed)  # same seed on every GPU: identical LoRA initialization
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or args.data_dir / "cache"

    # ---------------- data ----------------
    if args.role == "tid" and args.layout != "canvas":
        raise ValueError("--role tid needs --layout canvas")
    train_rows = load_rows(args.data_dir, "train")
    val_rows = load_rows(args.data_dir, "val")
    if args.overfit:
        names = [x.strip() for x in (args.overfit_folders or "").split(",") if x.strip()]
        pool = train_rows + val_rows if names else train_rows
        train_rows = pick_overfit_rows(pool, args.overfit, seed=args.seed, folders=names or None)
        val_rows = train_rows
        print("overfitting test on: " + ", ".join(r["id"] for r in train_rows))
    if not train_rows:
        raise ValueError(f"No training samples in {args.data_dir}")
    if not val_rows:
        print("[warn] no validation samples: using training samples for validation")
        val_rows = train_rows
    schedule = PoolSchedule(args.unspec_schedule or (OVERFIT_SCHEDULE if args.overfit else DEFAULT_SCHEDULE))

    store = LatentStore(cache_dir)
    if store.config.get("base_model") != args.base_model:
        print(f"[warn] cache built with {store.config.get('base_model')}, training {args.base_model}")
    if not 0.0 <= args.ref_aug_prob <= 1.0:
        raise ValueError("--ref-aug-prob must be in [0, 1]")
    n_augs = store.ref_augs if args.ref_aug_prob > 0 else 0
    if args.ref_aug_prob > 0:
        if store.ref_augs <= 0:
            raise ValueError(f"--ref-aug-prob {args.ref_aug_prob} but {cache_dir} has no augmented references: "
                             f"run cache_features.py --ref-augs N first")
        print(f"reference augmentation: p = {args.ref_aug_prob} on {args.ref_aug_on} reference(s), "
              f"{store.ref_augs} cached versions per input ({store.config.get('aug')})")
    args.cache_ref_augs = store.ref_augs
    args.cache_aug = store.config.get("aug")
    if args.preload:
        # augmented versions are only used for training rows
        paths = set(all_latent_paths(train_rows, store.ref_size, store.target_size, n_augs))
        paths |= set(all_latent_paths(val_rows, store.ref_size, store.target_size))
        store = LatentStore(cache_dir, sorted(paths, key=lambda t: (str(t[0]), t[1], -1 if t[2] is None else t[2])))
    for main_pos in [None] + list(range(4)):
        if args.role != "index" and main_pos not in (None, 0):
            continue
        store.prompt(prompt_key(main_pos, args.role))  # fail early if a template is missing

    # every GPU draws its own samples, orders, sigmas and noise (offset 1000 * rank; rank 0 = single GPU)
    sampler = PoolSampler(train_rows, seed=args.seed + 2 + 1000 * RANK)
    data_rng = random.Random(args.seed + 1 + 1000 * RANK)
    print(f"train: {len(train_rows)} samples {sampler.counts()}; val: {len(val_rows)} samples; "
          f"unspecified-pool schedule {schedule.spec!r}")
    print(f"GPUs: {WORLD}; samples per optimizer step: {args.grad_accum} x {WORLD} = {args.grad_accum * WORLD}")

    # ---------------- model ----------------
    transformer = build_transformer(args)
    role = None
    if args.role == "embed":
        role = MainRoleEmbedding(transformer.inner_dim).to(DEVICE).attach(transformer)
    sigmas = sigma_table(args)
    sigma_gen = torch.Generator().manual_seed(args.seed + 3 + 1000 * RANK)

    groups = [{"params": [p for p in transformer.parameters() if p.requires_grad],
               "lr": args.lr, "weight_decay": args.weight_decay}]
    if role is not None:
        groups.append({"params": [role.e_main], "lr": args.role_lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)))
    trainable = [p for g in groups for p in g["params"]]
    broadcast_parameters(trainable)       # identical start on every GPU
    torch.manual_seed(args.seed + RANK)   # from here on: own noise per GPU (rank 0 = single GPU)

    step = 0
    if args.resume:
        step = load_checkpoint(args.resume, transformer, role, optimizer, lr_sched, sampler, data_rng, sigma_gen,
                               args)
        broadcast_parameters(trainable)

    # validation, gap and sample images only on the first GPU
    main_proc = RANK == 0
    pipe = None
    if main_proc and args.image_every:
        # sample-image pipeline: the training transformer + VAE, prompts from the cache
        vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae", torch_dtype=DTYPE).to(DEVICE)
        pipe = MultiRefKontextPipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model, subfolder="scheduler"),
            vae=vae, text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None, transformer=transformer)
        pipe.set_progress_bar_config(disable=True)
    val_set = fixed_val_set(val_rows, store, args) if (main_proc and args.val_every) else []
    gap_set = designation_gap_set(val_rows, store, args) if (main_proc and args.gap_every) else []
    if args.gap_every and not gap_set:
        print("[warn] no designated validation samples: designation gap disabled")
    elif gap_set:
        n_cond = sum(len(item["wrong"]) + 2 for item in gap_set)  # right + wrong + none
        n_fwd = n_cond * len(parse_sigmas(args.gap_sigmas)) * args.gap_noises
        print(f"designation gap: {len(gap_set)} samples ({', '.join(item['id'] for item in gap_set)}), "
              f"{n_fwd} forward passes per evaluation")
    folders = image_folders(val_rows, args.image_folders) if (main_proc and args.image_every) else []

    writer = None
    if args.tensorboard and main_proc:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.output_dir / "tb")
    log_path = args.output_dir / "log.jsonl"
    if main_proc:
        with open(args.output_dir / "args.json", "w") as f:
            json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    def log(record):
        if not main_proc:
            return
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if writer is not None:
            for k, v in record.items():
                if k != "step" and isinstance(v, (int, float)):
                    writer.add_scalar(k, v, record["step"])

    if step == 0 and folders and not args.no_images_at_start:
        print(f"sample images: {sample_images(pipe, role, folders, store, args, 0)}")
    if step == 0 and val_set:
        rec = {"step": 0, **validation_loss(transformer, role, val_set, args)}
        log(rec)
        print("step 0 " + " ".join(f"{k} {v:.4f}" for k, v in rec.items() if k != "step"))
    if step == 0 and gap_set:
        rec = {"step": 0, **designation_gap(transformer, role, gap_set, args)}
        log(rec)
        print("step 0 " + " ".join(f"{k} {v:.4f}" for k, v in rec.items() if k != "step"))
    barrier()  # the other GPUs wait for the step-0 evaluation

    # ---------------- training ----------------
    transformer.train()
    window = defaultdict(list)
    t0 = time.time()
    while step < args.max_steps:
        p_unspec = schedule(step / args.max_steps)
        for _ in range(args.grad_accum):
            row, target = sampler.sample(p_unspec)
            ex = make_example(row, target, store, args.role, data_rng, shuffle=not args.no_shuffle,
                              aug_prob=args.ref_aug_prob, aug_on=args.ref_aug_on)
            if args.ref_aug_prob > 0:
                window["ref_aug_share"].append(sum(a is not None for a in ex["augs"]) / len(ex["augs"]))
            sigma = sample_sigma(args, sigmas, sigma_gen, ex["pool"])
            window[f"sigma_{ex['pool']}"].append(float(sigma))
            loss = forward_loss(transformer, role, ex, sigma, None, args)
            (loss / args.grad_accum).backward()
            window["loss"].append(loss.item())
            window[f"loss_{ex['pool']}"].append(loss.item())
            window["designated"].append(float(ex["pool"] == "designated"))
        average_gradients(trainable)  # same averaged gradient on every GPU -> same update
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        lr_sched.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        window["grad_norm"].append(float(grad_norm))

        if step % args.log_every == 0 or step == args.max_steps:
            merged = gather_window(window)  # samples of every GPU
            rec = {"step": step, **{k: sum(v) / len(v) for k, v in merged.items() if v},
                   "p_unspec": p_unspec, "lr": lr_sched.get_last_lr()[0],
                   "sec_per_step": (time.time() - t0) / args.log_every}
            if role is not None:
                rec["e_main_norm"] = role.e_main.detach().norm().item()
                ref_norm = role.pop_reference_norm()  # None if no designated forward since the last log
                if ref_norm:
                    rec["ref_token_norm"] = ref_norm
                    rec["e_main_ratio"] = rec["e_main_norm"] / ref_norm
            rec["designated_share"] = rec.pop("designated")
            log(rec)
            print(" ".join([f"step {step}"] + [f"{k} {v:.4g}" for k, v in rec.items() if k != "step"]))
            window.clear()
            t0 = time.time()
        if val_set and step % args.val_every == 0:  # val_set, gap_set, folders: first GPU only
            rec = {"step": step, **validation_loss(transformer, role, val_set, args)}
            log(rec)
            print(f"step {step} " + " ".join(f"{k} {v:.4f}" for k, v in rec.items() if k != "step"))
        if gap_set and step % args.gap_every == 0:
            rec = {"step": step, **designation_gap(transformer, role, gap_set, args)}
            log(rec)
            print(f"step {step} " + " ".join(f"{k} {v:.4f}" for k, v in rec.items() if k != "step"))
        if folders and step % args.image_every == 0:
            print(f"sample images: {sample_images(pipe, role, folders, store, args, step)}")
        saving = step % args.save_every == 0 or step == args.max_steps
        if saving:
            broadcast_parameters(trainable)  # guard against numerical drift between GPUs
            rank_states = gather_rng_states(sampler, data_rng, sigma_gen)
            if main_proc:
                save_checkpoint(args.output_dir / f"checkpoint-{step:06d}", transformer, role, optimizer, lr_sched,
                                step, rank_states, args)
        evaluating = ((args.val_every and step % args.val_every == 0)
                      or (args.gap_every and step % args.gap_every == 0)
                      or (args.image_every and step % args.image_every == 0))
        if saving or evaluating:
            barrier()  # the other GPUs wait for the first one

    if writer is not None:
        writer.close()
    print("done")
    if WORLD > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
