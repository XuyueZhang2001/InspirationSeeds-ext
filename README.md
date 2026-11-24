# Inspiration Seeds: Multi-Input Decomposition (work in progress)

This repository builds on [Inspiration Seeds](https://arxiv.org/abs/2602.08615) (Goldberg, Richardson, Vinker), which combines two input images into a new object using FLUX.1 Kontext with a LoRA trained on synthetic triplets. The triplets come from decomposing a source image into two concept clusters of CLIP-SAE features.

The long-term goal of this fork is to support **N input images with an optional designated main subject** (earlier plan: controllable per-image weight levels). The repository now covers data generation and a first fine-tuning pipeline:

1. decomposing a source image into **N semantically distinct reference images** (N = 2–4, chosen per image, 768 px): **`decompose_v7_2.py`** (adaptive SAE feature grouping + an image-space subject subspace), run over a grid of settings by **`run_sweep.sh`**;
2. scoring the decompositions of the sweep and picking, per source image, one folder for the no-main pool and one for the main pool: **`score_sweep.py`** (CLIP / DINOv2 checks + a VLM; replaces the hand-picking and `screen_sources.py`);
3. synthesizing **designated-subject target images** (1024 px): **`synthesize_v2_1.py`**: for every input designated as the main subject, a target pulled toward that input, so that the same input set gives different targets for different designations;
4. selecting the targets by hand, with automatic suggestions: **`select_targets.py`**;
5. building the training data: **`build_dataset.py`** and **`cache_features.py`**;
6. fine-tuning FLUX.1 Kontext dev (new LoRA; the main image is marked by its position id) on 2–4 reference images with an optional main image: **`train_multiref.py`**; evaluation with **`eval_multiref.py`**, inference with **`generate_multi.py`**.

Last README update: 2026-11-23 (sweep and automatic scoring: `run_sweep.sh`, `score_sweep.py`).

---

## Contents

| File | Purpose |
|---|---|
| `generate.py` | Original inference script: combines two images with FLUX.1 Kontext + the released LoRA (unchanged) |
| `decompose_original.py` | Original two-cluster decomposition script (unchanged, renamed) |
| **`decompose_v7_2.py`** | **Recommended** decomposition: adaptive grouping of SAE features, coverage expansion, merging, shared-part weakening, subject kept in one input (see below) |
| `decompose_groups.py`, `decompose_v4.py` … `decompose_v7_1.py` | Earlier versions of the adaptive decomposition, kept for comparison (history below) |
| `decompose.py` | Earlier fixed-K baseline (k-means into K = 2, 3, 4 clusters); no longer documented |
| **`run_sweep.sh`** | Runs `decompose_v7_2.py` (unchanged) once per setting of a `--top-k` × `--clean-frac` × `--seed` grid, then `score_sweep.py` (see *Sweep and automatic scoring*) |
| **`score_sweep.py`** | Scores every folder of a sweep (CLIP / DINOv2 checks, Qwen3-VL main-subject probabilities), picks per source image the best no-main folder (level 1) and the best main folder (level 2), writes a review (`_review/`) and optionally exports the picks to `data/no_main/` and `generate_source/` |
| `screen_sources.py` | Superseded by `score_sweep.py` (level 2). Earlier first pass over `data/no_main/`: folders with at least two distinct, source-related subjects were copied to `generate_source/screening/` for the manual choice of synthesis sources |
| **`synthesize_v2_1.py`** | **Recommended** designated-subject target synthesis for `decompose_v7_2.py` outputs: closeness-dependent 2 × 2 grid of interpolation weight × img2img strength (see below) |
| `synthesize_v2.py` | Previous version: the same synthesis with a full 3 × 3 grid for every main |
| `synthesize_target.py` | Earlier level-based weighted target synthesis for `decompose.py` outputs |
| **`select_targets.py`** | Suggests which synthesized targets to keep and writes review folders (`keep/`, `drop/`, `_overview.jpeg`) for the manual selection (see below) |
| **`build_dataset.py`** | Training / validation manifests: unspecified samples (decomposition + source image) and designated samples (decomposition + kept targets) |
| **`cache_features.py`** | Caches VAE latents (references 512 px, targets 1024 px) and the prompt embeddings, so training only needs the transformer |
| **`train_multiref.py`** | LoRA fine-tuning of FLUX.1 Kontext dev with 2–4 references and an optional main image (default: marked by position id $t = 3$); pool schedule, designation gap, validation images, checkpoints, overfitting test |
| **`eval_multiref.py`** | Designation gap and first-position bias of a checkpoint, with rotation images |
| **`generate_multi.py`** | Inference with 2–4 images and an optional main image (`--main k`), with a trained LoRA (and role embedding for `--role embed`) |
| `check_multiref.py` | Sanity checks of the multi-reference conditioning against the stock Kontext pipeline, plus baseline images of the base model |
| `multiref/` | Shared code: prompt templates (`prompts.py`), manifest / cache helpers (`data.py`), reference encoding, position ids, role embedding and pipeline (`conditioning.py`), training data sampling (`train_data.py`) |
| `fetch_samples.py` | Downloads a random subset of the paper's training data from Hugging Face |
| `probe_inputs.py` | Checks whether the released model uses both inputs (swap / duplicate / blank conditions, DINOv2 and CLIP metrics) |
| `summarize_none.py` | Prints a table of the unweighted (`none`) target measurements across folders |
| `check_sae.py` | Small diagnostic that prints the SAE structure and decoder bias |
| `assets/` | Example inputs from the original repository |
| `docs/` | Paper PDF and teaser |

---

## Pipeline overview

All scripts are run from the repository root; the defaults of every script follow this layout.

```
samples/sources/              source images (fetch_samples.py)
output/decompose/sweep/       run_sweep.sh: raw decompose_v7_2.py runs, one subfolder per setting
    topk<K>_c<clean>_s<seed>/<decompose_v7_2.py folders>   + measure.json, main_choice.json, score.json
    _sources/                 score_sweep.py: VLM concept inventory per source image, same-kind cache
    _review/                  score_sweep.py: per-image sheets, overview pages, folders.csv, inputs.csv, summary.json
    _logs/                    one decompose_v7_2.py log per setting
    overrides.csv             (optional, by hand) replaces automatic picks of single images
output/decompose/<run>/       other raw decompose_v7_2.py runs (nothing else goes to output/)
data/no_main/                 unspecified pool: no-main AND main picks of score_sweep.py (--export-no-main),
                              plus folders copied by hand earlier
generate_source/              synthesis sources: main picks of score_sweep.py (--export-main),
                              plus folders chosen by hand earlier
    _exported.json            (in both targets) folders written by score_sweep.py; others are never touched
data/synth/                   synthesize_v2_1.py targets
data/selection/review/        select_targets.py review folders (manual keep / drop): the designated pool
data/multiref/                build_dataset.py manifests + cache_features.py cache
runs/                         training runs
```

```bash
SRC=samples/sources

# 1. decomposition sweep (768 px inputs): decompose_v7_2.py for every top-k x clean-frac x seed
#    setting, then score_sweep.py (review in output/decompose/sweep/_review/, nothing exported yet)
./run_sweep.sh $SRC                         # default grid: TOPK="32 64" CLEAN="0.3 0.5 0.7" SEEDS="0 1"

# 2. check the picks (_review/overview_*.jpeg, <stem>.jpeg), correct single images in
#    output/decompose/sweep/overrides.csv if needed, rescore (seconds, everything is cached) and export:
#    no-main + main picks -> data/no_main/ (unspecified pool), main picks -> generate_source/
python score_sweep.py --export-no-main data/no_main --export-main generate_source

# 3. designated-subject targets (1024 px) for generate_source/
python synthesize_v2_1.py                   # generate_source -> data/synth

# 4. suggestions + manual selection in data/selection/review/<folder>/main<k>/keep|drop
#    (criteria: Target selection, Selection criteria)
python select_targets.py                    # data/synth -> data/selection

# 5. training data: manifests and cached latents / prompt embeddings
python build_dataset.py                     # data/no_main + data/selection/review -> data/multiref
python cache_features.py --manifest-dir data/multiref --target-size 1024

# 6. checks, overfitting test, training
python check_multiref.py --folder data/no_main/<folder>
python train_multiref.py --data-dir data/multiref --output-dir runs/overfit --overfit 8 \
    --max-steps 1500 --warmup-steps 50 --grad-accum 1 --val-every 100 --image-every 250 --save-every 500
python eval_multiref.py --checkpoint runs/overfit/checkpoint-001500
python train_multiref.py --data-dir data/multiref --output-dir runs/v1 --max-steps 10000

# 7. inference (default role tid, as in training)
python generate_multi.py --folder data/no_main/<folder> --main 2 --lora runs/v1/checkpoint-010000/lora
```

| Step | Resolution |
|---|---|
| Decomposed inputs (`decompose_v7_2.py --size`) | 768 × 768 |
| Synthesized targets (`synthesize_v2_1.py --size`) | 1024 × 1024 (inputs upsampled as img2img init images) |
| References in the model (`cache_features.py --ref-size`) | 512 × 512 (1024 tokens each), as in the original Inspiration Seeds |
| Targets in the model (`cache_features.py --target-size`) and outputs | 1024 × 1024 (4096 tokens) |

---

## Setup

### Hardware

- An NVIDIA GPU is required (both scripts hard-code `cuda`).
- `decompose*.py` (Kandinsky 2.2 + SAE): about 12–16 GB of VRAM.
- `synthesize_v2_1.py` (Kandinsky 2.2 only, no SAE): at most the same as the decomposition.
- `generate.py`, `generate_multi.py`, `check_multiref.py` (FLUX.1 Kontext dev, ~12B parameters, bf16): about 24 GB+ of VRAM, or use `pipe.enable_model_cpu_offload()`.
- `train_multiref.py` (transformer in bf16 + LoRA in fp32, gradient checkpointing, no quantization): developed on an 80 GB GPU; about 4 s per sample with 4 references and a 1024 px target. Single GPU; choose it with `CUDA_VISIBLE_DEVICES=1 python train_multiref.py ...`.
- `cache_features.py` (VAE + text encoders, no transformer): less than the generation scripts.

### Environment

The scripts carry PEP 723 headers for `uv run`, but on our server `uv run` installed a PyTorch build that was too new for the driver (CUDA 12.8). Using a manually created environment is more reliable:

```bash
uv venv seeds-env --python 3.12          # or: python3 -m venv seeds-env
source seeds-env/bin/activate

# PyTorch built for the driver's CUDA version (check "CUDA Version" in nvidia-smi)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# decomposition
uv pip install numpy diffusers scikit-learn tqdm transformers huggingface-hub
# Hugging Face sample download
uv pip install datasets pillow
# generation (generate.py)
uv pip install "diffusers @ git+https://github.com/huggingface/diffusers.git" peft accelerate protobuf sentencepiece

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Notes:
- An environment created with `uv venv` has no `pip`; install packages with `uv pip install`, not `pip install`.
- Run scripts with `python ...` inside the activated environment, not `uv run ...` (which would build a separate environment from the script header).
- The warning `CLIPImageProcessor requires torchvision` is harmless. Install `torchvision` from the same CUDA index to silence it.
- Fine-tuning (`train_multiref.py`) needs `peft` (installed with the generation packages); `--tensorboard` additionally needs `uv pip install tensorboard`.

### Hugging Face access

`generate.py` uses `black-forest-labs/FLUX.1-Kontext-dev`, which is gated. Accept the license on its model page, create a read token, and run `hf auth login`. The decomposition models (Kandinsky 2.2, `gytdau/clip-sae-128`) and the training dataset are not gated.

---

## Recommended decomposition (`decompose_v7_2.py`)

### Goal

The decomposed images are used as the inputs of synthetic training triplets (target = source image), so a good decomposition should be:

1. **distinct**: each input shows a different aspect of the source; the same concept should not appear in every input (otherwise its weight cannot be controlled);
2. **covering**: the main content of the source appears in at least one input;
3. **clean**: no content that is unrelated to the source (e.g. animals produced by the SAE bias);
4. **cheap**: only matrix operations on the CLIP embedding plus one Kandinsky decode per input (it is a data-generation step).

The guiding rule of v7.2 is: **every concept should appear in exactly one input.**

### Method

Notation: $e$ is the CLIP image embedding of the source (Kandinsky 2.2 prior image encoder, 1280-d), $a = \mathrm{SAE}_{enc}(e)$ (`gytdau/clip-sae-128`, 163,840 features), $d_j$ the $j$-th decoder column (decoder without bias), $u_j = d_j/\lVert d_j\rVert$.

1. **Features.** Keep the `--top-k` (64) most activated features for grouping. The other active features are used in step 6.
2. **Grouping.** Distance between features $1 - u_j^\top u_k$. The threshold $\delta$ is the `--delta-quantile` (0.2) quantile of the pairwise distances, capped at `--delta-cap` (0.9). Groups are built bottom-up from single features (FINCH-style, ConceptExpress-style threshold): two groups merge if their centroid distance is $\le \delta$ and the mean pairwise distance inside the merged group stays $\le \delta$. A quantile is used instead of the maximum because SAE decoder directions are nearly orthogonal; a max-based threshold merges everything into one group.
3. **Roles.** Salience of a group $= \sum_{j} a_j \lVert d_j\rVert$ over the group divided by the same sum over the top-k features. A group is **main** if it has $\ge 2$ features and salience $\ge$ `--min-salience` (0.1); at most `--n-max` (4) main groups, by salience. All other groups are **minor**.
4. **Cleaning.** In each main group, the `--clean-frac` fraction of features closest to another main group's centroid is removed (as in the original paper); these become **ambiguous** features.
5. **Group contributions.** $D_k = W_{dec}\, a^{(k)}$ (no bias), also $D_{minor}$ and $D_{amb}$.
6. **Coverage expansion.** Every active feature outside the top-k is assigned to the main group whose centroid direction is most similar, if the cosine is $\ge$ `--assign-threshold` (0.2). $D_i$ then includes the assigned features.
7. **Merging.** Inputs are built as in step 8 with $\alpha_r = 1$; if two inputs have cosine $\ge$ `--merge-threshold` (0.85), their groups are merged (`--merge-mode absolute`), never below `--n-min` (2) main groups. Fewer than 2 main groups → the image is marked *not decomposable* (only `recon.jpeg` is generated).
8. **Inputs with a weakened shared part.** The **shared part** is what every input has in common:
   $$S = e - \sum_{k\,\in\,\text{main}} D_k - \beta_{minor} D_{minor} - \beta_{amb} D_{amb}$$
   It contains the unassigned features, the SAE decoder bias ($\lVert \mathrm{dec}(0)\rVert \approx 26$; decoded alone it gives a brown furry animal) and the SAE reconstruction error. Input $i$ is
   $$e_i' = \alpha_r\, S + (1+\gamma)\, D_i, \qquad \text{rescaled to } \lVert e\rVert .$$
   With $\beta = \gamma = 1$ and $\alpha_r = 1$ this equals "remove all other groups and double the own group". $\alpha_r < 1$ makes the inputs more distinct, but $S$ is also the "base" that lets the decoder produce a coherent image, so too small values give artifacts. Because of the rescaling, $\alpha_r$ effectively sets the ratio between $S$ and $D_i$, not the overall strength.
9. **Subject identification.** The prior's CLIP text encoder encodes "a photo of a {noun}" for a built-in list of ~300 common nouns (or `--noun-vocab`); the noun with the highest cosine to $e$ is the subject (`--subject-map` overrides it per image). Text is used only for this ranking and for the `cos_text` metric.
10. **Image-space subject subspace.** For 40 templates (colors, materials, views, settings, styles), the Kandinsky prior predicts CLIP **image** embeddings for `template(noun)` and `template("object")`, `--prior-samples` (2) each, with the same noise for both (paired samples, `--prior-seed`):
    $$\Delta_k = \overline{\mathrm{prior}(\text{template}_k(\text{noun}))} - \overline{\mathrm{prior}(\text{template}_k(\text{object}))}$$
    The differences cancel the template words, so what remains is the noun's identity. SVD of the unnormalized $\Delta_k$; the top `--subject-rank` (4) right singular vectors form $Q$, and $P_{obj} = QQ^\top$. Results are cached per noun in `--text-cache`.
    *Why image space:* a subspace built from CLIP **text** embeddings (v7) barely overlaps with image embeddings because of the modality gap ($\lVert P e\rVert/\lVert e\rVert \approx 0.1$–$0.17$), so removing it did not remove the subject.
11. **Subject keeper.** The subject strength of each input is $\text{proj\_ratio}_i = \lVert P_{obj} e_i'\rVert / \lVert e_i'\rVert$. The input with the largest value (the **keeper**) keeps its subject unchanged; every other input is weakened:
    $$e_j'' = e_j' - (1 - \alpha_{obj})\, P_{obj}\, e_j' \quad (j \ne \text{keeper}), \qquad \text{rescaled to } \lVert e\rVert .$$
    Optional `--subject-restore` resets the keeper's subject component to the source level, for images where the subject is already weak in every input.
12. **Decoding.** Each input embedding is decoded by Kandinsky 2.2 (`--steps` 50, `--size` 768), with the same `--seed` for every input.

Steps 9–11 replaced two earlier ideas that did not work: weakening the subject in *every* input (small subjects such as a kite disappeared from all inputs), and a separate subject input $S + \gamma P_{obj} e$ (often dominated by the SAE bias, i.e. an unrelated animal).

### Usage

```bash
# default setting (alpha_r 0.7, cleaning 0.3), no debug images
python decompose_v7_2.py --image-dir samples/sources --output-dir output/decompose/v72

# one image, with diagnostic images
python decompose_v7_2.py --image samples/sources/<name>.jpeg --output-dir output/decompose/v72_test --debug-images

# another decoding seed (same embeddings, different renderings)
python decompose_v7_2.py --image-dir samples/sources --output-dir output/decompose/v72_seed2 --seed 2

# other cleaning fraction / shared-part strength
python decompose_v7_2.py --image-dir samples/sources --output-dir output/decompose/v72_c05 --clean-frac 0.5

# force the subject noun for some images
echo '{"14": "kite", "phone_01": "smartphone"}' > subjects.json
python decompose_v7_2.py --image-dir samples/sources --output-dir output/decompose/v72 --subject-map subjects.json
```

Existing output folders are skipped unless `--overwrite` is given, so use a new `--output-dir` for every setting. The first time a noun is used, its subspace costs 2 × 40 × `--prior-samples` prior runs (batched); afterwards it is read from `text_cache/`.

### Data-generation runs

The decompositions are used to build fine-tuning data, so every source image is decomposed with **several settings**, and one input set per pool is selected afterwards. The standard way is now the sweep of `run_sweep.sh` (a `--top-k` × `--clean-frac` × `--seed` grid) with the automatic selection of `score_sweep.py`; see *Sweep and automatic scoring* below.

The per-image folder name is `{stem}_topk{K}_s{seed}_c{clean-frac}_a{alpha-r}_obj{alpha-obj}` (default: `{stem}_topk64_s1_c0.3_a0.7_obj0.5`), so runs that differ in these parameters get different folder names. `--n-max` and `--subject-restore` are **not** in the name, so the `restore` and `n2` runs below would overwrite or skip the `base_s1` folders if they shared an output directory. Each run therefore still gets its own `--output-dir` (with `run_sweep.sh`: its own `OUT=`), which also keeps the runs easy to compare.

The single-setting runs below were the earlier hand-run settings. They change one thing at a time from the defaults (`--alpha-r 0.7 --clean-frac 0.3`, no debug images) and are still useful for parameters the sweep does not vary (`--alpha-r`, `--alpha-obj`, `--subject-restore`, `--n-max`), e.g. `OUT=output/decompose/sweep_a10 DECOMPOSE_ARGS="--alpha-r 1.0" ./run_sweep.sh samples/sources`.

| Run | Arguments (added to the defaults) | What changes | When it gives better pairs |
|---|---|---|---|
| `base_s1` | – | Defaults (`--alpha-r 0.7 --clean-frac 0.3 --seed 1`) | Default candidate |
| `base_s2` | `--seed 2` | Decoder seed only (same embeddings and `groups.json`) | A different rendering of the same decomposition; also a stability check |
| `base_s3` | `--seed 3` | Decoder seed only | As above |
| `c05` | `--clean-frac 0.5` | Stronger cleaning | Purer, more distinct groups; more content moved to the shared part or lost |
| `c02` | `--clean-frac 0.2` | Weaker cleaning | Fewer objects lost (e.g. small objects removed from every input); groups less pure |
| `a10` | `--alpha-r 1.0` | Shared part not weakened | More natural, coherent images; inputs share more content |
| `restore` | `--subject-restore` | Keeper's subject reset to the source level | The subject is weak or missing in every input |
| `obj03` | `--alpha-obj 0.3` | Subject weakened more in the non-keeper inputs | The subject still appears in several inputs |
| `n2` | `--n-max 2` | At most two inputs | Two-input pairs as in the paper; lower-salience groups become minor and are removed |

```bash
SRC=samples/sources          # source folder
OUT=output/decompose          # one subfolder per run

python decompose_v7_2.py --image-dir $SRC --output-dir $OUT/base_s1 --seed 1
python decompose_v7_2.py --image-dir $SRC --output-dir $OUT/base_s2 --seed 2
python decompose_v7_2.py --image-dir $SRC --output-dir $OUT/base_s3 --seed 3
python decompose_v7_2.py --image-dir $SRC --output-dir $OUT/c05     --clean-frac 0.5
python decompose_v7_2.py --image-dir $SRC --output-dir $OUT/c02     --clean-frac 0.2
python decompose_v7_2.py --image-dir $SRC --output-dir $OUT/a10     --alpha-r 1.0
python decompose_v7_2.py --image-dir $SRC --output-dir $OUT/restore --subject-restore
python decompose_v7_2.py --image-dir $SRC --output-dir $OUT/obj03   --alpha-obj 0.3
python decompose_v7_2.py --image-dir $SRC --output-dir $OUT/n2      --n-max 2
```

Resulting layout: `output/decompose/<run>/<stem>_topk64_s{seed}_c{c}_a{a}_obj{obj}/` (e.g. `output/decompose/c05/14_topk64_s1_c0.5_a0.7_obj0.5/`) with `input_*.jpeg`, `overview.jpeg`, `embeds.pt`, `groups.json`, plus `output/decompose/<run>/summary.json`. For one source image there are therefore up to 9 candidate input sets; comparing the `overview.jpeg` of the same `<stem>` across runs is the quickest visual check. Earlier, the good folders of all runs were copied by hand (flat, the folder names differ between settings) to `data/no_main/`, the unspecified pool of `build_dataset.py`; folders copied this way stay there and are never touched by the export of `score_sweep.py`. The `restore` and `n2` runs have the same folder names as `base_s1`, so only one of these three versions of a source can be kept in `data/no_main/`.

Notes:
- The subject subspaces are cached in `text_cache/` and shared by all runs (the prior settings are the same), so only the first run pays for them.
- Rerunning a command skips images that are already done, so an interrupted run can simply be restarted (in `tmux`, or with `nohup ... > $OUT/<run>.log 2>&1 &`).
- Runs `base_s2` and `base_s3` produce exactly the same `groups.json` as `base_s1`; only the images differ.

Fields that help with selection (`summary.json` per image, details in `groups.json`):

| Field | Use |
|---|---|
| `decomposable` | `false` → no inputs; skip |
| `n_main` | Number of inputs (2–4) |
| `mean_input_input_cos` | Lower = inputs more distinct; very high values mean near-duplicate inputs |
| `mean_cos_to_e` | Too low = inputs drift away from the source |
| `max_cos_to_bias` | High = some input is dominated by the SAE bias (risk of unrelated animals) |
| `subject`, `subject.margin_top1_top2` | Small margin = uncertain subject; check visually |
| `subject.subject_strength_inputs_after[*].proj_ratio` vs. `subject_strength_source.proj_ratio` | The keeper should be close to the source value, the others clearly lower |

These fields are not used by `score_sweep.py` (it only reads `image`, `decomposable`, `top_k`, `clean_frac` and `seed` from `groups.json`); they remain useful for looking into single folders.

### Parameters

Defaults are the values in the code (updated 2026-11-14: `--clean-frac 0.3`, `--alpha-r 0.7`).

**Input / output**

| Argument | Default | Description |
|---|---|---|
| `--image` / `--image-dir` | (one required) | One image, or a folder (`.jpg .jpeg .png .webp .bmp`) |
| `--output-dir` | `output/decompose/v72` | Output root; one subfolder per image |
| `--overwrite` | off | Recompute folders that already contain `groups.json` |
| `--debug-images` | off | Also decode diagnostic images (see *Output*); several extra decodes per image |

**Grouping and roles**

| Argument | Default | Description |
|---|---|---|
| `--top-k` | 64 | Number of most activated SAE features that are grouped |
| `--n-max` | 4 | Maximum number of main groups (= maximum number of inputs) |
| `--min-salience` | 0.1 | Minimum salience of a main group |
| `--delta-mode` | `quantile` | `quantile` or `max` (ConceptExpress original; merges everything with SAE features) |
| `--delta-quantile`, `--delta-cap` | 0.2, 0.9 | Quantile of pairwise distances used as $\delta$, and its upper limit |
| `--delta-fallback`, `--chain-factor` | none, 1.0 | Fallback $\delta$; factor on the mean-spread check |

**Cleaning, coverage, merging**

| Argument | Default | Description |
|---|---|---|
| `--clean-mode` | `fraction` | `fraction` or `margin` |
| `--clean-frac` | 0.3 | Fraction of each main group removed (closest to another main group); 0 = no cleaning. Higher = purer groups but more content moved to the shared part or lost |
| `--clean-margin` | 0.1 | Threshold for `--clean-mode margin` |
| `--assign-threshold` | 0.2 | Coverage expansion: minimum cosine to assign a remaining feature to a main group (1 = off) |
| `--merge-mode` | `absolute` | `absolute` (input cosine ≥ threshold), `relative`, or `off` |
| `--merge-threshold` | 0.85 | Input cosine above which two groups are merged |
| `--merge-margin` | 0.0 | Margin for `--merge-mode relative` |
| `--n-min` | 2 | Merging never goes below this number of main groups |

**Input construction**

| Argument | Default | Description |
|---|---|---|
| `--gamma` | 1.0 | Amplification of the own group: $(1+\gamma) D_i$ |
| `--beta-minor` | 1.0 | Fraction of the minor groups removed from every input |
| `--beta-amb` | 1.0 | Fraction of the ambiguous (cleaned) features removed from every input |
| `--alpha-r` | 0.7 | Fraction of the shared part $S$ kept in every input (1 = not weakened); 0.5 gave more artifacts |
| `--renorm` | `auto` | Rescale inputs to $\lVert e\rVert$; `auto` = when $\alpha_r \ne 1$ or the subject is weakened |

**Subject**

| Argument | Default | Description |
|---|---|---|
| `--subject-mode` | `auto` | `auto`: subject keeper (steps 9–11); `off`: skip steps 9–11 (= v6 construction) |
| `--noun-vocab` | built-in | Text file with one noun per line |
| `--subject-map` | none | JSON `{"image stem or file name": "noun"}` overriding the retrieval |
| `--subject-rank` | 4 | Dimension of the subject subspace. Lower (2) if removing the subject causes artifacts; higher (8) if it does not remove the subject |
| `--alpha-obj` | 0.5 | Fraction of the subject component kept in the non-keeper inputs (1 = no weakening, 0 = full removal) |
| `--subject-restore` | off | Reset the keeper's subject component to the source level |
| `--prior-samples`, `--prior-steps`, `--prior-guidance`, `--prior-seed` | 2, 25, 4.0, 0 | Prior sampling for the subject subspace (changing them recomputes the cache) |
| `--text-cache` | `text_cache` | Cache for noun embeddings and subject subspaces |

**Generation**

| Argument | Default | Description |
|---|---|---|
| `--seed` | 1 | Kandinsky decoder seed, shared by all inputs. Changes only the rendering (layout, pose, details), not the embeddings or `groups.json`; weak content (small objects, weak groups) may appear for one seed and not another |
| `--steps` | 50 | Kandinsky decoder steps |
| `--size` | 768 | Output resolution. 768 gives better decompositions than 512; for training the inputs are downscaled to 512 when they are cached (`cache_features.py --ref-size`) |

### Output

One folder per image: `<output-dir>/{image stem}_topk{top-k}_s{seed}_c{clean-frac}_a{alpha-r}_obj{alpha-obj}/`, with numbers in their shortest form (`--alpha-r 1.0` → `a1`). `--n-max`, `--subject-restore` and the other parameters are not in the name. An existing folder with the same name is skipped unless `--overwrite` is given. Downstream scripts take the source stem as the part before `_topk`, so they accept both this naming and the older `{stem}_topk64_nmax4` folders.

| File | Content |
|---|---|
| `recon.jpeg` | Kandinsky reconstruction of $e$ |
| `input_0.jpeg` … | One input per main group, ordered by salience |
| `overview.jpeg` | source, recon, inputs; captions `g{i} n=kept/before cleaning +assigned [noun]` (the noun marks the keeper) and salience |
| `embeds.pt` | `e`, top features, main groups, `D_main`, `D_core`, assigned features, `R`, `D_minor`, `D_amb`, final `e_inputs` |
| `groups.json` | All parameters; groups (features, activations, salience, cleaning), merges, coverage, per-input conditions (`cos_to_e`, norms, `cos_to_bias`), input–input cosines, `shared_weakening` (split of the shared part into unassigned features / bias / SAE error), `subject` (see below) |
| with `--debug-images` | `debug_R.jpeg` (shared part alone), `debug_bias.jpeg` (SAE bias alone), `input_{i}_unique.jpeg` (input without the component common to all groups), `debug_source_no_subject.jpeg` and `debug_R_no_subject.jpeg` (subject subspace fully removed), `debug_overview.jpeg` |

`groups.json → subject`: `noun`, `source` (`retrieval` / `subject_map`), `top_candidates` (top 5 with cosine), `margin_top1_top2` (small = uncertain retrieval), `singular_values`, `energy_top_r` (share of the $\Delta_k$ energy in the top r vectors), `subject_strength_source`, `subject_strength_inputs_before` / `_after` (each with `cos_text`, `cos_subject_dir` and `proj_ratio`), `keeper_group`, `subject_restore`.

`<output-dir>/summary.json`: one line per image (number of main groups before/after merging, mean input–input cosine, mean cosine to $e$, subject and keeper, maximum cosine to the bias, …). The terminal prints, per image, the subject, the keeper, the per-input `proj_ratio` before → after, and the grouping summary.

### How to read the results

- **Subject correct?** Check `noun` and `margin_top1_top2`; fix wrong ones with `--subject-map`.
- **Does the subspace hit the subject?** `subject_strength_source.proj_ratio` should be clearly larger than with text (≈ 0.1–0.17), and the subject should disappear in `debug_source_no_subject.jpeg`.
- **Subject lost?** If every `proj_ratio` *before* weakening is far below the source value, the subject was already lost by cleaning or minor/ambiguous removal: try `--subject-restore` or a lower `--clean-frac`.
- **Unrelated animals** in an input: that input is dominated by the SAE bias (`cos_to_bias` high, weak own group).
- **Stability:** rerun with another `--seed`; inputs whose content changes between seeds come from weak groups.

### Version history

| Version | Change |
|---|---|
| `decompose_original.py` | Paper: k-means into 2 clusters, cleaning, $e \pm \lambda(d_1 - d_0)$ |
| `decompose.py` | K clusters, one-vs-rest directions (fixed-K baseline) |
| `decompose_groups.py` v1 | Adaptive grouping (FINCH + ConceptExpress threshold) |
| v2 | Quantile threshold capped at 0.9 (max-based threshold merged everything) |
| v3 | Main/minor groups, cleaning, "remove other groups" inputs ($\beta$, $\gamma$), 50 steps / 768 px |
| `decompose_v4.py` | Coverage expansion (`--assign-threshold`) |
| `decompose_v5.py` | Input-based merging, `--debug-images` |
| `decompose_v6.py` | Weakening of the whole shared part (`--alpha-r`) |
| `decompose_v7.py` | Subject subspace from CLIP text deltas, separate subject input |
| `decompose_v7_1.py` | Subject subspace from prior-predicted image embeddings |
| **`decompose_v7_2.py`** | Subject kept in one input (keeper), weakened in the others; separate subject input removed |

---

## Sweep and automatic scoring (`run_sweep.sh`, `score_sweep.py`)

### Goal

Every source image is decomposed with several settings, and the right setting depends on the image. Instead of comparing the folders by eye and copying the good ones by hand, `score_sweep.py` checks what the inputs of every folder actually show and picks, **per source image**:

- a **no-main pick** (level 1): all inputs are different, come from the source and each clearly shows its own concept. Used as an unspecified sample (target = the source image).
- a **main pick** (level 2): a level-1 folder in which at least two inputs each show a different, prominent object. Used as a synthesis source for designated-subject targets (replaces `screen_sources.py` and the manual move to `generate_source/`).

`decompose_v7_2.py` is not changed. The picks of one image never depend on other images.

### Sweep (`run_sweep.sh`)

```bash
./run_sweep.sh samples/sources/14.jpeg      # one image
./run_sweep.sh samples/sources              # every image in the folder
TOPK=64 CLEAN="0.3 0.7" SEEDS=1 ./run_sweep.sh samples/sources/14.jpeg   # quick test
```

| Variable | Default | Meaning |
|---|---|---|
| `OUT` | `output/decompose/sweep` | Sweep root |
| `TOPK` | `"32 64"` | `--top-k` values |
| `CLEAN` | `"0.3 0.5 0.7"` | `--clean-frac` values |
| `SEEDS` | `"0 1"` | `--seed` values |
| `DECOMPOSE_ARGS` | – | Further `decompose_v7_2.py` options for every run, e.g. `"--steps 25"` |

The defaults give 2 × 3 × 2 = 12 folders per source image in `$OUT/topk<K>_c<clean>_s<seed>/`, logs in `$OUT/_logs/`. Existing folders are skipped by `decompose_v7_2.py`, so an interrupted sweep continues where it stopped. At the end `score_sweep.py` runs on the images of the sweep **without export**.

`--seed` only changes the Kandinsky decoding: the SAE grouping (`groups.json`) depends on `--top-k` and `--clean-frac` only, so the seed axis gives another rendering of the same decomposition (the default grid has 6 different groupings per image).

### Checks

**Cheap checks (every folder)**, with the CLIP image encoder of the Kandinsky 2.2 prior and DINOv2 (`measure.json`):

| Label | Condition |
|---|---|
| `unrelated` | `rel_clip` = cos(input, source) < `--min-rel` |
| `source_copy` | `rel_clip` > `--max-rel` |
| `duplicate` | two inputs look the same: `pair_clip` ≥ `--dup-clip` or `pair_dino` ≥ `--dup-dino`; the input less related to the source is marked |

**VLM checks** (Qwen3-VL-8B-Instruct, only on folders that pass the cheap checks):

1. **Source inventory** (once per source image, `_sources/<stem>.json`): the VLM lists at most `--max-concepts` visible concepts of the source, each ONE thing (`boy` and `kite`, never `boy flying a kite`), with a type (`object`, `material`, `color`, `shape`, `pattern`, `style`, `scene`). The SAE groups have no names, so this inventory is the vocabulary of the checks.
2. **Main-choice table** (per folder, `main_choice.json`): one multiple-choice question per input, *"Which ONE of these is the main subject of this image? 0 = none of these, 1 = boy, 2 = kite, …"*. The probabilities of the answer digits are read from one forward pass (no text is generated), so every input gets a distribution over the concepts. A yes/no question ("is X shown?") is answered yes for anything visible and saturates. The digits limit the inventory to 9 concepts.

The **own concept** of an input is the most likely answer, `own_p` its probability. Further labels:

| Label | Condition |
|---|---|
| `not_in_source` | "none of these" is the most likely answer (broken image or content not in the source) |
| `unclear` | `own_p` < `--min-own` |
| `shared_concept` | two inputs have the same own concept (the one with the lower `own_p` is marked), or the own concept of another input is an **object** and gets a main-subject probability ≥ `--leak` in this input (it competes as the subject). Scenes, colors, styles etc. are not checked for leaks: a meadow behind every input is fine |

An input gets the first matching label (order of the two tables), otherwise `unique`.

- **Level 1**: `no_main_ok` = decomposable, ≥ 2 inputs, all inputs `unique`.
- **Level 2**: `main_ok` = `no_main_ok` and `n_objects` ≥ `--min-objects`, where `n_objects` is the number of different kinds among the unique inputs whose own concept is an `object` with `own_p` ≥ `--prominent`. Two objects are the same kind if the VLM (text only) answers *"Are a and b the same kind of object?"* with P(yes) ≥ 0.5 (bird / parrot → same; cached in `_sources/same_kind.json`).

### Score and picks

$$\text{score} = \sum_{\text{unique inputs}} \text{own\_p} \;+\; \bigl(1 - \text{mean pair\_clip between inputs}\bigr)$$

- **No-main pick**: best-scoring `no_main_ok` folder of the image; **main pick**: best-scoring `main_ok` folder. They can be the same folder.
- **Cascade**: before the VLM, `own_p` is counted as 1, so this *cheap score* is an upper bound of the final score. The VLM checks the folders that pass the cheap checks from the best cheap score down and skips a folder once its cheap score cannot beat the picks already found; at most `--vlm-top` folders per image (0 = no limit). Skipped folders are marked `not checked` and cannot be picked.
- **Overrides**: `<sweep-root>/overrides.csv` with columns `stem,no_main,main` replaces the automatic picks of an image. A value is a folder name as in `folders.csv` `name` (e.g. `topk64_c0.3_s0/14_topk64_s0_c0.3_a0.7_obj0.5`), `none` for no pick, empty for automatic. Rerun the scorer to apply it.
- With `--no-vlm` only the cheap checks run: `no_main_ok` then means "passes the cheap checks", and there are no main picks.

### Usage

```bash
python score_sweep.py                                   # all folders under output/decompose/sweep
python score_sweep.py --image samples/sources/14.jpeg   # only the folders of one source image
python score_sweep.py --image-dir samples/sources       # the folders of the images in a folder
python score_sweep.py --leak 0.7 --min-objects 3        # other thresholds (cached, takes seconds)
python score_sweep.py --export-no-main data/no_main --export-main generate_source
```

Every measurement and VLM answer is cached (`measure.json`, `main_choice.json`, `_sources/`); changing a threshold and rerunning takes seconds (the VLM may be asked about a few more folders if the picks change). `--remeasure` recomputes the CLIP / DINOv2 measurements, `--reask` asks the VLM again. Cached VLM answers are also asked again when `--vlm-model` or the prompts (`PROMPT_VERSION` in the script) change.

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `--sweep-root` | `output/decompose/sweep` | Every folder with a `groups.json` below it is scored (except `_review/`, `_logs/`, `_sources/`); source image and settings are read from `groups.json`, never from folder names |
| `--image` / `--image-dir` | all | Only the folders of these source images (by stem) |
| `--export-no-main` | – | Sync no-main **and** main picks to this folder (`data/no_main`) |
| `--export-main` | – | Sync main picks to this folder (`generate_source`) |
| `--no-vlm` | off | Cheap checks only |
| `--vlm-top` | 4 | VLM-check at most this many folders per image (0 = all) |
| `--vlm-model` | `Qwen/Qwen3-VL-8B-Instruct` | VLM |
| `--vlm-size` | 448 | Images are resized to this size for the VLM |
| `--max-concepts` | 8 | Concepts in a source inventory (≤ 9) |
| `--dino-model` | `facebook/dinov2-base` | DINOv2 model of the duplicate check |
| `--min-rel` | 0.4 | `unrelated` if `rel_clip` < this |
| `--max-rel` | 0.92 | `source_copy` if `rel_clip` > this |
| `--dup-clip` | 0.8 | duplicate pair if `pair_clip` ≥ this |
| `--dup-dino` | 0.75 | duplicate pair if `pair_dino` ≥ this |
| `--min-own` | 0.4 | `unclear` if `own_p` < this |
| `--leak` | 0.3 | `shared_concept` if another input's own object gets a main-subject probability ≥ this |
| `--prominent` | 0.5 | an object counts for level 2 if its `own_p` ≥ this |
| `--min-objects` | 2 | `main_ok` needs this many different prominent objects |
| `--overview-rows` | 20 | Rows per overview page |
| `--remeasure`, `--reask` | off | Ignore the cached measurements / VLM answers |

### Output

| Path | Content |
|---|---|
| `<folder>/measure.json` | `rel_clip`, `rel_dino`, `pair_clip`, `pair_dino` |
| `<folder>/main_choice.json` | Concept names and the main-choice probabilities of every input |
| `<folder>/score.json` | Labels, own concepts, flags (`no_main_ok`, `main_ok`, `n_objects`), score, picks and reason of the last scoring |
| `_sources/<stem>.json` | Source inventory (raw VLM answers included) |
| `_review/<stem>.jpeg` | Every folder of one source image, picks first: verdict, score, reason, then source, `recon.jpeg` and the inputs with own concept, `own_p`, label, type, "none" probability, `rel_clip` / `rel_dino`. Redrawn for the images scored in a run |
| `_review/overview_XX.jpeg` | One row per pick of every image (an image without a pick: its best folder, in red): the whole dataset at a glance |
| `_review/folders.csv`, `inputs.csv` | One line per folder / per input of every scored folder |
| `_review/summary.json` | Counts per number of inputs and per setting, label rates, images without a no-main / main pick, picked object names, thresholds of the last run |

The review covers every folder that has a `score.json` of the current script version, so scoring one image or a batch adds to it. `score.json` files of an older version are left out (the script prints which images to rescore).

### Export

`--export-no-main` and `--export-main` **sync** the target folder with the picks of the images scored in this run:

- new picks are copied unchanged, under their own folder name, and listed in `<target>/_exported.json`;
- a folder this script exported earlier for one of these images that is **no longer a pick** is **deleted**;
- folders not listed in `_exported.json` (e.g. copied by hand earlier) are never touched.

Main picks also go to `data/no_main/`, so that `build_dataset.py` finds byte-identical inputs for the designated samples synthesized on `generate_source/`.

> **Freeze the picks before synthesis.** The review folders of `select_targets.py` refer to the inputs in `generate_source/<folder>/`. If a later rescoring (other thresholds, more seeds) changes a main pick and is exported, the old folder is deleted from `generate_source/` and `data/no_main/`, and the designated samples synthesized on it lose their inputs. Once `synthesize_v2_1.py` has run on a pick, keep it with `overrides.csv`, or export to new folders.

---

## Designated-subject target synthesis (`synthesize_v2_1.py`)

### Goal

The per-image weight levels (`high` / `medium` / `low`) of the earlier `synthesize_target.py` are replaced by an **optional designated main subject**: the user either designates one input as the main subject, or designates none. The training data therefore contains two kinds of samples:

| Kind | Inputs | Designation | Target |
|---|---|---|---|
| Unspecified | decomposed inputs | none | the real source image |
| Designated | decomposed inputs | input $k$ | a synthesized image pulled toward input $k$ (this script) |

Designating different inputs of the same input set gives different targets, i.e. **counterfactual samples**: the model has to read the designation instead of inferring the main subject from the images. The designated input should dominate the target and keep its rough shape and pose, while the other inputs still contribute; its exact shape does not have to be preserved (a designed sofa need not match the input sofa), and its category does not have to be preserved either.

### Method

Input: decomposition folders written by `decompose_v7_2.py` (2–4 inputs per folder): `embeds.pt` (`e`, `e_inputs`), `input_0.jpeg` …, `recon.jpeg`, and optionally `groups.json` (subject and keeper, only used for display). $f(\cdot)$ is the CLIP image encoder of the Kandinsky 2.2 prior.

1. **Closeness of every input to the source** (before any generation):
   $$\text{closeness}_k = \cos\big(f(\texttt{input\_k.jpeg}),\ f(\texttt{recon.jpeg})\big)$$
   This is an image-level measure. The embedding-level cosine $\cos(e, e_k)$ is not used because it does not separate the inputs: in one folder the three inputs all had $\cos(e, e_k) \approx 0.84$ but image-level similarities of 0.37, 0.76 and 0.47. If `recon.jpeg` is missing, it is decoded from $e$ and saved in the output folder.
2. **Class**: input $k$ is **close** if $\text{closeness}_k \ge$ `--close-threshold` (0.6), otherwise **far**.
3. **Target embedding** for main input $k$ (one knob $t$, no per-image levels):
   $$e_T = (1 - t)\, e + t\, e_k, \qquad \text{rescaled to } \lVert e\rVert$$
   $t = 0$ is the source embedding, $t = 1$ is input $k$ itself. Moving toward $e_k$ amplifies group $k$ and attenuates the other groups.
4. **Decoding with Kandinsky 2.2 img2img** at `--size` (1024 px, the size of the real source images that are the targets of unspecified samples): init image = `input_k.jpeg` (768 px, upsampled to 1024), condition = $e_T$. The strength $s$ is the diffusers img2img strength, i.e. the fraction of the noise schedule applied to the init image; **larger $s$ = further from the main input**. About 0.4 keeps outline, details and colors; 0.7 keeps the rough layout and proportions; 0.9 almost ignores the init image. The number of denoising steps actually run is `int(steps * s)`. The img2img pipeline reuses the decoder's unet, scheduler and movq, so no extra weights are loaded.
5. **2 × 2 grid per main, chosen by class** (same seed for every cell):

   | Class | $t$ | $s$ | Reason |
   |---|---|---|---|
   | close | 0, 0.3 | 0.7, 0.9 | $t = 0$ with a large $s$ already gives a target close to the reconstruction |
   | far | 0.3, 0.6 | 0.4, 0.7 | $t = 0$ has no visible effect at $s \ge 0.7$; a smaller $s$ is needed to keep the main at all |

6. **Measurements** on re-encoded images (recorded, not used as a filter yet), for target $T$ and main $k$:
   - `cos_main` $= \cos(f(T), f(\text{input}_k))$; close to 1 = the target is a copy of the main input;
   - `lift` $= \text{cos\_main} - \text{closeness}_k$: how much more the target resembles the main than the plain reconstruction does; about 0 = the designation had no effect;
   - `margin` $= \text{cos\_main} - \max_{i \ne k} \cos(f(T), f(\text{input}_i))$; below 0 = the target still resembles another input more than the main.

### How the grids were chosen

`synthesize_v2.py` ran the full 3 × 3 grid ($t \in \{0, 0.3, 0.6\}$, $s \in \{0.4, 0.7, 0.9\}$) on 14 folders (41 mains, 768 px targets at the time); the best cell was picked by hand for 7 mains.

- The input closest to the source ("dominant" input) had a closeness of 0.61–0.85, the other inputs 0.26–0.68. The dominant input was the v7.2 keeper in 11 of 14 folders.
- **Close mains**: hand-picked cells (0, 0.9), (0, 0.9), (0.3, 0.7). Their `lift` was only +0.01 to +0.04, i.e. the target is close to the reconstruction.
- **Far mains**: hand-picked cells (0.6, 0.7), (0.3, 0.7), (0.6, 0.9), (0.6, 0.4). At $t = 0$, $s \ge 0.7$ the median `lift` was about 0.
- Hand-picked cells had `cos_main` 0.52–0.78 and `margin` down to −0.15. Using "`cos_main` ≤ 0.8 and `margin` ≥ −0.15" as a criterion, 26 of 27 far mains have at least one usable cell with $s \in \{0.4, 0.7\}$, the same as with the full 3 × 3 grid; with $s \in \{0.7, 0.9\}$ only 21 of 27.
- The default grids contain 6 of the 7 hand-picked cells; the missing one, (0.6, 0.9), had almost the same values as (0.6, 0.7).

In this analysis the closeness was approximated from `synthesize_v2.py` outputs; `synthesize_v2_1.py` computes it directly, so the threshold may need a small adjustment after the first run.

### Usage

```bash
# default: class-dependent 2 x 2 grid for every input of every folder, 1024 px targets
python synthesize_v2_1.py --root generate_source --output-dir data/synth

# try the first 3 folders only
python synthesize_v2_1.py --limit 3

# other threshold, or an intermediate strength for far mains
python synthesize_v2_1.py --close-threshold 0.65
python synthesize_v2_1.py --far-s 0.4,0.55,0.7

# only inputs 0 and 2 as main subject, with a reference column without init image
python synthesize_v2_1.py --mains 0,2 --embed-only

# treat every input as far (threshold above 1) or as close (threshold 0)
python synthesize_v2_1.py --close-threshold 1.1
python synthesize_v2_1.py --close-threshold 0
```

Folders that already contain `synth_meta.json` are skipped unless `--overwrite` is given. At the end, the terminal prints the number of targets, how many mains were close / far, the min / median / max closeness, and in how many folders the dominant input is the keeper.

### Parameters

| Argument | Default | Description |
|---|---|---|
| `--root` | `generate_source` | Directory with decomposition folders (searched recursively for `embeds.pt` + `input_0.jpeg`), or one such folder |
| `--exclude` | `screening` | Comma-separated directory names under `--root` that are skipped (the unreviewed candidates of the earlier `screen_sources.py`; not used by `score_sweep.py`) |
| `--output-dir` | `data/synth` | Output root; the folder structure under `--root` is kept |
| `--close-threshold` | 0.6 | An input is close if $\cos(f(\text{input}), f(\text{recon})) \ge$ this value; 0 = every input close, > 1 = every input far |
| `--close-t` | `0,0.3` | Interpolation weights $t$ for close mains |
| `--close-s` | `0.7,0.9` | img2img strengths for close mains |
| `--far-t` | `0.3,0.6` | Interpolation weights $t$ for far mains |
| `--far-s` | `0.4,0.7` | img2img strengths for far mains |
| `--mains` | all | Comma-separated input indices to designate as main |
| `--embed-only` | off | Add a reference column decoded from $e_T$ alone (no init image) |
| `--seed` | 1 | Kandinsky seed, shared by every cell |
| `--steps` | 50 | Kandinsky steps; img2img runs `int(steps * s)` of them. Use the value of the decomposition run |
| `--size` | 1024 | Target resolution (1024 = size of the real source images); the 768 px inputs are upsampled to it as img2img init images |
| `--guidance-scale` | 4.0 | Decoder guidance scale |
| `--tile` | 256 | Tile size in the overview image |
| `--limit` | none | Only process the first N folders |
| `--overwrite` | off | Recompute folders that already contain `synth_meta.json` |

The grid lists can have any length (e.g. `--far-s 0.4,0.55,0.7`). Strengths must be in (0, 1] and give at least one denoising step.

### Output

One folder per decomposition folder: `<output-dir>/<name>/`.

| File | Content |
|---|---|
| `overview.jpeg` | First row: recon and all inputs, each with its closeness and class (the keeper is marked with its noun). Then one block per main: one row per $t$ (first cell = the main input with its class), one column per $s$, each cell captioned with `main`, `lift` and `margin` |
| `main{k}/t{t}_s{s}.jpeg` | Individual targets (`t{t}_noinit.jpeg` with `--embed-only`) |
| `synth_meta.json` | Threshold and grids; per input: closeness, class, $\cos(e, e_k)$, `cos_to_bias`; dominant input, subject and keeper; per main and cell: $t$, $s$, denoising steps, $\cos(e_T, e)$, $\cos(e_T, e_k)$, `cos_main`, `lift`, `margin`, cosine to every input and to recon, `shift_toward` |
| `recon.jpeg` | Only if the decomposition folder had none (decoded from $e$) |

`<output-dir>/summary.json`: one entry per processed folder (closeness and class per input, dominant input, keeper, number of targets).

### How to read the results

- **Copy of the main input**: `cos_main` above about 0.8; most likely at $(t, s) = (0.6, 0.4)$ (26 % of far mains in the 3 × 3 analysis).
- **Designation without effect**: `lift` about 0, typical for $t = 0$ with a large $s$ on far mains.
- **Main not dominant**: `margin` clearly below 0; the target still resembles another input more.
- **Abrupt change between strengths**: some mains are near copies at $s = 0.4$ and lose the main at $s = 0.7$; an intermediate strength (`--far-s 0.4,0.55,0.7`) may help.
- **Unusable mains**: in one folder, every cell of the 3 × 3 grid resembled another input much more than the main (`margin` −0.28 to −0.39); such mains should be dropped.
- `cos_to_bias` from the decomposition does **not** reliably identify inputs dominated by the SAE bias (an unrelated animal had 0.36, a clean input of the same folder 0.57).

### Version history

| Script | Change |
|---|---|
| `synthesize_target.py` | Per-image levels (`low` / `medium` / `high` = 1 / 3 / 9), $c_i = w_i / m_i$, $e_{target} = r + \sum_i c_i d_i$, rank check; only for `decompose.py` outputs |
| `synthesize_v2.py` | Designated main subject: $e_T = (1-t)\,e + t\,e_k$ decoded by img2img from the main input; full 3 × 3 grid ($t \in \{0, 0.3, 0.6\}$, $s \in \{0.4, 0.7, 0.9\}$) for every main; reads `decompose_v7_2.py` outputs |
| **`synthesize_v2_1.py`** | Closeness to the source decides a 2 × 2 grid per main (close / far); `lift` and `margin` replace the averaged shift |

---

## Target selection (`select_targets.py`)

### Principle

The script only **suggests** which synthesized cells to keep; the final choice is made by hand. The global CLIP measurements can filter out clear failures, but they cannot tell which of the remaining cells is best (e.g. $s = 0.7$ vs $0.9$), so every cell left in `keep/` is used as a target and training picks one of them at random.

Every cell $(t, s)$ of every main $k$ is checked:

| Reason | Condition (default) | Meaning |
|---|---|---|
| `copy` | `cos_main` > `--copy-max` (0.80) | near copy of the main input |
| `no_effect` | `lift` < `--lift-min` (0.0) | not closer to the main than the plain reconstruction |
| `not_dominant` | `margin` < `--margin-min` (−0.15) | still resembles another input more than the main |
| `not_distinct` | `distinct` < `--distinct-min` (0.03) | cannot be told apart from the targets of the other mains of the folder |

$$\text{distinct} = \cos(f(T), f(\text{input}_k)) - \max_{j \ne k}\ \overline{\cos(f(T_j), f(\text{input}_k))}$$

where the mean runs over the cells of main $j$ that pass the first three checks (the check is skipped if no other main has such a cell). The thresholds were set on 14 folders with 768 px targets (all 6 hand-picked cells inside the v2.1 grids pass) and were not recalibrated for 1024 px.

### Review folders

```
data/selection/
    review/<folder>/
        _overview.jpeg          everything needed to decide, in one image
        _inputs.jpeg            all inputs, labelled input_0, input_1, ...
        info.json               decomposition folder, inputs, original path of every cell (do not edit)
        main<k>/
            _main.jpeg          the main input k
            keep/t0.30_s0.40.jpeg          suggested to keep
            drop/t0.60_s0.40__copy.jpeg    suggested to drop (reasons after "__")
    review/_created.txt         folders created so far (do not edit)
    suggestions.csv             one row per cell: suggestion, reasons, metrics (rewritten on every run)
    summary.json                thresholds and counts
```

`_overview.jpeg`: the first row shows all inputs; then one row per main with the main input and every candidate cell, captioned with the cell name (= file name), `cos_main`, `lift`, `margin`, `distinct`, the suggestion (green frame = keep, red = drop with the reasons) and **`now`**, i.e. whether the cell is currently in `keep/`. Images added to `keep/` by hand are appended with a blue frame.

### Manual selection

| To | Do |
|---|---|
| keep a cell | have its file in `keep/` (moving it from `drop/` without renaming is fine; the `__reason` suffix is ignored) |
| drop a cell | move it out of `keep/` (or delete it) |
| drop a whole main | empty its `keep/`, or delete `main<k>/` |
| drop a whole folder | delete `review/<folder>/` |
| add another image | copy it into `keep/` (e.g. a cell from another synthesis run); it must be generated from the same inputs |

Nothing has to be recorded elsewhere: `build_dataset.py` reads the content of the `keep/` folders when it runs, and uses those files **as they are** as targets. The review folder is therefore part of the dataset.

Rerunning `select_targets.py` never changes `keep/` / `drop/` of an existing folder and never recreates a deleted one (listed in `_created.txt`); it only redraws `_overview.jpeg` (so `now` reflects your changes) and adds new folders. `--overwrite` rebuilds everything from the suggestions and discards all manual changes.

### Selection criteria (designated targets)

A designated target teaches the model "input $k$ is the main object". It only does so if the target can be attributed to input $k$ **alone**. Keep a cell only if all three hold:

| Criterion | Meaning | Fails, for example |
|---|---|---|
| **① Recognizably the main input** | The main subject keeps the **visual identity** of input $k$ (shape, material, color, texture), not only its category | A bird-shaped statue becomes a real bird; a ceramic vase becomes a plain vase |
| **② Not explained by another input** | No other input of the folder explains the target as well, in particular when two inputs share a category | Two inputs are birds (a statue and a bird on a tower): a target with real birds fits both |
| **③ Not a copy** | The other inputs visibly contribute (background, material, color, elements); not the main input with a new background | The target is almost input $k$ itself |

A change of the subject is fine as long as ① and ② hold: a bird with the statue's material and form, on a pedestal, is a good target for the statue; two real birds are not. Also check, as before: a fused object rather than a collage, no structural artifacts, and targets of different mains of one folder that look different (otherwise they teach nothing about the designation). One or two good cells per main are better than four mediocre ones; if no cell of a main passes, drop the main (the folder keeps its other mains).

Why it matters: an ambiguous target is a noisy label that the model cannot resolve. In the overfitting tests (*Current findings*), the main whose kept target failed ① and ② (statue → real birds, with a second bird input in the folder) was the only one never learned: the model either copied the statue (the base model's default for that input) or produced the target of the other bird input; the three mains with clear targets were learned.

Automatic hints: `margin` (criterion ②, cosine to the main minus the largest cosine to another input) and `cos_main` (criterion ③) of `suggestions.csv`. The default `--margin-min` (−0.15) is lenient and global CLIP features capture the category more than the identity, so criterion ① can only be checked by eye; look at folders with two inputs of the same category first. Synthesis sources should have inputs that show different subjects: two inputs of one category make their targets ambiguous from the start. `score_sweep.py` checks this for the main picks (level 2: at least `--min-objects` prominent objects of different kinds, same-kind test by the VLM); when overriding a main pick by hand (`overrides.csv`, step 2 of the pipeline), follow the same rule.

### Usage and parameters

```bash
python select_targets.py --synth-root data/synth --output-dir data/selection
```

| Argument | Default | Description |
|---|---|---|
| `--synth-root` | `data/synth` | Output root of `synthesize_v2_1.py` (searched for `synth_meta.json`) |
| `--output-dir` | `data/selection` | Review folders, `suggestions.csv`, `summary.json` |
| `--copy-max`, `--lift-min`, `--margin-min`, `--distinct-min` | 0.80, 0.0, −0.15, 0.03 | Thresholds of the suggestions |
| `--overview-tile` | 288 | Image size in `_overview.jpeg` |
| `--overwrite` | off | Rebuild every review folder from the suggestions (discards manual changes and deletions) |

---

## Training data (`build_dataset.py`, `cache_features.py`)

### Sample pools

| Pool | Inputs | Main | Target | Source |
|---|---|---|---|---|
| unspecified | `input_*.jpeg` of a decomposition folder | none | the original source image (path from `groups.json` → `image`, else `<stem>.*` in `--source-dir`) | every folder of `--decomp-root` (`data/no_main`), one sample each; `recon.jpeg` is not used |
| designated | the same inputs | input $k$ | the files in `review/<folder>/main<k>/keep/`, one picked at random per draw | the review folders of `select_targets.py` |

- **Input consistency.** The designated targets were synthesized on `generate_source/`, a copy of part of `data/no_main` (main picks of `score_sweep.py`, which are exported to both folders; earlier: `screen_sources.py` + manual review). For every review folder the inputs are compared byte by byte with the folder of the same name in `--decomp-root`; if identical, the `data/no_main` paths are used (both pools share the cached latents), otherwise the inputs of the synthesis are kept and the folder is reported, because a target must come with the inputs it was synthesized from.
- **Split by source image**, stratified: `--val-frac` (10 %) of the sources that have designated samples and 10 % of the others go to validation, so validation contains both pools.
- **No ratio in the manifests.** The share of the two pools is decided during training by the pool schedule (see below); the manifests only record the pool of every sample, and the input order is shuffled at training time.

Output (`--output-dir`, default `data/multiref`): `train.jsonl`, `val.jsonl` (one sample per line: `id`, `stem`, `pool`, `folder`, `n_inputs`, `inputs`, `main` (0-based or null), `targets`, `cells`) and `stats.json` (counts per split, pool and number of inputs; review folders whose inputs differ from `--decomp-root`).

### Cache

`cache_features.py` encodes every input at `--ref-size` (512 → latent [16, 64, 64]) and every target at `--target-size` (1024 → [16, 128, 128]) with the FLUX VAE (mode of the latent distribution, normalized like the pipeline; non-square images are center-cropped), and every prompt template with the T5 and CLIP text encoders (`max_sequence_length` 512). Files: `<cache-dir>/latents/<key>.pt`, `prompts.pt`, `config.json`. Existing latents are skipped, so the script can be rerun after the manifests change.

**After a change in `keep/`, rerun `build_dataset.py` and then `cache_features.py`**: newly kept files are new paths and must be cached before training.

### Usage and parameters

```bash
python build_dataset.py --decomp-root data/no_main \
    --review-dir data/selection/review --source-dir samples/sources --output-dir data/multiref
python cache_features.py --manifest-dir data/multiref --target-size 1024
```

| Script | Argument | Default | Description |
|---|---|---|---|
| `build_dataset.py` | `--decomp-root` | `data/no_main` | Decomposition folders (unspecified pool) |
| | `--review-dir` | `data/selection/review` | Review folders (designated pool) |
| | `--source-dir` | `samples/sources` | Fallback location of the source images |
| | `--val-frac`, `--seed` | 0.1, 0 | Validation fraction of the source images, split seed |
| | `--min-inputs`, `--max-inputs` | 2, 4 | Allowed number of inputs |
| `cache_features.py` | `--manifest-dir`, `--cache-dir` | `data/multiref`, `<manifest-dir>/cache` | Manifests and cache location |
| | `--ref-size`, `--target-size` | 512, 1024 | Resolution of the references and of the targets |
| | `--base-model` | `black-forest-labs/FLUX.1-Kontext-dev` | VAE and text encoders |
| | `--overwrite` | off | Recompute existing latents |

---

## Multi-reference fine-tuning (`train_multiref.py`)

A new LoRA is trained on FLUX.1 Kontext dev (the paper's released LoRA is not used). The transformer and the diffusers pipeline are not modified; all multi-reference code is in `multiref/`.

### Model input

- **References.** Each reference is encoded separately (512 px → 32 × 32 = 1024 tokens). The groups are concatenated, contiguous and in model order, after the 4096 tokens of the image being generated. With 4 references the image sequence has 8192 tokens (the same number of reference tokens as the paper's 1024 px grid, without white padding).
- **Position ids** $(t, h, w)$; the generated image has $t = 0$:
  - `canvas` (default): $t = 1$ for every reference, groups side by side along $w$ (a "virtual canvas"); the same ids as concatenating the latents along the width for the stock pipeline, only in contiguous per-reference order;
  - `per_t` (ablation): reference $k$ gets $t = k + 1$, with $h, w$ scaled so that every reference spans the whole grid of the generated image. The base model gave unstable results with it (`check_multiref.py`, check 5).
- **Main image** (`--role`):

  | Role mode | Order of the references | How the main image is marked | Prompt |
  |---|---|---|---|
  | **`tid` (default)** | shuffled | position id $t$ = `--main-t` (3) instead of 1 for every token of the main reference; $h, w$ and the order unchanged (`canvas` only) | "… Use the marked image as the main object …" |
  | `embed` | shuffled | learnable vector $E_{main}$ (3072-d, zero-initialized) added to every token of the main reference after `transformer.x_embedder` (forward hook) | "… Use the marked image as the main object …" |
  | `slot` | main first, others shuffled | position only | "… Use the first image as the main object …" |
  | `index` | shuffled | named in the prompt | "… Use image k as the main object …" |

  Without a main image, every mode uses the base prompt "Combine the elements in the reference images into a single object inspired by them." The generator only ever sees these fixed templates (`multiref/prompts.py`); free-form user text would be mapped to "main image k" or "none" beforehand. `tid` and `embed` share the same cached prompt.
- **Why `tid` is the default.** Kontext already uses the $t$ axis of its 3-axis RoPE to tell the generated image ($t = 0$) from the context ($t = 1$). Moving the main reference to $t$ = `--main-t` rotates its queries and keys in the $t$ frequency band, so it changes the attention between the generated image and the main reference in every layer, without extra parameters (unlike `embed`, whose $E_{main}$ stayed too small to be used). The order of the references and their $(h, w)$ ids do not change, so, unlike `slot`, no position becomes "the main position", and a sample without a main image gives exactly the same input as before. The base model does not react to the marker (designation gap ≈ 0 at step 0, baseline images of `check_multiref.py`), so the LoRA has to learn its meaning. `tid` needs the `canvas` layout (`per_t` already uses $t$ for every reference). See *Current findings* for the overfitting tests of the four modes.
- **Why $t = 3$ (default since 2026-11-22; $t = 2$ before).** $t$ is a continuous RoPE coordinate, not an index into an embedding table, so skipping $t = 2$ leaves nothing unused: attention only depends on the offset $\Delta t$ between tokens. The 16 dimensions of the $t$ band form 8 rotation pairs with frequencies $\theta_i = 10000^{-2i/16}$ (1, 0.32, 0.1, …). Relative to an ordinary reference ($t = 1$), the main reference is rotated by an extra $(t_{main} - 1)\,\theta_i$: 1 rad / 0.32 rad / 0.1 rad in the three highest pairs with $t = 2$, twice that with $t = 3$, so the marker perturbs the attention more and is easier to learn; the main reference is also 2 instead of 1 away from the other references. Both values are equally unseen in pre-training (Kontext only uses $t \in \{0, 1\}$). In the overfitting tests $t = 3$ gave a 2–3 times larger designation gap from step 2000 on, a high-$\sigma$ gap that kept growing instead of fading, and a "no main" output that is no longer one of the designated targets (*Current findings*). Larger values are not necessarily better: the highest pair has a period of $2\pi$, so beyond $\Delta t \approx 3$ it wraps around. Runs trained with another value must use the same `--main-t` at inference (`generate_multi.py` reads it from the run when `--main-t` is not given).
- **Pipeline.** `MultiRefKontextPipeline` overrides only `prepare_latents`, so that the injected reference tokens and ids are used instead of one encoded `image`. The output size must be passed together with `max_area = height × width` (the pipeline rescales the output to `max_area` otherwise); `generate_multi` does this.

### Training step

Batch 1, `--grad-accum` samples per optimizer step; as in the diffusers example `train_dreambooth_lora_flux_kontext.py` where possible.

1. **Pool**: unspecified with probability `--unspec-schedule(progress)`, else designated.
2. **Sample**: unspecified → uniform over folders; designated → uniform over folders, then mains, then kept targets (every source counts the same, however many mains and targets it has).
3. **Order** of the references shuffled (`slot`: main moved first); prompt template of the main position; cached latents and prompt embeddings.
4. **Noise**: $\sigma$ from the scheduler's training table (`--weighting-scheme none` = uniform, or `logit_normal`), or for designated samples from `--designated-sigma` (below); $x_\sigma = (1-\sigma)\,x_0 + \sigma\,\varepsilon$, packed target tokens.
5. **References** appended after the target tokens; the main reference gets $t$ = `--main-t` (`tid`) or $E_{main}$ (`embed`).
6. **Loss**: MSE between the transformer output of the target tokens and $\varepsilon - x_0$ (timestep $\sigma$, guidance embedding `--guidance-scale` 3.5).

**Pool schedule** (default `0:1.0,0.3:1.0,0.4:0.65`): unspecified samples only for the first 30 % of the steps (learn to combine the references), linear transition until 40 %, then 65 % unspecified / 35 % designated. A single value (`0.3`) is a constant probability.

**Noise level of designated samples.** Which input is the main image decides the composition, and the composition is decided at high noise; at low $\sigma$ the target is mostly visible in $x_\sigma$ and the main image hardly matters. `--designated-sigma logit_normal` therefore draws the $\sigma$ of designated samples directly as $\sigma = \mathrm{sigmoid}(\mathcal N(\mu, s))$ with `--designated-sigma-mean` $\mu$ = 1 and `--designated-sigma-std` $s$ = 1 (median $\sigma \approx 0.73$; $\mu > 0$ = more high-noise samples), clamped to the range of the sigma table. Unspecified samples keep `--weighting-scheme`; the loss weighting is unchanged. In the overfitting tests this learned the designation about twice as fast as the uniform density. Note that `--weighting-scheme logit_normal` indexes the *descending* sigma table, so there a positive `--logit-mean` favours **low** $\sigma$ (as in the diffusers example).

### Defaults

| Item | Default |
|---|---|
| Base model | `black-forest-labs/FLUX.1-Kontext-dev` transformer, bf16, gradient checkpointing |
| Main image | `--role tid`, `--main-t 3`, `--layout canvas` |
| LoRA | rank 32, alpha 32, on the Linear layers **inside** the transformer blocks: image and text attention, both feed-forwards of the double blocks, `proj_mlp` / `proj_out` of the single blocks (the final output projection, also named `proj_out`, is excluded); weights in fp32, forward in bf16 autocast |
| $E_{main}$ | only with `--role embed`: zero init, own parameter group |
| Optimizer | AdamW; LoRA lr 1e-4 (weight decay 1e-4), $E_{main}$ lr 1e-3 (no weight decay); 500 warmup steps, then constant; gradient clipping 1.0 |
| Noise | uniform $\sigma$ for both pools (`--weighting-scheme none`, `--designated-sigma same`) |
| Steps | `--max-steps` 10000 optimizer steps, `--grad-accum` 4 |

### Validation, checkpoints, logs

- **Validation loss** every `--val-every` steps on a fixed set of validation samples (fixed order, target, noise, $\sigma \in \{0.25, 0.5, 0.75\}$), reported per pool (`val_loss_unspecified`, `val_loss_designated`). Flow-matching losses decrease slowly even when samples are memorized: compare steps and pools, not absolute values.
- **Designation gap** every `--gap-every` steps (default = `--val-every`) and at step 0, on at most `--gap-samples` fixed designated validation samples (whole folders with the most mains first). For a sample with main $k$ and its target, with the same noise and target for every condition and for every $\sigma$ in `--gap-sigmas` (0.9, 0.75, 0.5, 0.25):

  | Logged value | Definition | Meaning |
  |---|---|---|
  | `gap` | $\overline{L(\text{main} = j)}_{j \ne k} - L(\text{main} = k)$ | > 0: the model uses the designation |
  | `gap_acc` | share of samples with $L(\text{main} = k) < \min_{j \ne k} L(\text{main} = j)$ | chance level: 1 / number of inputs |
  | `gap_none` | $L(\text{no main}) - L(\text{main} = k)$ | > 0: the designated target is predicted better with the designation than without |

  Each value is logged overall and per $\sigma$ (`gap_s0.9`, `gap_acc_s0.9`, …). A wrong main $j$ is designated in the same way as the right one ($t$ = `--main-t` on input $j$ for `tid`). **Look at the high-$\sigma$ values first**: they show whether the designation changes the composition; a gap only at low $\sigma$ means that the marker is used to recall details of memorized targets. At step 0 the gap should be ≈ 0 (exactly 0 for `embed`), which checks the metric itself.
- **Sample images** every `--image-every` steps and at step 0 (`runs/<run>/samples/step_XXXXXX.jpeg`): a few validation folders, preferably with ≥ 2 designated mains; per folder the inputs, the generated images (no main, then each main) and the targets. Only the VAE is loaded in addition (prompts come from the cache).
- **Checkpoints** every `--save-every` steps: `checkpoint-XXXXXX/lora/` (diffusers LoRA format), `role_embedding.safetensors` (only with `--role embed`), `train_state.pt` (resume with `--resume`). `args.json` in the run folder records all settings (read by `eval_multiref.py`).
- **Log** `log.jsonl` (optionally TensorBoard): loss per pool, gradient norm, pool probability `p_unspec`, share of designated samples, mean $\sigma$ per pool (`sigma_designated`, `sigma_unspecified`), learning rate, time per step, the validation losses and the gap values; with `--role embed` also `e_main_norm`, `ref_token_norm` and `e_main_ratio`. Training-loss lines are dominated by the random $\sigma$ and pool of each sample; trends show in the validation loss and the gap.

### Overfitting test

`--overfit N` trains on about N fixed training samples: whole designated folders with all their mains (counterfactual pairs) up to about N / 2 samples, one target per main, filled up with unspecified samples (first those of the same folders). The same samples are used for validation and the gap, and the pools are drawn 50 : 50 unless `--unspec-schedule` is given.

```bash
# quick test (default role tid)
python train_multiref.py --data-dir data/multiref --output-dir runs/overfit \
    --overfit 8 --max-steps 1500 --warmup-steps 50 --grad-accum 1 \
    --val-every 100 --image-every 250 --save-every 500

# setting of the reference runs in *Current findings* (overfit_tid_t3; overfit_tid_hs_long with --main-t 2)
python train_multiref.py --data-dir data/multiref --output-dir runs/overfit_tid_t3 \
    --role tid --main-t 3 \
    --overfit 8 --max-steps 5000 --warmup-steps 50 --grad-accum 1 --lr 3e-4 \
    --unspec-schedule 0.3 --designated-sigma logit_normal \
    --val-every 250 --image-every 500 --save-every 1000
```

What to check:
- `gap_acc` rises from chance (1 / number of inputs) to 1 and `gap` becomes clearly positive, **also at $\sigma$ = 0.9**; with the second command this happened at about step 1250 ($t = 3$; step 1500 with $t = 2$).
- In the sample images of the designated folder, the images of the different mains differ and follow their own targets; "no main" follows the unspecified target; no main is a plain copy of its input. A main that keeps failing while the others work is often a target problem: check it against the *Selection criteria*.
- `val_loss_designated` and `val_loss_unspecified` decrease (the designated loss can end lower, since the overfitting subset has one target per main).
- Then run `eval_multiref.py` on the last checkpoint (below).

The overfitting test only shows that the marker *can* carry the designation; it says nothing about generalization (validation = training samples).

### Evaluation (`eval_multiref.py`)

Measures, for a checkpoint, whether the model follows the **designation** or a position (a model trained with `slot` can learn "the first input is the main image" even without a designated main), and whether a designation produces the target or a **copy** of the main input.

```bash
# checkpoint (overfitting run: the overfit subset; normal run: the validation split)
python eval_multiref.py --checkpoint runs/<run>/checkpoint-005000

# base model with the run's settings, for comparison
python eval_multiref.py --checkpoint runs/<run>/checkpoint-005000 --base-only \
    --output-dir runs/<run>/eval_base

# more validation samples of a normal run
python eval_multiref.py --checkpoint runs/<run>/checkpoint-010000 --split val --max-samples 16

# loss metrics only (no generated images, fast)
python eval_multiref.py --checkpoint runs/<run>/checkpoint-005000 --copy-seeds 0 --image-folders 0
```

The settings (role, layout, `--main-t`, LoRA, data, overfit subset, seed) are read from `runs/<run>/args.json`. For each designated sample (main $k$, same noise and $\sigma$ for every condition):

| Metric | Definition | Reading |
|---|---|---|
| `gap`, `gap_acc` | as in training | equals the gap logged at the checkpoint's step with the same samples and noise |
| `bias` | $\overline{L(\text{base prompt}, j \text{ first})}_{j \ne k} - L(\text{base prompt}, k \text{ first})$ | > 0: target $k$ is favoured when input $k$ is first, without any designation |
| `prompt_effect` | $L(\text{base}, k \text{ first}) - L(\text{designated } k)$ | `slot` only (same order, only the prompt differs) |
| `bias_ratio` | `bias` / `gap` | < 0.2: the designation is followed; > 0.7: a position shortcut; ignore it while `gap` < 0.01 |

**Copy rate** (`--copy-seeds` > 0, on generated images). For every evaluated designated sample and seed, an image is generated with main $k$ designated (inputs in their original order), plus one image without a main per folder. After the transformer is unloaded, the images are compared with image-level CLIP features of the Kandinsky 2.2 prior (the features of `select_targets.py`):

| Metric | Definition | Reading |
|---|---|---|
| `copy_rate` | share with $\cos(\text{out}, \text{input}_k) > \max_{T \in \text{targets}_k} \cos(\text{out}, T)$ | the output is closer to the main **input** than to its own **target**: the designation produced the input. Wanted: near 0 |
| `near_copy_rate` | share with $\cos(\text{out}, \text{input}_k) \ge$ `--copy-max` (0.80, the copy threshold of `select_targets.py`) | near copies regardless of the target |
| `input_acc` | share where input $k$ is the input most similar to the output (chance 1 / n inputs) | the main input dominates; together with a high `copy_rate`: "designated = copied" |
| `target_acc` | share where target $k$ is more similar to the output than every target of the other mains (chance 1 / n mains) | image-level designation accuracy (folders with ≥ 2 mains) |
| `near_copy_rate_no_main` | no-main images with $\max_j \cos(\text{out}, \text{input}_j) \ge$ `--copy-max` | one input copied although nothing is designated |

`cos_target_input` (similarity of the kept target to its main input) is reported for reference: when it is itself high, outputs that follow the target are also close to the input. Compare with `--base-only`, which copies one reference under every condition. Per-main values are in `summary.json` (`copy.per_main`) and the console; `copy.csv` has one line per image, `copy_overview.jpeg` shows the images (red caption = copy) and `copy/` the generated images. With 8 designated samples and 2 seeds this generates about 20 images (≈ 28 denoising steps each); `--copy-seeds 0` skips it.

Output (default `<checkpoint>/eval/`): `summary.json` (overall and per $\sigma$, copy metrics), `per_sample.csv`, `copy.csv`, `copy_overview.jpeg`, and `rotation_<folder>.jpeg` for `--image-folders` (2) designated folders: per seed one row "no main, input $i$ first" and one row "main = input $i$". If the first row changes with the first input in the same way as the second row, the first position acts as the main image. With `tid` no position is special, so `bias` should stay at the base model's value.

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | `checkpoint-XXXXXX` folder |
| `--run-args` | `<checkpoint>/../args.json` | Settings of the run (fallback: `train_state.pt`) |
| `--base-only` | off | Base model with the run's settings (no LoRA, no $E_{main}$) |
| `--split` | `auto` | `auto` = the overfit subset for an overfitting run, else `val`; or `train`, `val`, `overfit` |
| `--data-dir`, `--cache-dir` | the run's | Override the data location |
| `--max-samples`, `--sigmas`, `--noises` | the run's `--gap-samples`, `--gap-sigmas`, `--gap-noises` | Designated samples, $\sigma$ values, noise draws |
| `--image-folders`, `--image-seeds`, `--image-steps` | 2, 2, 28 | Rotation images (0 folders = none); `--image-steps` also for the copy images |
| `--copy-seeds`, `--copy-max` | 2, 0.80 | Seeds per designated sample for the copy rate (0 = none); near-copy threshold |
| `--output-dir` | `<checkpoint>/eval` | Output folder |

### Parameters (`train_multiref.py`)

| Argument | Default | Description |
|---|---|---|
| `--data-dir`, `--cache-dir`, `--output-dir` | `data/multiref`, `<data-dir>/cache`, `runs/multiref` | Manifests, cache, run folder |
| `--unspec-schedule` | `0:1.0,0.3:1.0,0.4:0.65` (`0.5` with `--overfit`) | Probability of the unspecified pool over training progress (`progress:prob` points, linear in between; one value = constant) |
| `--overfit` | 0 | Overfitting test on about N samples |
| `--no-shuffle`, `--preload` | off | Fixed input order (debugging); keep all latents in memory |
| `--base-model` | `black-forest-labs/FLUX.1-Kontext-dev` | Transformer weights |
| `--role` | `tid` | Main-image designation: `tid`, `embed`, `slot`, `index` |
| `--main-t` | 3 | Position id $t$ of the main reference with `--role tid` (the others have 1; 2 before 2026-11-22) |
| `--layout` | `canvas` | Position ids of the references (`per_t` not with `tid`) |
| `--lora-rank`, `--lora-alpha`, `--lora-layers` | 32, = rank, see Defaults | LoRA configuration |
| `--init-lora` | none | Start from LoRA weights with the same rank and layers |
| `--guidance-scale` | 3.5 | Guidance embedding (use the same value at inference) |
| `--max-steps`, `--grad-accum` | 10000, 4 | Optimizer steps, samples per step |
| `--lr`, `--role-lr`, `--weight-decay`, `--warmup-steps`, `--max-grad-norm` | 1e-4, 1e-3, 1e-4, 500, 1.0 | Optimization (`--role-lr` only for $E_{main}$) |
| `--weighting-scheme`, `--logit-mean`, `--logit-std`, `--mode-scale` | `none`, 0, 1, 1.29 | Timestep density (diffusers); `logit_normal` with `--logit-mean` > 0 favours low $\sigma$ |
| `--designated-sigma` | `same` | $\sigma$ of designated samples: as unspecified, or `logit_normal` (direct, $\mu > 0$ favours high $\sigma$) |
| `--designated-sigma-mean`, `--designated-sigma-std` | 1.0, 1.0 | $\mu$, $s$ of `--designated-sigma logit_normal` |
| `--seed` | 0 | Sampling, noise and overfit-subset seed |
| `--log-every`, `--val-every`, `--val-samples` | 10, 500, 16 | Logging and validation loss |
| `--gap-every`, `--gap-samples`, `--gap-sigmas`, `--gap-noises` | = `--val-every`, 8, `0.9,0.75,0.5,0.25`, 1 | Designation gap (0 = off) |
| `--image-every`, `--image-folders`, `--image-steps`, `--no-images-at-start` | 1000, 3, 28, off | Sample images |
| `--save-every`, `--resume`, `--tensorboard` | 1000, none, off | Checkpoints and logging |

---

## Inference (`generate_multi.py`)

```bash
# base model (baseline)
python generate_multi.py --folder data/no_main/<folder> --main 2

# fine-tuned LoRA (run trained with the default --role tid)
python generate_multi.py --images a.jpg b.jpg c.jpg --main 1 --lora runs/v1/checkpoint-010000/lora

# run trained with --role embed: LoRA + role embedding
python generate_multi.py --images a.jpg b.jpg c.jpg --main 1 --role embed \
    --lora runs/v1/checkpoint-010000/lora --role-embedding runs/v1/checkpoint-010000/role_embedding.safetensors
```

`--role`, `--layout` and `--main-t` must be the ones of the training run (defaults `tid`, `canvas`, 3). Without `--main-t` the value is read from the LoRA's run (`runs/<run>/args.json`), else 3; a mismatching `--role` or `--main-t` is reported. Runs trained before 2026-11-22 without `--main-t` used $t = 2$. **Main guidance** (`--main-guidance w`, `tid` / `embed`, needs `--main`): at every step with $\sigma \ge$ `--main-guidance-min-sigma` the transformer runs with and without the designation (no marker, base prompt) and uses $v = v_{none} + w\,(v_{main} - v_{none})$; $w = 1$ is off, several weights can be given (one output file each, `__mg{w}`), guided steps cost two forward passes. It amplifies a weak designation at high noise, where the composition is decided:

```bash
python generate_multi.py --folder <folder> --main 3 --lora runs/<run>/checkpoint-005000/lora \
    --main-guidance 1 2 3 5 --main-guidance-min-sigma 0.5 --num-seeds 2 --output-dir outputs/main_guidance
```

`--main k` is 1-based in the given order; without it no image is designated. Defaults: `--ref-size 512`, `--size 1024`, `--steps 28`, `--guidance-scale 3.5`, `--num-seeds 4`. Output (`--output-dir`, default `outputs/multi`): one image and one preview (inputs with the main framed in red, result, prompt) per seed, and a json with the prompt and the model order of the inputs.

## Sanity checks (`check_multiref.py`)

```bash
python check_multiref.py --folder data/no_main/<folder>                           # all checks
python check_multiref.py --skip-model                                              # checks 0 and 0b only
python check_multiref.py --folder data/no_main/<folder> --baseline-roles tid,slot --main-t 2
```

| Check | Expected |
|---|---|
| 0 `canvas` tokens / ids equal the stock packing of width-concatenated latents (no weights needed) | PASS |
| 0b `tid` marker: with a main reference only the $t$ id of its tokens becomes `--main-t`; tokens, spans and $(h, w)$ unchanged (no weights needed) | PASS |
| 1 one reference: `generate_multi` equals the stock pipeline (reference and output at `--size`) | max diff 0 |
| 2 two references: equals the stock pipeline with width-concatenated latents up to token order | small difference |
| 3 $E_{main} = 0$: hook on equals hook off | max diff 0 |
| 4 a random $E_{main}$ changes only the tokens of the main reference | PASS |
| 5 baseline images of the base model: first row no main, "marked" prompt without marker, `per_t`; then one row per `--baseline-roles` (default `tid`) with each input as main | `outputs/check_multiref/baseline_<folder>.jpeg`; with `tid` all references should still be used (no image dropped, no unrelated object) |

---

## Test images from the paper's training set

Dataset: [`kfirgold99/Inspiration-Seeds-Training`](https://huggingface.co/datasets/kfirgold99/Inspiration-Seeds-Training) (2085 rows, Parquet, MIT). Columns:

- `output_image`: the 1024×1024 **source image** ($I_{comb}$), the image to decompose.
- `input_image`: the paper's own decomposition as a 1024×1024 grid ($I_A$ top-left, $I_B$ bottom-right).
- `sample_id`: e.g. `006536c4-…__topk_32__frac_0.7`. Some rows use numeric ids instead of UUIDs.

```bash
python fetch_samples.py
```

This streams the dataset, shuffles with a fixed seed, keeps 20 distinct sources, and writes:

```
samples/
├── sources/        # source images to decompose
├── paper_inputs/   # paper's decomposition: <uid>_A.jpeg, <uid>_B.jpeg
└── samples.json    # uid, sample_id, topk, frac (null if not encoded in the id)
```

A `Fatal Python error: PyGILState_Release` printed at interpreter exit comes from the streaming loader's background thread and does not affect the saved files.

---

## Current findings

### Adaptive decomposition (v4 – v7.2)

- Adaptive grouping gives 2–4 main groups per image; coverage expansion separated the inputs on 16/20 test images, and merging removes near-duplicate inputs.
- The main subject usually sits in the shared part $S$ and therefore appears in every input (e.g. a phone with different screen patterns). Scaling all of $S$ (`--alpha-r`) helps only a little; 0.7 is a better trade-off than 0.5.
- The SAE decoder bias alone decodes to a brown furry animal; inputs whose own group is weak are dominated by it and show unrelated animals.
- A subject subspace built from CLIP text embeddings does not work (modality gap: `proj_ratio` ≈ 0.17 on the phone image, and removing it left the phone unchanged). The prior-based image-space subspace (v7.1) is in the same space as $e$.
- Weakening the subject in every input made small subjects (a kite) disappear from all inputs; v7.2 keeps the subject in one input instead.

### Multi-reference fine-tuning (overfitting tests)

All tests use `--overfit 8`: one designated folder with 4 mains (4 counterfactual samples) and 4 unspecified samples; validation and gap on the same samples, so the gap is measured on only 4 designated samples (`gap_acc` in steps of 0.25).

| Role | Setting | Result |
|---|---|---|
| `embed` | 1500 steps, pools 50 : 50, default learning rates | **Not learned.** "No main" and all four mains give the unspecified target; `val_loss_designated` 0.475 → 0.445. $E_{main}$ grows slowly (norm 0.25 at step 160, 1.05 at step 1500, long plateaus): a 3072-d vector of norm ≈ 1 is a small perturbation of the token states, so the model does not use it and $E_{main}$ gets no consistent gradient |
| `embed` | `--lr 3e-4 --unspec-schedule 0.3 --designated-sigma logit_normal`, 2000 steps (`runs/overfit_embed_hs`) | **Learned in the loss** (`gap_acc` 1.0 at step 2000, `gap_s0.9` 0.005, `bias_ratio` 0.11); $E_{main}$ norm 4.6 (10 % of the token norm) and still growing. Not yet visible in the sample images. The first failure was mainly the setting (learning rate, steps, $\sigma$ density) |
| `index` | same subset | **Not learned** ("image k" in the prompt) |
| `slot` | `--lr 3e-4 --unspec-schedule 0.3 --designated-sigma logit_normal`, 5000 steps | **Learned, through a shortcut.** `gap_s0.9` > 0.01 at about step 1000 and ≈ 0.12 at step 2000, but `eval_multiref.py` gave `bias_ratio` ≈ 0.97: the model favours whichever input is first, also without a designation |
| `tid`, $t = 2$ | same as `slot` (`runs/overfit_tid_hs_long`) | **Learned, without a position shortcut**: `bias` ≈ 0 as for the base model (`bias_ratio` −0.07 at step 5000), but weak at high noise |
| **`tid`, $t = 3$** | same as `slot` (`runs/overfit_tid_t3`) | **Learned, 2–3 times stronger than $t = 2$**, also at high noise; current default. Details below |

`tid` with $t = 2$ (`log.jsonl`):

| Step | `gap` | `gap_s0.9` | `gap_s0.25` | `gap_acc` | `val_loss_designated` / `_unspecified` |
|---|---|---|---|---|---|
| 0 | ≈ 0 | ≈ 0 | ≈ 0 | 0.25 (chance) | 0.463 / 0.367 |
| 250 – 1250 | < 0.0005 | < 0.0008 | ≈ 0 | 0.31 – 0.69 | 0.44 → 0.39 / ≈ 0.32 |
| 1500 | 0.003 | 0.004 | 0.0005 | 0.94 | 0.344 / 0.304 |
| 1750 | 0.018 | **0.051** | 0.002 | 0.94 | 0.308 / 0.289 |
| 2000 – 5000 | 0.010 – 0.017 | 0.007 – 0.024 | 0.018 – 0.027 | **1.0** (every $\sigma$) | 0.25 → 0.045 / 0.28 → 0.144 |

- **The base model ignores $t = 2$** (gap ≈ 0 at step 0; `eval_multiref.py --base-only`: gap and bias within ±0.002, which is the noise level of these metrics); the LoRA learns the marker, about 1.5 times slower than `slot` (onset ≈ 1500 vs ≈ 1000 steps, i.e. ≈ 260 vs ≈ 175 draws of each designated sample).
- **Weaker at high noise than `slot`**: `gap_s0.9` ≈ 0.009 at step 2000 vs ≈ 0.12 for `slot`. Part of the `slot` gap comes from the position shortcut, so the two are not directly comparable.
- **The gap moves to low noise while the targets are memorized**: `gap_s0.9` peaks at step 1750 (0.051) and falls to 0.007 at step 5000, while `gap_s0.25` grows to ≈ 0.022. Late in the run the marker mostly recalls the details of a memorized target; the composition-level designation (high $\sigma$), which is what has to generalize, stays weak.
- **Sample images at step 5000**: main = person and main = bird on the tower follow their targets; main = flowers is almost identical to "no main" (which already leans toward the flowers instead of the source composition); main = bust is a blurred **copy** of that input instead of its target. The unspecified folders reproduce their source images.
- **Main guidance** (`generate_multi.py --main-guidance`, step-2000 checkpoint of the $t = 3$ run, where the sample images do not yet differ) pulls each designated main toward its own target (layout and large structures first, the subject later): the learned direction is right, only weak at high noise.

$t = 3$ against $t = 2$ (same subset, seed and settings; gap / `gap_s0.9` / `gap_s0.25`):

| Step | $t = 2$ | $t = 3$ |
|---|---|---|
| 1250 | 0.0005 / 0.0008 / 0.0000 | 0.0015 / 0.0024 / 0.0001 |
| 2000 | 0.0065 / 0.0089 / 0.0055 | 0.0110 / 0.0114 / 0.0107 |
| 3000 | 0.0124 / 0.0101 / 0.0202 | 0.0299 / 0.0227 / 0.0534 |
| 4000 | 0.0115 / 0.0131 / 0.0185 | 0.0261 / 0.0162 / 0.0547 |
| 5000 | 0.0106 / 0.0067 / 0.0223 | 0.0320 / 0.0224 / 0.0622 |

- **Stronger and earlier**: onset at ≈ 1250 instead of ≈ 1500 steps; from step 2000 on, the gap is larger at every one of the 13 evaluations (≈ 3× at step 5000). Relative to the designated loss, the gap at step 5000 is ≈ 60 % ($t = 3$) vs ≈ 25 % ($t = 2$). The validation losses are similar (unspecified slightly lower with $t = 3$).
- **The high-$\sigma$ gap keeps growing** (0.011 → 0.022) instead of fading (0.012 → 0.007), so the composition-level designation is not displaced by memorization.
- **Sample images at step 5000**: person, flowers and bird on the tower follow their targets with both values; "no main" gives the source composition with $t = 3$ but the flowers target with $t = 2$ (`gap_none` 0.022 vs 0.006): "no designation" became a separate condition.
- **The statue main failed with both values** (a copy of the statue, or the other bird input's target). Its kept target fails the *Selection criteria* ① and ②: real birds instead of the statue, in a folder with a second bird input. This is a label problem, not a limit of the marker.
- **High-noise sampling** (`--designated-sigma logit_normal`) learned the designation about twice as fast as uniform $\sigma$.
- **Normal runs are undertrained**: with `--lr 1e-4`, 3000 steps and about 20 draws per designated sample, no gap appears; the data (20 sources, 14 designated folders, 1 designated validation folder) is also too small to measure generalization.
- The training loss printed every 10 steps is dominated by the random $\sigma$ and pool of each sample; trends show only in the fixed validation loss and the gap.

### Fixed-K baseline

| Setting | Observation |
|---|---|
| K = 2, λ = 1 (original method) | Works on simple images, but inputs often share the main subject |
| K = 3, λ = 1 | Too weak: inputs stay close to the source and to each other (other clusters keep half their weight) |
| K = 4, λ = 3 (= K − 1) | Clear separation on `cat1` (form/material, painted figurine, cat identity, color/stripe pattern) |
| K = 4, λ = 3.5 | No clear gain over λ = 3; more artifacts |

Theoretical λ range per K (between "same pairwise distance as K = 2" and "remove other clusters exactly"):

| K | $2(K-1)/K$ | $K-1$ |
|---|---|---|
| 2 | 1 | 1 |
| 3 | 1.33 | 2 |
| 4 | 1.5 | 3 |

Current working default: **λ = K − 1**.

### Cleaning fraction

| `--clean-clusters-frac` | Clusters | Residual | Typical failure |
|---|---|---|---|
| High (0.5–0.7) | Small, pure | Large, holds shared concepts | The main subject (e.g. a person) appears in every input |
| Low (0–0.2) | Large, mixed | Small | Fewer shared subjects, but more semantically unclear inputs: collages of unrelated fragments, attributes without their subject, hybrids (e.g. a bird-person) |

Lowering the cleaning fraction also lengthens $v_i$, so the same λ pushes further and causes more artifacts; λ and the cleaning fraction should be tuned together.

### Measuring weights in generated targets

On 20 K = 2 folders with `--weights none` (the target is the unedited reconstruction, so ideally $\hat w = m$), the projection measurement ($\hat e \approx a_0 r + \sum_i \alpha_i d_i$ by NNLS):

- relative residual 0.46–0.63 in every folder: more than half of the re-encoded embedding is not explained by $r$ and the $d_i$ (cosine to the synthesized embedding ≈ 0.8);
- $a_0$ fairly stable (0.70–1.00);
- $\alpha_i$ erratic (0.00–3.35) with no consistent direction; clusters with small $m_i$ are often pushed to 0;
- only 5/20 within the 0.1 tolerance.

With weights, the projection could contradict what is visible (e.g. `high,low` targets that clearly resemble input 0 measured as $\alpha_0 \approx 0$), and iterative calibration diverged (coefficients up to $10^{12}$ before damping and clipping were added). The **Kandinsky → CLIP round trip** changes the embedding by more than the cluster contributions themselves, so embedding-space measurements of $\hat w$ are not reliable. This is why the default is now: label = $w$, plus a rank check based on similarity to the input images.

---

## Known issues

### Adaptive decomposition (`decompose_v7_2.py`)

1. **Content removed from every input.** Minor groups, ambiguous (cleaned) features ($\beta = 1$) and the weakened subject in non-keeper inputs are subtracted from all inputs; small objects that land there appear in no input. A planned fix is to assign them to exactly one input and to check coverage per noun.
2. **One subject per image.** Only the top-1 noun is handled; multi-object scenes have several subjects, and the retrieved one may be a minor object.
3. **SAE bias.** Weak groups decode to unrelated animals.
4. **Retrieval vocabulary.** Subjects outside the built-in list are matched to the nearest listed noun; use `--subject-map` or `--noun-vocab`.

### Sweep scoring (`score_sweep.py`)

1. **Uncalibrated thresholds.** The defaults have not yet been compared with hand labels on a larger batch. The docstring mentions a `check_scoring.py` for this comparison, but it is not in the repository yet.
2. **The score favours more inputs.** Every unique input adds its `own_p` (≥ `--min-own`), so a clear 4-input folder usually beats a clean 2-input folder, and the picks may lean toward 3–4 inputs. Check `summary.json` → `by_n_inputs`.
3. **`--vlm-top 4` of 12 folders.** Folders below the first four candidates of an image are not checked and cannot be picked; if all four fail, the image has no pick although a later folder might pass. Check `images_without_no_main_pick` / `images_without_main_pick` and raise `--vlm-top` if needed.
4. **Two unspecified samples per source.** If the no-main pick and the main pick of an image differ, both go to `data/no_main/` with the same target (the source image), so that source counts twice in the unspecified pool. Folders copied by hand earlier add to this.
5. **Export deletes folders** that are no longer picks (see *Export*), including synthesis sources that already have designated targets.

### Designated-subject synthesis (`synthesize_v2_1.py`)

1. **Provisional grids and threshold.** The close / far threshold and both 2 × 2 grids come from 7 hand-picked cells on 14 folders.
2. **Coarse measurements.** `cos_main`, `lift` and `margin` use global CLIP features: they show whether the target moved toward the main, but cannot tell $s = 0.7$ from $s = 0.9$ (shape and layout), so `select_targets.py` only suggests and the final selection is manual. Its thresholds were not recalibrated for 1024 px targets.
3. **Close mains barely change.** For close mains the selected targets are close to the reconstruction (`lift` +0.01 to +0.04).
4. **Some mains have no usable cell**, and some change abruptly between $s = 0.4$ (near copy) and $s = 0.7$ (main lost).
5. **Targets of different mains can look alike** when the inputs of a folder are similar; such input sets do not give useful counterfactual pairs.
6. **Synthesized targets are Kandinsky images** (1024 px, generated from 768 px inputs) and have a lower quality than the real source images used as targets of unspecified samples. Kandinsky 2.2 is mostly used at 512–768 px; at 1024 px check the overviews for repeated structures, and the far-class $s = 0.4$ cells for blur (the upsampled init image is kept the most there).

### Multi-reference fine-tuning (`train_multiref.py`)

1. **Designation shown only on an overfitting subset** (see *Current findings*): with `tid` ($t = 3$) it is learned on one designated folder (3 of 4 mains); generalization is untested. `index` did not learn it, `slot` learned a first-position shortcut, `embed` learned it more slowly.
2. **Copying the main input.** Kontext tends to copy one reference (the base model copies the same input under every condition); when a designated target is ambiguous, the model falls back to that copy. Select targets by the *Selection criteria* and check the copy rate of a full run (`eval_multiref.py`: `copy_rate`, `copy_overview.jpeg`).
3. **Quality gap between the pools.** Unspecified targets are real images, designated targets are Kandinsky images; the model could associate "designated" with the Kandinsky look. Compare the image quality of both modes on validation folders.
4. **Single GPU only**; no multi-GPU or distributed training.
5. **`--init-lora`** needs the same rank and target layers; the paper's AI-Toolkit LoRA may not match (the number of unexpected keys is printed).
6. **Sample images are slow** (28 denoising steps per condition, ≈ 25 s per logging interval in which they are made); use `--no-images-at-start` or a larger `--image-every` for long runs.

---

## Next steps

### Multi-reference fine-tuning

1. **Review the designated targets by the *Selection criteria*** (in particular folders with two inputs of one category) before building `data/multiref_v2`.
2. **Full training run** on the extended data (`data/multiref_v2`: more sources, more designated folders, `--val-frac` ≥ 0.15 so that validation has several designated folders) with `tid`, $t = 3$ and `--designated-sigma logit_normal`; enough designated updates (more steps or a larger designated share than the default schedule), judged by the validation `gap_s0.9` and `gap_acc`. `eval_multiref.py` on the final checkpoint: `bias_ratio` should stay well below 0.2.
3. **Image-level evaluation**: copy rate, `input_acc` and `target_acc` are in `eval_multiref.py`; still missing: permutation invariance (same main, shuffled inputs), balance without a main (every input used) and collage detection.
4. **Optional**: $t = 4$, or `tid` combined with $E_{main}$, if the high-noise gap stays weak in the full run.

### Adaptive decomposition

1. **Coverage**: assign minor groups, ambiguous features and removed parts to exactly one input instead of removing them from all; add a per-noun coverage check (strength in the source vs. maximum over the inputs) to flag samples where a concept is lost.
2. **Bias-dominated groups**: detect inputs with a high `cos_to_bias` / weak own group and drop or merge them.
3. **Several subjects**: handle the top nouns with a sufficient margin, each kept in one input.
4. **Seed stability** as a filter: decode with 2–3 seeds and keep inputs whose content is stable.
5. **Calibrate `score_sweep.py`** against hand labels (no-main / main / reject per folder) on a first batch of the sweep, and decide whether the score should be normalized by the number of inputs.
6. **Calibrate the suggestions of `select_targets.py`** against the manual selection (cells moved between `keep/` and `drop/`) on 1024 px targets, and add a structural metric such as DINO to tell shape-preserving cells apart.

### Fixed-K baseline (earlier plan)

1. **Diagnose shared semantics**
   - Decode the dropped features' contribution (together with $r$) and check whether the shared concept is there.
   - Rerun with `--clean-clusters-frac 0` and compare.
   - Measure the CLIP text similarity between each $e_i$ and prompts such as "a photo of a person" (Kandinsky's prior has a text encoder in the same space). If the similarity drops but the person is still rendered, the cause is mainly Kandinsky's prior; if it does not drop, the cause is the representation or the decomposition.
   - Check how often the paper's own A/B pairs share the same subject.
2. **Possible fixes**: subtract the dropped features from every input ($e_i = e - \mathrm{dec}(s_{drop}) + \lambda v_i$); reassign ambiguous features instead of dropping them; cosine-based clustering; VLM-based filtering of groups whose inputs all contain the same subject.
3. **Calibrate λ per K**: sweep λ in embedding space (purity of re-encoded inputs, pairwise separation, norm ratio, cosine to $e$) on 50–100 sources, then confirm visually on a few values.
4. **Implement Step 3 filters** (separation, similarity to source, purity) as a separate evaluation script.
5. **Validate the rank check**: run `high,low` and `low,high` on the K = 2 folders, compare the pass rate and the CLIP/DINO agreement with visual judgment.
6. **Test whether weights survive the round trip**: compare $\Delta e = e_{target}(c) - e$ with $\Delta\hat e = \hat e(c) - \hat e(\text{none})$ (cosine similarity); high values mean weight changes are visible after generation.
7. **Reduce the round-trip error**: more Kandinsky inference steps (currently 18), a different guidance scale, or averaging over seeds; measure the relative residual of `none` targets for each setting.
8. **Batch mode for Step 4**: sample level combinations per pattern (Step 6 of the plan), generate targets, keep samples that pass the rank check, and write training samples with their metadata.

---

## Acknowledgements

Original code, model and data: Kfir Goldberg, Elad Richardson, Yael Vinker — *Inspiration Seeds: Learning Non-Literal Visual Combinations for Generative Exploration* ([arXiv:2602.08615](https://arxiv.org/abs/2602.08615)). MIT License.
