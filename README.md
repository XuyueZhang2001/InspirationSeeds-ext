# Inspiration Seeds: K-way Decomposition (work in progress)

This repository builds on [Inspiration Seeds](https://arxiv.org/abs/2602.08615) (Goldberg, Richardson, Vinker), which combines two input images into a new object using FLUX.1 Kontext with a LoRA trained on synthetic triplets. The triplets come from decomposing a source image into two concept clusters of CLIP-SAE features.

The long-term goal of this fork is to support **N input images with controllable per-image weights**. The current stage covers the data side: generalizing the decomposition from 2 clusters to K ∈ {2, 3, 4} clusters, and testing it on the paper's own training images.

---

## Contents

| File | Purpose |
|---|---|
| `generate.py` | Original inference script: combines two images with FLUX.1 Kontext + the released LoRA (unchanged) |
| `decompose_original.py` | Original two-cluster decomposition script (unchanged, renamed) |
| `decompose.py` | **New** K-way decomposition (K = 2, 3, 4, …), saves embeddings and metadata |
| `fetch_hf_samples.py` | Downloads a random subset of the paper's training data from Hugging Face |
| `check_sae.py` | Small diagnostic that prints the SAE structure and decoder bias |
| `assets/` | Example inputs from the original repository |
| `docs/` | Paper PDF and teaser |

---

## Setup

### Hardware

- An NVIDIA GPU is required (both scripts hard-code `cuda`).
- `decompose*.py` (Kandinsky 2.2 + SAE): about 12–16 GB of VRAM.
- `generate.py` (FLUX.1 Kontext dev, ~12B parameters, bf16): about 24 GB+ of VRAM, or use `pipe.enable_model_cpu_offload()`.

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

### Hugging Face access

`generate.py` uses `black-forest-labs/FLUX.1-Kontext-dev`, which is gated. Accept the license on its model page, create a read token, and run `hf auth login`. The decomposition models (Kandinsky 2.2, `gytdau/clip-sae-128`) and the training dataset are not gated.

---

## K-way decomposition (`decompose.py`)

### Method

For each source image, value of K, top-k and cleaning fraction:

1. **Encode**: CLIP image embedding $e$ (Kandinsky 2.2 prior, 1280-d); SAE-encode $e$ and keep the top-k features.
2. **Cluster**: decode each feature as a one-hot vector and run k-means with K clusters on these directions. K = 2 uses the original setting (`n_init="auto"`); K > 2 uses 10 restarts.
3. **Clean**: for each feature, compute its distance to the nearest *other* cluster center; within each cluster, remove the given fraction of features with the smallest such distance. For K = 2 this is identical to the original script.
4. **Contributions**: $d_i = W_{dec}\, s_i$ (computed with `model.decoder`, which has no bias), residual $r = e - \sum_i d_i$, natural proportions $m_i = \lVert d_i\rVert / \sum_j \lVert d_j\rVert$.
5. **One-vs-rest direction**: $v_i = d_i - \frac{1}{K-1}\sum_{j \ne i} d_j$.
6. **Input embedding**: $e_i = e + \lambda v_i$ (default), or $e + \lambda v_i / \lVert v_i\rVert$ with `--direction-norm unit`. Each $e_i$ is decoded by Kandinsky into one input image.

Expanding step 6:

$$e_i = r + (1+\lambda)\, d_i + \Big(1 - \tfrac{\lambda}{K-1}\Big) \sum_{j \ne i} d_j$$

So $\lambda = K-1$ removes the other clusters exactly, which matches the paper's $\lambda = 1$ for K = 2.

### Usage

```bash
# one image, K = 2 (top-k auto = 32, cleaning 0.5, lambda 1)
python decompose.py --image assets/example_decomposition_inputs/cat1.jpeg --output-dir out --k 2

# a folder of images, several K with lambda = K - 1
python decompose.py --image-dir hf_samples/sources --output-dir hf_decomposed --k 2 --lambda 1
python decompose.py --image-dir hf_samples/sources --output-dir hf_decomposed --k 3 --lambda 2
python decompose.py --image-dir hf_samples/sources --output-dir hf_decomposed --k 4 --lambda 3
```

| Argument | Default | Description |
|---|---|---|
| `--image` / `--image-dir` | (one required) | A single image, or a folder of images (`.jpg .jpeg .png .webp .bmp`) |
| `--output-dir` | `outputs` | Output root |
| `--k` | `2` | Comma-separated numbers of clusters, e.g. `2,3,4` |
| `--top-k` | `auto` | `auto` = 16·K (32 / 48 / 64 for K = 2 / 3 / 4), or a comma-separated list |
| `--clean-clusters-frac` | `0.5` | Comma-separated cleaning fractions; `0` disables cleaning |
| `--lambda` | `1.0` | Step size along $v_i$ |
| `--direction-norm` | `none` | `none` (paper scale) or `unit` (normalized direction) |
| `--save-steps` | off | Also save the half step ($\lambda/2$) for each cluster |
| `--overwrite` | off | Recompute folders that already contain `meta.json` (otherwise skipped) |

All combinations of K × top-k × cleaning fraction are run. Each image is CLIP/SAE-encoded once and reused for every combination.

### Output

One folder per combination: `{image}_K{K}_topk{top-k}_frac{frac}_lam{lambda}/`

| File | Content |
|---|---|
| `input_0.jpeg` … `input_{K-1}.jpeg` | One input image per cluster |
| `input_i_half.jpeg` | Half step (only with `--save-steps`) |
| `recon.jpeg` | Kandinsky reconstruction of the unedited embedding $e$ |
| `embeds.pt` | `e`, `d` (K×1280), `r`, `v`, `e_inputs`, `cluster_labels` |
| `meta.json` | Parameters; per cluster: kept and dropped feature ids/values, size before and after cleaning, $m_i$, $\lVert d_i\rVert$, $\lVert v_i\rVert$; global: $\lVert e\rVert$, $\lVert r\rVert$, $\lVert r\rVert/\lVert e\rVert$, minimum cluster size after cleaning, silhouette before and after cleaning |

Filtering metrics are **recorded only**; no decomposition is discarded.

### Reproducing the original script

```bash
python decompose.py --image <img> --output-dir reproduce --k 2 \
  --top-k 32,64 --clean-clusters-frac 0.5,0.7 --lambda 1 --save-steps
```

| `decompose_original.py` | `decompose.py` |
|---|---|
| `step_-1.jpeg` | `input_0.jpeg` |
| `step_-0.5.jpeg` | `input_0_half.jpeg` |
| `step_0.jpeg` | `recon.jpeg` |
| `step_0.5.jpeg` | `input_1_half.jpeg` |
| `step_1.jpeg` | `input_1.jpeg` |

Results should match up to half-precision rounding (directions are computed in float32 and cast back).

The original script accepts only one image per call; to run it on a folder:

```bash
for img in hf_samples/sources/*.jpeg; do
  python decompose_original.py --image "$img" --output-dir hf_decomposed_original \
    --top-k 32,64 --clean-clusters-frac 0.5,0.7
done
```

---

## Test images from the paper's training set

Dataset: [`kfirgold99/Inspiration-Seeds-Training`](https://huggingface.co/datasets/kfirgold99/Inspiration-Seeds-Training) (2085 rows, Parquet, MIT). Columns:

- `output_image`: the 1024×1024 **source image** ($I_{comb}$), the image to decompose.
- `input_image`: the paper's own decomposition as a 1024×1024 grid ($I_A$ top-left, $I_B$ bottom-right).
- `sample_id`: e.g. `006536c4-…__topk_32__frac_0.7`. Some rows use numeric ids instead of UUIDs.

```bash
python fetch_hf_samples.py
```

This streams the dataset, shuffles with a fixed seed, keeps 20 distinct sources, and writes:

```
hf_samples/
├── sources/        # source images to decompose
├── paper_inputs/   # paper's decomposition: <uid>_A.jpeg, <uid>_B.jpeg
└── samples.json    # uid, sample_id, topk, frac (null if not encoded in the id)
```

A `Fatal Python error: PyGILState_Release` printed at interpreter exit comes from the streaming loader's background thread and does not affect the saved files.

---

## Current findings

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

---

## Known issues

1. **Shared semantics across inputs.** On semantically dense images, the dominant concept (e.g. a person) appears in most or all input images. This also happens with the original K = 2 script, so it is not specific to the K-way extension. Likely causes:
   - **Cleaning moves shared concepts into the residual.** Features that relate to several clusters lie near cluster boundaries and are removed by cleaning. Their contribution stays in $r$, and $e_i = e + \lambda v_i$ keeps $r$ at full strength in every input. Increasing λ does not help because these features are not in $v_i$.
   - **SAE features are not monosemantic.** Composite features (e.g. "glowing human silhouette") spread one concept across several clusters.
   - **CLIP embeddings are global semantic summaries.** Subject identity is the strongest signal and is hard to remove by linear edits; there is no spatial separation.
   - **Kandinsky's generative prior.** Decoding is non-linear and threshold-like: a small residual "person" signal is often rendered as a full person.
   - **Source images with tightly coupled concepts** (everything revolves around one theme) do not contain K independent aspects.
2. **Uneven cluster sizes.** k-means can produce small clusters; with cleaning 0.5 some clusters end up with only a few features, more often for larger K.
3. **Clustering uses Euclidean distance on raw decoder directions.** Directions with large norm can dominate. Cosine-based clustering (normalize first) is not implemented yet.
4. **No filtering yet.** Separation, similarity-to-source and purity filters (Step 3 of the plan) are not implemented; only cluster-level metrics are recorded.
5. **Reconstruction loss.** Every image passes through Kandinsky 2.2 at 512 px, so even `recon.jpeg` differs from the source. Inputs and future synthesized targets are lower quality than the source images.
6. **SAE bias.** `model.decode()` adds a constant bias (‖decode(0)‖ ≈ 26), while `model.decoder` (the linear layer) has none. Clustering uses `decode()`, which is unaffected because the bias is shared by all features; contributions $d_i$ use `decoder`, so the bias stays in $r$.
7. **λ is not yet calibrated** beyond a handful of images.

---

## Next steps

1. **Diagnose shared semantics**
   - Decode the dropped features' contribution (together with $r$) and check whether the shared concept is there.
   - Rerun with `--clean-clusters-frac 0` and compare.
   - Measure the CLIP text similarity between each $e_i$ and prompts such as "a photo of a person" (Kandinsky's prior has a text encoder in the same space). If the similarity drops but the person is still rendered, the cause is mainly Kandinsky's prior; if it does not drop, the cause is the representation or the decomposition.
   - Check how often the paper's own A/B pairs share the same subject.
2. **Possible fixes**: subtract the dropped features from every input ($e_i = e - \mathrm{dec}(s_{drop}) + \lambda v_i$); reassign ambiguous features instead of dropping them; cosine-based clustering; VLM-based filtering of groups whose inputs all contain the same subject.
3. **Calibrate λ per K**: sweep λ in embedding space (purity of re-encoded inputs, pairwise separation, norm ratio, cosine to $e$) on 50–100 sources, then confirm visually on a few values.
4. **Implement Step 3 filters** (separation, similarity to source, purity) as a separate evaluation script.
5. **Step 4**: sample level combinations, synthesize targets $e_{target} = r + \sum_i c_i d_i$ from the saved `embeds.pt`, measure $\hat w$, and calibrate.

---

## Acknowledgements

Original code, model and data: Kfir Goldberg, Elad Richardson, Yael Vinker — *Inspiration Seeds: Learning Non-Literal Visual Combinations for Generative Exploration* ([arXiv:2602.08615](https://arxiv.org/abs/2602.08615)). MIT License.
