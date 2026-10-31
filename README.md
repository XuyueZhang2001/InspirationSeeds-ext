# Inspiration Seeds: K-way Decomposition (work in progress)

This repository builds on [Inspiration Seeds](https://arxiv.org/abs/2602.08615) (Goldberg, Richardson, Vinker), which combines two input images into a new object using FLUX.1 Kontext with a LoRA trained on synthetic triplets. The triplets come from decomposing a source image into two concept clusters of CLIP-SAE features.

The long-term goal of this fork is to support **N input images with controllable per-image weights**. The current stage covers the data side:

1. generalizing the decomposition from 2 clusters to K ∈ {2, 3, 4} clusters (`decompose.py`), and testing it on the paper's own training images;
2. synthesizing weighted target images from a decomposition (`synthesize_target.py`), and checking whether the weights are visible in the generated targets.

---

## Contents

| File | Purpose |
|---|---|
| `generate.py` | Original inference script: combines two images with FLUX.1 Kontext + the released LoRA (unchanged) |
| `decompose_original.py` | Original two-cluster decomposition script (unchanged, renamed) |
| `decompose.py` | **New** K-way decomposition (K = 2, 3, 4, …), saves embeddings and metadata |
| `synthesize_target.py` | **New** weighted target synthesis from `decompose.py` outputs, with an input-similarity rank check |
| `fetch_hf_samples.py` | Downloads a random subset of the paper's training data from Hugging Face |
| `summarize_none.py` | Prints a table of the unweighted (`none`) target measurements across folders |
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

## Weighted target synthesis (`synthesize_target.py`)

### Method

Given one `decompose.py` output folder and one weight per input image:

1. **Weights → proportions**: levels `low` / `medium` / `high` map to center scores 1 / 3 / 9 (or are sampled log-uniformly from center × [0.8, 1.25] with `--sample-scores`); numeric weights are used as given. Scores are normalized to $w$. **$w$ is the label of the sample.**
2. **Coefficients**: $c_i = w_i / m_i$, clipped to $c_{max}$ (default 3) unless `--allow-exceed`.
3. **Target embedding**: $e_{target} = s_r\, r + \sum_i c_i d_i$ ($s_r$ = `--residual-scale`, default 1), optionally rescaled to $\lVert e\rVert$ (`--rescale-norm`), decoded once with Kandinsky (seed 1, as in `decompose.py`).
4. **Measurement and rank check** (default `--measure inputs`): `recon.jpeg`, every `input_i.jpeg` and the target are re-encoded with CLIP (and DINOv2 with `--dino`). All of them went through the same Kandinsky round trip, so they are comparable:

   $$s_i = \cos\big(f(\text{target}) - f(\text{recon}),\ f(\text{input}_i) - f(\text{recon})\big)$$

   i.e. how much the target moved from the reconstruction toward input $i$. For every pair with $w_i / w_j \ge$ `--rank-min-ratio` (default 1.5), the check requires $s_i > s_j$. A sample passes if every such pair is ordered correctly (for every feature extractor used). Patterns with all weights equal have no ordered pairs and are reported as `n/a`.

The rank check is a filter, not a calibration: the absolute values of $s_i$ are not proportions.

### Usage

```bash
# one folder (weights in input_0, input_1, ... order)
python synthesize_target.py --folder <decompose output folder> --weights high,low,low,medium

# every decompose output folder under a directory, only K = 2, also measured with DINOv2
python synthesize_target.py --root hf_decomposed --k 2 --weights high,low --dino

# unweighted target (c = 1): identical to recon.jpeg
python synthesize_target.py --root hf_decomposed --weights none
```

| Argument | Default | Description |
|---|---|---|
| `--folder` / `--root` | (one required) | One output folder, or all output folders (containing `embeds.pt` and `meta.json`) found under a directory |
| `--weights` | (required) | `none`, levels (`high,low,low,medium`) or numbers (`0.6,0.2,0.1,0.1`) |
| `--k` | none | Only process folders with this K (folders whose K differs from the number of weights are skipped anyway) |
| `--sample-scores`, `--score-seed` | off, 0 | Sample level scores within their intervals |
| `--c-max`, `--allow-exceed` | 3, off | Coefficient cap and whether to disable clipping |
| `--rescale-norm`, `--residual-scale` | off, 1 | Target embedding options |
| `--seed` | 1 | Kandinsky seed |
| `--measure` | `inputs` | `inputs` (rank check), or the older `projection` / `features` measurements |
| `--dino` | off | Also measure with DINOv2 (`facebook/dinov2-base`) |
| `--rank-min-ratio` | 1.5 | Minimum weight ratio for a pair to be rank-checked |
| `--calibrate`, `--tolerance`, `--max-iters`, `--damping`, `--w-hat-floor` | off, 0.1, 2, 0.5, 0.02 | Optional iterative calibration; only with `--measure projection` or `features` |
| `--tag`, `--overwrite` | auto, off | Output subfolder name; recompute existing targets |

### Output

Results are written next to the inputs, in `<decompose folder>/targets/<tag>/`, where `<tag>` is derived from the weights (`high-low-low-medium`, `high-low-low-medium_s3`, `w0.60-0.20-0.10-0.10`, `none`):

| File | Content |
|---|---|
| `target.jpeg` | The target image |
| `target_iter{n}.jpeg` | Every generated target (only one unless `--calibrate`) |
| `overview.jpeg` | Inputs labeled with level and $w$, followed by the target labeled with $s$ and PASS/FAIL |
| `target_meta.json` | Levels, scores, $w$, $m$, requested and used $c$, $s_i$, raw cosine similarities, relative shift from `recon`, per-pair rank results, `rank_passed`, c_max warnings |

With `--root`, a summary `targets_summary_<weights>.json` is written under the root, and the number of folders passing the rank check is printed. Requests that needed $c > c_{max}$ are listed in a warning block at the very end of the run.

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
8. **Weights are hard to measure in generated images.** The Kandinsky → CLIP round trip has a relative residual of about 0.55, larger than the cluster contributions. Projection-based $\hat w$ is unreliable (see *Current findings*), and iterative calibration is therefore off by default.
9. **The input-similarity rank check is new and not yet validated.** It only checks ordering, not proportions, and has not yet been compared systematically with human judgment or across CLIP and DINO.
10. **Labels describe the target embedding, not the target image.** If a weight does not survive the round trip, the prompt level and the image disagree; the rank check is meant to filter such samples.

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
5. **Validate the rank check**: run `high,low` and `low,high` on the K = 2 folders, compare the pass rate and the CLIP/DINO agreement with visual judgment.
6. **Test whether weights survive the round trip**: compare $\Delta e = e_{target}(c) - e$ with $\Delta\hat e = \hat e(c) - \hat e(\text{none})$ (cosine similarity); high values mean weight changes are visible after generation.
7. **Reduce the round-trip error**: more Kandinsky inference steps (currently 18), a different guidance scale, or averaging over seeds; measure the relative residual of `none` targets for each setting.
8. **Batch mode for Step 4**: sample level combinations per pattern (Step 6 of the plan), generate targets, keep samples that pass the rank check, and write training samples with their metadata.

---

## Acknowledgements

Original code, model and data: Kfir Goldberg, Elad Richardson, Yael Vinker — *Inspiration Seeds: Learning Non-Literal Visual Combinations for Generative Exploration* ([arXiv:2602.08615](https://arxiv.org/abs/2602.08615)). MIT License.
