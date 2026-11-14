# Inspiration Seeds: Multi-Input Decomposition (work in progress)

This repository builds on [Inspiration Seeds](https://arxiv.org/abs/2602.08615) (Goldberg, Richardson, Vinker), which combines two input images into a new object using FLUX.1 Kontext with a LoRA trained on synthetic triplets. The triplets come from decomposing a source image into two concept clusters of CLIP-SAE features.

The long-term goal of this fork is to support **N input images with controllable per-image weights**. The current stage covers the data side:

1. decomposing a source image into **N semantically distinct reference images** (N = 2–4, chosen per image). The current recommended script is **`decompose_v7_2.py`** (adaptive SAE feature grouping + an image-space subject subspace); `decompose.py` is the earlier fixed-K baseline;
2. synthesizing weighted target images from a decomposition (`synthesize_target.py`, currently for `decompose.py` outputs), and checking whether the weights are visible in the generated targets.

Last README update: 2026-11-14.

---

## Contents

| File | Purpose |
|---|---|
| `generate.py` | Original inference script: combines two images with FLUX.1 Kontext + the released LoRA (unchanged) |
| `decompose_original.py` | Original two-cluster decomposition script (unchanged, renamed) |
| **`decompose_v7_2.py`** | **Recommended** decomposition: adaptive grouping of SAE features, coverage expansion, merging, shared-part weakening, subject kept in one input (see below) |
| `decompose_groups.py`, `decompose_v4.py` … `decompose_v7_1.py` | Earlier versions of the adaptive decomposition, kept for comparison (history below) |
| `decompose.py` | Fixed-K baseline (k-means into K = 2, 3, 4 clusters), saves embeddings and metadata |
| `synthesize_target.py` | **New** weighted target synthesis from `decompose.py` outputs, with an input-similarity rank check |
| `fetch_hf_samples.py` | Downloads a random subset of the paper's training data from Hugging Face |
| `probe_inputs.py` | Checks whether the released model uses both inputs (swap / duplicate / blank conditions, DINOv2 and CLIP metrics) |
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
python decompose_v7_2.py --image-dir hf_samples/sources --output-dir output/hf_v72

# one image, with diagnostic images
python decompose_v7_2.py --image hf_samples/sources/<name>.jpeg --output-dir output/v72_test --debug-images

# another decoding seed (same embeddings, different renderings)
python decompose_v7_2.py --image-dir hf_samples/sources --output-dir output/hf_v72_seed2 --seed 2

# other cleaning fraction / shared-part strength
python decompose_v7_2.py --image-dir hf_samples/sources --output-dir output/hf_v72_c05 --clean-frac 0.5

# force the subject noun for some images
echo '{"14": "kite", "phone_01": "smartphone"}' > subjects.json
python decompose_v7_2.py --image-dir hf_samples/sources --output-dir output/hf_v72 --subject-map subjects.json
```

Existing output folders are skipped unless `--overwrite` is given, so use a new `--output-dir` for every setting. The first time a noun is used, its subspace costs 2 × 40 × `--prior-samples` prior runs (batched); afterwards it is read from `text_cache/`.

### Data-generation runs

The decompositions are used to build fine-tuning data, so every source folder is decomposed with **several settings**, and the good input sets are selected afterwards. All runs start from the defaults (`--alpha-r 0.7 --clean-frac 0.3`, no debug images) and change one thing at a time. Each run needs its own `--output-dir`, because the per-image folder name (`{stem}_topk64_nmax4`) does not contain the parameters.

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
SRC=hf_samples/sources          # source folder
OUT=output/hf_v72               # one subfolder per run

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

Resulting layout: `output/hf_v72/<run>/<stem>_topk64_nmax{2|4}/` with `input_*.jpeg`, `overview.jpeg`, `embeds.pt`, `groups.json`, plus `output/hf_v72/<run>/summary.json`. For one source image there are therefore up to 9 candidate input sets; comparing the `overview.jpeg` of the same `<stem>` across runs is the quickest visual check.

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

No thresholds are fixed yet; they should be set after looking at a first batch.

### Parameters

Defaults are the values in the code (updated 2026-11-14: `--clean-frac 0.3`, `--alpha-r 0.7`).

**Input / output**

| Argument | Default | Description |
|---|---|---|
| `--image` / `--image-dir` | (one required) | One image, or a folder (`.jpg .jpeg .png .webp .bmp`) |
| `--output-dir` | `groups_out` | Output root; one subfolder per image |
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
| `--size` | 768 | Output resolution |

### Output

One folder per image: `<output-dir>/{image stem}_topk{top-k}_nmax{n-max}/` (the folder name does not contain the other parameters).

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
| `decompose.py` | K clusters, one-vs-rest directions (baseline below) |
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

## Fixed-K baseline (`decompose.py`)

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

Given one `decompose.py` output folder and one weight per input image (folders from `decompose_v7_2.py` are not supported yet: they store `groups.json` instead of `meta.json` and a different set of contributions):

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

### Adaptive decomposition (v4 – v7.2)

- Adaptive grouping gives 2–4 main groups per image; coverage expansion separated the inputs on 16/20 test images, and merging removes near-duplicate inputs.
- The main subject usually sits in the shared part $S$ and therefore appears in every input (e.g. a phone with different screen patterns). Scaling all of $S$ (`--alpha-r`) helps only a little; 0.7 is a better trade-off than 0.5.
- The SAE decoder bias alone decodes to a brown furry animal; inputs whose own group is weak are dominated by it and show unrelated animals.
- A subject subspace built from CLIP text embeddings does not work (modality gap: `proj_ratio` ≈ 0.17 on the phone image, and removing it left the phone unchanged). The prior-based image-space subspace (v7.1) is in the same space as $e$.
- Weakening the subject in every input made small subjects (a kite) disappear from all inputs; v7.2 keeps the subject in one input instead.

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

1. **Content removed from every input.** Minor groups, ambiguous (cleaned) features ($eta = 1$) and the weakened subject in non-keeper inputs are subtracted from all inputs; small objects that land there appear in no input. A planned fix is to assign them to exactly one input and to check coverage per noun.
2. **One subject per image.** Only the top-1 noun is handled; multi-object scenes have several subjects, and the retrieved one may be a minor object.
3. **SAE bias.** Weak groups decode to unrelated animals.
4. **Retrieval vocabulary.** Subjects outside the built-in list are matched to the nearest listed noun; use `--subject-map` or `--noun-vocab`.
5. **Not yet connected to target synthesis.** `synthesize_target.py` reads `decompose.py` outputs only.

### Fixed-K baseline (`decompose.py`)

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

### Adaptive decomposition

1. **Coverage**: assign minor groups, ambiguous features and removed parts to exactly one input instead of removing them from all; add a per-noun coverage check (strength in the source vs. maximum over the inputs) to flag samples where a concept is lost.
2. **Bias-dominated groups**: detect inputs with a high `cos_to_bias` / weak own group and drop or merge them.
3. **Several subjects**: handle the top nouns with a sufficient margin, each kept in one input.
4. **Seed stability** as a filter: decode with 2–3 seeds and keep inputs whose content is stable.
5. **Target synthesis for v7.2 outputs**: adapt `synthesize_target.py` to `groups.json` / `embeds.pt` of `decompose_v7_2.py`.

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
