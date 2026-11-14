# Inspiration Seeds: Multi-Input Decomposition (work in progress)

This repository builds on [Inspiration Seeds](https://arxiv.org/abs/2602.08615) (Goldberg, Richardson, Vinker), which combines two input images into a new object using FLUX.1 Kontext with a LoRA trained on synthetic triplets. The triplets come from decomposing a source image into two concept clusters of CLIP-SAE features.

The long-term goal of this fork is to support **N input images with an optional designated main subject** (earlier plan: controllable per-image weight levels). The current stage covers the data side:

1. decomposing a source image into **N semantically distinct reference images** (N = 2–4, chosen per image). The current recommended script is **`decompose_v7_2.py`** (adaptive SAE feature grouping + an image-space subject subspace);
2. synthesizing **designated-subject target images** from a decomposition (**`synthesize_v2_1.py`**): for every input designated as the main subject, a target pulled toward that input, so that the same input set gives different targets for different designations.

Last README update: 2026-11-14.

---

## Contents

| File | Purpose |
|---|---|
| `generate.py` | Original inference script: combines two images with FLUX.1 Kontext + the released LoRA (unchanged) |
| `decompose_original.py` | Original two-cluster decomposition script (unchanged, renamed) |
| **`decompose_v7_2.py`** | **Recommended** decomposition: adaptive grouping of SAE features, coverage expansion, merging, shared-part weakening, subject kept in one input (see below) |
| `decompose_groups.py`, `decompose_v4.py` … `decompose_v7_1.py` | Earlier versions of the adaptive decomposition, kept for comparison (history below) |
| `decompose.py` | Earlier fixed-K baseline (k-means into K = 2, 3, 4 clusters); no longer documented |
| **`synthesize_v2_1.py`** | **Recommended** designated-subject target synthesis for `decompose_v7_2.py` outputs: closeness-dependent 2 × 2 grid of interpolation weight × img2img strength (see below) |
| `synthesize_v2.py` | Previous version: the same synthesis with a full 3 × 3 grid for every main |
| `synthesize_target.py` | Earlier level-based weighted target synthesis for `decompose.py` outputs |
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
- `synthesize_v2_1.py` (Kandinsky 2.2 only, no SAE): at most the same as the decomposition.
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
4. **Decoding with Kandinsky 2.2 img2img**: init image = `input_k.jpeg`, condition = $e_T$. The strength $s$ is the diffusers img2img strength, i.e. the fraction of the noise schedule applied to the init image; **larger $s$ = further from the main input**. About 0.4 keeps outline, details and colors; 0.7 keeps the rough layout and proportions; 0.9 almost ignores the init image. The number of denoising steps actually run is `int(steps * s)`. The img2img pipeline reuses the decoder's unet, scheduler and movq, so no extra weights are loaded.
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

`synthesize_v2.py` ran the full 3 × 3 grid ($t \in \{0, 0.3, 0.6\}$, $s \in \{0.4, 0.7, 0.9\}$) on 14 folders (41 mains); the best cell was picked by hand for 7 mains.

- The input closest to the source ("dominant" input) had a closeness of 0.61–0.85, the other inputs 0.26–0.68. The dominant input was the v7.2 keeper in 11 of 14 folders.
- **Close mains**: hand-picked cells (0, 0.9), (0, 0.9), (0.3, 0.7). Their `lift` was only +0.01 to +0.04, i.e. the target is close to the reconstruction.
- **Far mains**: hand-picked cells (0.6, 0.7), (0.3, 0.7), (0.6, 0.9), (0.6, 0.4). At $t = 0$, $s \ge 0.7$ the median `lift` was about 0.
- Hand-picked cells had `cos_main` 0.52–0.78 and `margin` down to −0.15. Using "`cos_main` ≤ 0.8 and `margin` ≥ −0.15" as a criterion, 26 of 27 far mains have at least one usable cell with $s \in \{0.4, 0.7\}$, the same as with the full 3 × 3 grid; with $s \in \{0.7, 0.9\}$ only 21 of 27.
- The default grids contain 6 of the 7 hand-picked cells; the missing one, (0.6, 0.9), had almost the same values as (0.6, 0.7).

In this analysis the closeness was approximated from `synthesize_v2.py` outputs; `synthesize_v2_1.py` computes it directly, so the threshold may need a small adjustment after the first run.

### Usage

```bash
# default: class-dependent 2 x 2 grid for every input of every folder
python synthesize_v2_1.py --root hf_generate_source --output-dir output/synth/v2_1

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
| `--root` | `hf_generate_source` | Directory with decomposition folders (searched recursively for `embeds.pt` + `input_0.jpeg`), or one such folder |
| `--output-dir` | `output/synth/v2_1` | Output root; the folder structure under `--root` is kept |
| `--close-threshold` | 0.6 | An input is close if $\cos(f(\text{input}), f(\text{recon})) \ge$ this value; 0 = every input close, > 1 = every input far |
| `--close-t` | `0,0.3` | Interpolation weights $t$ for close mains |
| `--close-s` | `0.7,0.9` | img2img strengths for close mains |
| `--far-t` | `0.3,0.6` | Interpolation weights $t$ for far mains |
| `--far-s` | `0.4,0.7` | img2img strengths for far mains |
| `--mains` | all | Comma-separated input indices to designate as main |
| `--embed-only` | off | Add a reference column decoded from $e_T$ alone (no init image) |
| `--seed` | 1 | Kandinsky seed, shared by every cell |
| `--steps` | 50 | Kandinsky steps; img2img runs `int(steps * s)` of them. Use the value of the decomposition run |
| `--size` | 768 | Output resolution. Use the value of the decomposition run |
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

1. **Content removed from every input.** Minor groups, ambiguous (cleaned) features ($\beta = 1$) and the weakened subject in non-keeper inputs are subtracted from all inputs; small objects that land there appear in no input. A planned fix is to assign them to exactly one input and to check coverage per noun.
2. **One subject per image.** Only the top-1 noun is handled; multi-object scenes have several subjects, and the retrieved one may be a minor object.
3. **SAE bias.** Weak groups decode to unrelated animals.
4. **Retrieval vocabulary.** Subjects outside the built-in list are matched to the nearest listed noun; use `--subject-map` or `--noun-vocab`.

### Designated-subject synthesis (`synthesize_v2_1.py`)

1. **Provisional grids and threshold.** The close / far threshold and both 2 × 2 grids come from 7 hand-picked cells on 14 folders.
2. **Coarse measurements.** `cos_main`, `lift` and `margin` use global CLIP features: they show whether the target moved toward the main, but cannot tell $s = 0.7$ from $s = 0.9$ (shape and layout), so the best cell is still picked visually. No automatic selection or filtering is implemented yet.
3. **Close mains barely change.** For close mains the selected targets are close to the reconstruction (`lift` +0.01 to +0.04).
4. **Some mains have no usable cell**, and some change abruptly between $s = 0.4$ (near copy) and $s = 0.7$ (main lost).
5. **Targets of different mains can look alike** when the inputs of a folder are similar; such input sets do not give useful counterfactual pairs.
6. **Synthesized targets are Kandinsky images** (768 px) and have a lower quality than the real source images used as targets of unspecified samples.

---

## Next steps

### Adaptive decomposition

1. **Coverage**: assign minor groups, ambiguous features and removed parts to exactly one input instead of removing them from all; add a per-noun coverage check (strength in the source vs. maximum over the inputs) to flag samples where a concept is lost.
2. **Bias-dominated groups**: detect inputs with a high `cos_to_bias` / weak own group and drop or merge them.
3. **Several subjects**: handle the top nouns with a sufficient margin, each kept in one input.
4. **Seed stability** as a filter: decode with 2–3 seeds and keep inputs whose content is stable.
5. **Automatic cell selection for `synthesize_v2_1.py`**: label more mains, then set thresholds on `cos_main`, `lift` and `margin` (and a structural metric such as DINO) to pick one cell per main and drop mains without a usable cell; check per folder that the targets of different mains are distinguishable.

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
