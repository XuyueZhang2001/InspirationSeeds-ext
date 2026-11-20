"""Multi-reference conditioning for FLUX.1 Kontext (diffusers 0.43).

Reference images
    Each reference is encoded separately by the VAE (ref_size x ref_size, default 512 ->
    64 x 64 latent -> 32 x 32 = 1024 tokens). The token groups are concatenated, in model
    order, after the tokens of the image being generated. Every group is contiguous, so the
    tokens of reference k are a fixed slice of the sequence.

Position ids (t, h, w) of the reference tokens; the generated image always has t = 0
    canvas  t = 1 for every reference, groups side by side along w ("virtual canvas").
            Same ids as concatenating the latents along the width and passing them to the
            stock pipeline as one reference; only the token order differs (the stock
            packing interleaves the references row by row, which would break the
            contiguous slices needed by the role embedding).
    per_t   reference k gets t = k + 1; h and w are scaled so that every reference spans
            the whole token grid of the generated image (ablation).

Main image (role mode "tid", canvas layout only)
    The main reference keeps its place on the canvas (same h, w ids) but gets t = main_t
    (MAIN_T = 2 by default) instead of 1. The base model already distinguishes position ids,
    so the marker is visible from the first step, and without a main image all references
    have t = 1: no position is special, unlike role mode "slot" (main image moved first).

Main image (role mode "embed")
    MainRoleEmbedding adds a learnable vector E_main to every token of the main reference
    right after the patch embedding, through a forward hook on transformer.x_embedder.
    E_main starts at zero, so the model is unchanged before training. The slice is
    counted from the END of the image sequence (references come last), so it does not
    depend on the output resolution.

Pipeline
    MultiRefKontextPipeline is FluxKontextPipeline with prepare_latents overridden: when
    references are injected, it returns them as the image tokens instead of encoding the
    `image` argument. Everything else (scheduler, denoising loop, LoRA loading) is the
    stock pipeline.

Main guidance (inference only, role modes "tid" and "embed"; added 2026-11-20)
    Classifier-free-guidance-like amplification of the designation. At every denoising step
    with sigma >= main_guidance_min_sigma the transformer is run twice:
        v_main  designated input (t marker / E_main on, marked prompt)
        v_none  no main image (every reference at t = 1, E_main off, base prompt)
    and the prediction is v = v_none + w * (v_main - v_none), w = main_guidance (1 = off).
    The transformer's forward is wrapped for the duration of one pipeline call, so the
    stock denoising loop is unchanged. Guided steps cost two forward passes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import FluxKontextPipeline
from PIL import Image
from safetensors.torch import load_file, save_file

from .prompts import BASE, ROLE_MODES, arrange

LAYOUTS = ("canvas", "per_t")
MAIN_T = 2.0  # t id of the main reference in role mode "tid" (the other references have t = 1)


def marker_pos(role_mode: str, main_pos):
    """Position of the reference that gets the t marker: main_pos for role mode "tid", else None."""
    return main_pos if role_mode == "tid" else None


# --------------------------------------------------------------------------------------
# Images and latents
# --------------------------------------------------------------------------------------
def load_rgb(path) -> Image.Image:
    return Image.open(path).convert("RGB")


def center_crop_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    if w == h:
        return image
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return image.crop((left, top, left + s, top + s))


@torch.no_grad()
def encode_image(vae, image_processor, image: Image.Image, size: int, device, dtype) -> torch.Tensor:
    """VAE latent of one image at size x size, normalized as in FluxKontextPipeline._encode_vae_image.

    The image is center-cropped to a square, resized with the pipeline's image processor and
    encoded with the mode of the latent distribution (same as the pipeline's "argmax").
    Returns [1, 16, size / 8, size / 8].
    """
    image = center_crop_square(image.convert("RGB"))
    image = image_processor.resize(image, size, size)
    pixels = image_processor.preprocess(image, size, size).to(device=device, dtype=dtype)
    latents = vae.encode(pixels).latent_dist.mode()
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, (H/2) * (W/2), 4C], the Flux 2 x 2 patch packing."""
    b, c, h, w = latents.shape
    return FluxKontextPipeline._pack_latents(latents, b, c, h, w)


def make_ids(h2: int, w2: int, t: float, h_off: float = 0.0, w_off: float = 0.0,
             scale_h: float = 1.0, scale_w: float = 1.0) -> torch.Tensor:
    """(t, h, w) ids of an h2 x w2 token grid, row-major like _prepare_latent_image_ids. Returns [h2 * w2, 3]."""
    ids = torch.zeros(h2, w2, 3)
    ids[..., 0] = t
    ids[..., 1] = torch.arange(h2)[:, None] * scale_h + h_off
    ids[..., 2] = torch.arange(w2)[None, :] * scale_w + w_off
    return ids.reshape(h2 * w2, 3)


def build_reference_tokens(latents_list, layout: str = "canvas", target_hw=None, main_pos=None,
                           main_t: float = MAIN_T):
    """Packed tokens, ids and token slices of the references (in model order).

    latents_list: list of [B, 16, h, w] latents (same B)
    layout:       "canvas" or "per_t"
    target_hw:    (h2, w2) token grid of the generated image, required for "per_t"
    main_pos:     reference (model order) that gets t = main_t instead of 1 (role mode "tid",
                  see marker_pos); None = no marker. Canvas layout only; h, w ids are unchanged.

    Returns (tokens [B, L, 64], ids [L, 3], spans) where spans[k] = (start, end) is the
    slice of reference k inside the reference block (0 = first reference token).
    """
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout {layout!r}; expected one of {LAYOUTS}")
    if layout == "per_t" and target_hw is None:
        raise ValueError("layout 'per_t' needs target_hw")
    if main_pos is not None and layout != "canvas":
        raise ValueError("a t marker on the main reference (role mode 'tid') needs layout 'canvas'")
    if main_pos is not None and not 0 <= main_pos < len(latents_list):
        raise ValueError(f"main_pos {main_pos} out of range for {len(latents_list)} references")
    tokens, ids, spans = [], [], []
    start, w_offset = 0, 0
    for k, lat in enumerate(latents_list):
        _, _, h, w = lat.shape
        h2, w2 = h // 2, w // 2
        tok = pack_latents(lat)
        if layout == "canvas":
            i = make_ids(h2, w2, t=main_t if k == main_pos else 1, w_off=w_offset)
            w_offset += w2
        else:
            th, tw = target_hw
            i = make_ids(h2, w2, t=k + 1, scale_h=th / h2, scale_w=tw / w2)
        tokens.append(tok)
        ids.append(i)
        spans.append((start, start + tok.shape[1]))
        start += tok.shape[1]
    return torch.cat(tokens, dim=1), torch.cat(ids, dim=0), spans


# --------------------------------------------------------------------------------------
# Main-image role embedding
# --------------------------------------------------------------------------------------
class MainRoleEmbedding(nn.Module):
    """Learnable vector added to the tokens of the main reference after transformer.x_embedder.

    Usage:
        role = MainRoleEmbedding(transformer.inner_dim).to(device).attach(transformer)
        role.set_main(spans, main_pos)   # before each forward; main_pos None = no main image
        ...
        role.clear()

    Diagnostics: every forward with a main image also accumulates the mean per-token norm of the
    reference tokens (before E_main is added); pop_reference_norm() returns and resets it, so
    the training log can report ||E_main|| relative to the tokens it is added to.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.e_main = nn.Parameter(torch.zeros(dim))  # zero init: no effect before training
        self._span = None  # (start, end, total) inside the reference block
        self._handle = None
        self._norm_sum = None  # running sum (tensor, no host sync) of the reference token norm
        self._norm_count = 0

    def attach(self, transformer):
        self.detach()
        self._handle = transformer.x_embedder.register_forward_hook(self._hook)
        return self

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def set_main(self, spans, main_pos):
        """spans from build_reference_tokens; main_pos = position of the main reference in model order."""
        if main_pos is None:
            self._span = None
            return
        start, end = spans[main_pos]
        self._span = (start, end, spans[-1][1])

    def clear(self):
        self._span = None

    def pop_reference_norm(self):
        """Mean per-token norm of the reference tokens since the last call (None if no forward)."""
        if not self._norm_count:
            return None
        value = (self._norm_sum / self._norm_count).item()
        self._norm_sum, self._norm_count = None, 0
        return value

    def _hook(self, module, inputs, output):
        # output: [B, target tokens + reference tokens, D]; references are the last `total` tokens
        if self._span is None:
            return output
        start, end, total = self._span
        ref0 = output.shape[1] - total
        if ref0 < 0:
            raise RuntimeError(f"Sequence of length {output.shape[1]} is shorter than the reference block ({total}).")
        with torch.no_grad():  # diagnostics only
            norm = output[:, ref0:].detach().float().norm(dim=-1).mean()
            self._norm_sum = norm if self._norm_sum is None else self._norm_sum + norm
            self._norm_count += 1
        e = self.e_main.to(device=output.device, dtype=output.dtype)
        a, b = ref0 + start, ref0 + end
        # out-of-place, so gradients reach both the transformer input and E_main
        return torch.cat([output[:, :a], output[:, a:b] + e, output[:, b:]], dim=1)

    def save(self, path):
        save_file({"e_main": self.e_main.detach().float().cpu().contiguous()}, str(path))

    def load(self, path):
        state = load_file(str(path))
        with torch.no_grad():
            self.e_main.copy_(state["e_main"].to(self.e_main.device, self.e_main.dtype))
        return self


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
class MultiRefKontextPipeline(FluxKontextPipeline):
    """FluxKontextPipeline that accepts several reference images (see generate_multi)."""

    _multiref = None  # (tokens [B, L, 64], ids [L, 3]) injected for the next call

    def prepare_latents(self, image, batch_size, num_channels_latents, height, width, dtype, device,
                        generator=None, latents=None):
        if self._multiref is None:
            return super().prepare_latents(image, batch_size, num_channels_latents, height, width, dtype, device,
                                           generator, latents)
        # noise and ids of the image being generated, exactly as in the stock pipeline
        latents, _, latent_ids, _ = super().prepare_latents(None, batch_size, num_channels_latents, height, width,
                                                            dtype, device, generator, latents)
        tokens, ids = self._multiref
        image_latents = tokens.to(device=device, dtype=dtype)
        if image_latents.shape[0] != batch_size:
            if batch_size % image_latents.shape[0] != 0:
                raise ValueError(f"Cannot expand {image_latents.shape[0]} reference sets to batch size {batch_size}.")
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1)
        return latents, image_latents, latent_ids, ids.to(device=device, dtype=dtype)

    @torch.no_grad()
    def encode_references(self, images, ref_size: int = 512):
        """VAE latents of the references, each [1, 16, ref_size / 8, ref_size / 8]."""
        device = self._execution_device
        return [encode_image(self.vae, self.image_processor, im, ref_size, device, self.vae.dtype) for im in images]

    @torch.no_grad()
    def generate_multi(self, references, main=None, role_mode: str = "embed", layout: str = "canvas",
                       role_embedding: MainRoleEmbedding | None = None, ref_size: int = 512,
                       height: int = 1024, width: int = 1024, prompt: str | None = None,
                       main_t: float = MAIN_T, main_guidance: float = 1.0, main_guidance_min_sigma: float = 0.0,
                       **kwargs):
        """Generate from 2-4 reference images.

        references:     list of PIL images, in the user's order
        main:           index of the main image in that order (0-based), or None
        role_mode:      "embed" (E_main on the main image), "slot" (main moved first), "index" (named in prompt),
                        "tid" (main image gets position id t = main_t)
        role_embedding: MainRoleEmbedding attached to self.transformer (used with role_mode "embed")
        main_t:         t id of the main image with role_mode "tid"
        main_guidance:  weight w of the main guidance (1 = off; see the module docstring)
        main_guidance_min_sigma: main guidance only at steps with sigma >= this value
        prompt:         override the template prompt (normally left None)
        kwargs:         passed to the pipeline (num_inference_steps, guidance_scale, generator, ...)

        Returns (pipeline output, info) with info = {order, main_pos, prompt}.
        """
        if role_mode not in ROLE_MODES:
            raise ValueError(f"Unknown role mode {role_mode!r}; expected one of {ROLE_MODES}")
        order, main_pos, template = arrange(len(references), main, role_mode, shuffle=False)
        latents = self.encode_references([references[i] for i in order], ref_size)
        prompt = prompt or template
        out = self.generate_from_latents(latents, main_pos, role_mode, layout, role_embedding,
                                         height=height, width=width, prompt=prompt, main_t=main_t,
                                         main_guidance=main_guidance,
                                         main_guidance_min_sigma=main_guidance_min_sigma, **kwargs)
        return out, {"order": order, "main_pos": main_pos, "prompt": prompt}

    @torch.no_grad()
    def generate_from_latents(self, ref_latents, main_pos=None, role_mode: str = "embed", layout: str = "canvas",
                              role_embedding: MainRoleEmbedding | None = None, height: int = 1024, width: int = 1024,
                              main_t: float = MAIN_T, main_guidance: float = 1.0,
                              main_guidance_min_sigma: float = 0.0, **kwargs):
        """Generate from reference latents that are already in model order (e.g. cached latents).

        ref_latents: list of [1, 16, h, w] normalized VAE latents, in model order
        main_pos:    position of the main reference in that order, or None
        main_t:      t id of the main reference with role_mode "tid"
        main_guidance, main_guidance_min_sigma: see generate_multi (needs main_pos and role_mode tid / embed)
        kwargs:      passed to the pipeline: prompt or prompt_embeds + pooled_prompt_embeds, steps, guidance, ...
                     (the prompt must match role_mode and main_pos, see multiref.prompts)
        """
        multiple = self.vae_scale_factor * 2
        target_hw = (height // multiple, width // multiple)
        tokens, ids, spans = build_reference_tokens(ref_latents, layout, target_hw,
                                                    main_pos=marker_pos(role_mode, main_pos), main_t=main_t)
        if role_embedding is not None:
            role_embedding.set_main(spans, main_pos if role_mode == "embed" else None)
        restore = None
        if main_guidance != 1.0 and main_pos is not None:
            if role_mode not in ("tid", "embed"):
                raise ValueError("main guidance needs role mode 'tid' or 'embed'")
            restore = self._wrap_main_guidance(role_mode, spans, main_pos, role_embedding, main_guidance,
                                               main_guidance_min_sigma,
                                               kwargs.get("max_sequence_length", 512))
        self._multiref = (tokens, ids)
        try:
            # max_area must match height * width: the pipeline rescales the output to max_area
            return self(height=height, width=width, max_area=height * width, **kwargs)
        finally:
            self._multiref = None
            if restore is not None:
                restore()
            if role_embedding is not None:
                role_embedding.clear()

    def _wrap_main_guidance(self, role_mode, spans, main_pos, role_embedding, weight, min_sigma,
                            max_sequence_length):
        """Wrap transformer.forward with the main guidance; returns a function that undoes it.

        The stock loop calls the transformer with keyword arguments only (hidden_states, timestep =
        sigma, guidance, pooled_projections, encoder_hidden_states, txt_ids, img_ids, ...). The
        "no main" pass replaces the prompt by the base prompt, sets the t id of every reference
        token to 1 (img_ids: t = 0 for the generated image, 1 or main_t for the references) and
        switches E_main off.
        """
        device = self._execution_device
        none_embeds, none_pooled, none_txt_ids = self.encode_prompt(
            prompt=BASE, prompt_2=None, device=device, num_images_per_prompt=1,
            max_sequence_length=max_sequence_length)
        transformer = self.transformer
        had_instance_forward = "forward" in transformer.__dict__
        orig_forward = transformer.forward

        def guided_forward(*args, **kw):
            out_main = orig_forward(*args, **kw)
            sigma = float(kw["timestep"].flatten()[0])
            if sigma < min_sigma:
                return out_main
            batch = kw["hidden_states"].shape[0]
            dtype = kw["encoder_hidden_states"].dtype
            img_ids = kw["img_ids"].clone()
            img_ids[..., 0] = (img_ids[..., 0] > 0).to(img_ids.dtype)  # references: t = 1
            kw_none = dict(kw, img_ids=img_ids,
                           encoder_hidden_states=none_embeds.to(dtype).repeat(batch, 1, 1),
                           pooled_projections=none_pooled.to(dtype).repeat(batch, 1),
                           txt_ids=none_txt_ids)
            if role_embedding is not None and role_mode == "embed":
                role_embedding.set_main(spans, None)
            try:
                out_none = orig_forward(*args, **kw_none)
            finally:
                if role_embedding is not None and role_mode == "embed":
                    role_embedding.set_main(spans, main_pos)
            if isinstance(out_main, tuple):
                v_main, v_none = out_main[0], out_none[0]
                return (v_none + weight * (v_main - v_none),) + tuple(out_main[1:])
            out_main.sample = out_none.sample + weight * (out_main.sample - out_none.sample)
            return out_main

        transformer.forward = guided_forward

        def restore():
            if had_instance_forward:
                transformer.forward = orig_forward
            else:
                del transformer.forward

        return restore
