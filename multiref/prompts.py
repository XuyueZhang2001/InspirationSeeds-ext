"""Fixed prompt templates.

The generator only ever sees these templates; free-form user text is mapped to
"main image k" or "no main image" before generation.

Role modes (how the main image is designated):
    embed   main image marked by the learnable role embedding E_main; order unchanged
    slot    main image moved to the first position
    index   main image named by its position in the prompt ("image k"); no embedding
    tid     main image keeps its position; its tokens get position id t = 2 instead of 1
            (multiref.conditioning.build_reference_tokens). Uses the "embed" sentence
            ("the marked image"), so it shares the cached text embedding "main_embed".
            Without a main image every reference has t = 1, so no position is special.
"""

ROLE_MODES = ("embed", "slot", "index", "tid")
# role modes whose main-image sentence is the "embed" one (same cached text embedding)
MARKED_MODES = ("embed", "tid")
MAX_REFS = 4

BASE = "Combine the elements in the reference images into a single object inspired by them."
MAIN = {
    "embed": "Use the marked image as the main object and the other images as inspiration.",
    "slot": "Use the first image as the main object and the other images as inspiration.",
    "index": "Use image {k} as the main object and the other images as inspiration.",
}


def prompt_key(main_pos, role_mode: str) -> str:
    """Key of the template for a main position (0-based, or None for no main image)."""
    if role_mode not in ROLE_MODES:
        raise ValueError(f"Unknown role mode {role_mode!r}; expected one of {ROLE_MODES}")
    if main_pos is None:
        return "base"
    if role_mode == "index":
        return f"main_index_{main_pos + 1}"
    if role_mode in MARKED_MODES:
        return "main_embed"
    return f"main_{role_mode}"


def render_prompt(main_pos, role_mode: str) -> str:
    """Prompt for a main position (0-based, or None for no main image)."""
    key = prompt_key(main_pos, role_mode)
    if key == "base":
        return BASE
    if role_mode == "index":
        return f"{BASE} {MAIN['index'].format(k=main_pos + 1)}"
    if role_mode in MARKED_MODES:
        return f"{BASE} {MAIN['embed']}"
    return f"{BASE} {MAIN[role_mode]}"


def all_prompts(max_refs: int = MAX_REFS) -> dict:
    """Every template that can occur, keyed by prompt_key (used to precompute text embeddings)."""
    out = {"base": BASE}
    for mode in ("embed", "slot"):
        out[prompt_key(0, mode)] = render_prompt(0, mode)
    for k in range(max_refs):
        out[prompt_key(k, "index")] = render_prompt(k, "index")
    return out


def arrange(n: int, main, role_mode: str, shuffle: bool = False, rng=None):
    """Order in which the references enter the model, and the matching prompt.

    n:     number of references
    main:  index of the main reference in the original order (0-based), or None
    shuffle: shuffle the order first (training); rng is a random.Random

    Returns (order, main_pos, prompt):
        order     original indices in model order
        main_pos  position of the main reference in model order (None if no main)
    """
    order = list(range(n))
    if shuffle:
        rng.shuffle(order)
    if main is not None and role_mode == "slot":
        order.remove(main)
        order.insert(0, main)
    main_pos = order.index(main) if main is not None else None
    return order, main_pos, render_prompt(main_pos, role_mode)
