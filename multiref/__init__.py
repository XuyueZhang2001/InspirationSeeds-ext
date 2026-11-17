"""Multi-reference conditioning for FLUX.1 Kontext (2-4 reference images, optional main image).

Modules:
    prompts       fixed prompt templates (no torch)
    data          manifest / cache helpers (no torch)
    conditioning  reference encoding, token ids, main-image role embedding, pipeline (torch + diffusers)

Nothing is imported here, so the torch-free modules can be used without torch installed.
"""
