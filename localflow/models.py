"""Model registry, pinned by commit.

Every entry pins an exact Hugging Face revision rather than tracking `main`, so
the weights cannot change underneath us between runs.

Format note: CTranslate2 models ship a `model.bin` that is CTranslate2's own
binary tensor format, NOT a Python pickle. Loading it does not deserialize
Python objects, so it does not carry the code-execution risk that a raw
`torch.load` of an untrusted `.bin` checkpoint would.
"""

from __future__ import annotations

from dataclasses import dataclass

# Blackwell (sm_120) note: CTranslate2's int8 path fails with
# CUBLAS_STATUS_NOT_SUPPORTED on RTX 50-series. float16 is the only safe
# compute type here, and on 16GB of VRAM it costs us nothing.
COMPUTE_TYPE = "float16"
DEVICE = "cuda"


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    revision: str
    label: str


REGISTRY: dict[str, ModelSpec] = {
    # Roughly 1.6 GB of VRAM in FP16.
    "large-v3-turbo": ModelSpec(
        repo="deepdml/faster-whisper-large-v3-turbo-ct2",
        revision="4df90f75321148c3a29a9e2351b7ddf8f5b115a8",
        label="Whisper large-v3-turbo (FP16)",
    ),
}

DEFAULT_MODEL = "large-v3-turbo"


def get(key: str = DEFAULT_MODEL) -> ModelSpec:
    if key not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown model {key!r}. Known models: {known}")
    return REGISTRY[key]
