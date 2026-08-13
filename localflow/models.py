"""Model registry and compute device selection, both pinned.

Every registry entry pins an exact Hugging Face revision rather than tracking
`main`, so the weights cannot change underneath us between runs.

Format note: CTranslate2 models ship a `model.bin` that is CTranslate2's own
binary tensor format, NOT a Python pickle. Loading it does not deserialize
Python objects, so it does not carry the code-execution risk that a raw
`torch.load` of an untrusted `.bin` checkpoint would.

On device selection: the GPU is the fast path and stays the default wherever it
is usable. The CPU path exists so the project runs at all on a machine without
an NVIDIA card, and it is not a drop-in substitute. A CPU decodes Whisper
roughly two orders of magnitude slower than a modern GPU does, so choosing the
CPU also means choosing a smaller model. resolve_device() and pick_model()
below are the two halves of that decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import cuda

# Whisper's native input rate. It lives here rather than in audio.py because the
# model is what dictates it; the recorder captures at this rate to satisfy the
# model, which is what lets the numpy array go straight into faster-whisper with
# no resampling and no ffmpeg dependency. Keeping it in this module also means
# nothing has to import the microphone layer just to know the number.
SAMPLE_RATE = 16000

# Blackwell (sm_120) note: CTranslate2's int8 path fails with
# CUBLAS_STATUS_NOT_SUPPORTED on RTX 50-series. float16 is the only safe compute
# type on the GPU, and on 16GB of VRAM it costs us nothing.
CUDA_COMPUTE_TYPE = "float16"

# CTranslate2 has no float16 kernels for the CPU, so asking for it there gets
# silently promoted to float32 and runs at half speed for no benefit. int8 is
# the quantised path CTranslate2 is built around and is roughly 3x faster than
# float32 on the same core count. The accuracy cost is small and the speed is
# the difference between usable and not.
CPU_COMPUTE_TYPE = "int8"


@dataclass(frozen=True)
class Device:
    """Where inference runs, and why that was chosen."""

    name: str  # "cuda" or "cpu", as CTranslate2 spells them
    compute_type: str
    reason: str

    @property
    def is_gpu(self) -> bool:
        return self.name == "cuda"

    def summary(self) -> str:
        return f"{self.name} ({self.compute_type})"


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    revision: str
    label: str
    # English-only checkpoints are the small ones worth running on a CPU, and
    # they produce nonsense for any other language. pick_model() checks this
    # before selecting one automatically.
    english_only: bool = False


REGISTRY: dict[str, ModelSpec] = {
    # Roughly 1.6 GB of VRAM in FP16. The GPU default.
    "large-v3-turbo": ModelSpec(
        repo="deepdml/faster-whisper-large-v3-turbo-ct2",
        revision="4df90f75321148c3a29a9e2351b7ddf8f5b115a8",
        label="Whisper large-v3-turbo (FP16)",
    ),
    # The CPU defaults. small is the smallest checkpoint whose output still
    # reads as clean dictation; base trades noticeably more accuracy for about
    # three times the speed, and is there for machines where small is too slow.
    "small.en": ModelSpec(
        repo="Systran/faster-whisper-small.en",
        revision="d1d751a5f8271d482d14ca55d9e2deeebbae577f",
        label="Whisper small.en",
        english_only=True,
    ),
    "small": ModelSpec(
        repo="Systran/faster-whisper-small",
        revision="536b0662742c02347bc0e980a01041f333bce120",
        label="Whisper small (multilingual)",
    ),
    "base.en": ModelSpec(
        repo="Systran/faster-whisper-base.en",
        revision="3d3d5dee26484f91867d81cb899cfcf72b96be6c",
        label="Whisper base.en",
        english_only=True,
    ),
    "base": ModelSpec(
        repo="Systran/faster-whisper-base",
        revision="ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
        label="Whisper base (multilingual)",
    ),
}

# What "auto" resolves to on each device. The GPU is fast enough that there is
# no reason to run anything but the best checkpoint; the CPU is not.
AUTO = "auto"
CUDA_MODEL = "large-v3-turbo"
CPU_MODEL = "small"

DEFAULT_MODEL = AUTO


def resolve_device(preference: str = AUTO) -> Device:
    """Decide where inference runs.

    `preference` is the `device` setting: "auto", "cuda" or "cpu". "auto" takes
    the GPU whenever it is genuinely usable and falls back to the CPU otherwise.
    "cuda" is fatal if the GPU is not available, because a user who asked for it
    explicitly needs to hear why they did not get it rather than quietly running
    a hundred times slower.
    """
    preference = (preference or AUTO).lower()
    if preference not in (AUTO, "cuda", "cpu"):
        raise ValueError(
            f"Unknown device {preference!r}. Use \"auto\", \"cuda\" or \"cpu\"."
        )

    if preference == "cpu":
        return Device("cpu", CPU_COMPUTE_TYPE, "cpu requested in settings")

    problem, count = _probe_cuda()

    if problem is None and count > 0:
        return Device("cuda", CUDA_COMPUTE_TYPE, f"{count} CUDA device(s) visible")

    detail = problem or "CTranslate2 reports no CUDA device"
    if preference == "cuda":
        raise RuntimeError(
            f"device is set to \"cuda\" in settings.json, but the GPU is not usable:\n"
            f"{detail}\n"
            "Set it to \"auto\" to fall back to the CPU, or fix the GPU install."
        )
    return Device("cpu", CPU_COMPUTE_TYPE, f"no usable GPU ({detail.splitlines()[0]})")


def _probe_cuda() -> tuple[str | None, int]:
    """Return (why the GPU is unusable or None, number of CUDA devices).

    Both halves are needed. cuda.problem() catches a machine with an NVIDIA card
    whose CUDA libraries are missing, where CTranslate2 still counts the device
    and then dies on the first encode. The device count catches the opposite: a
    complete library install on a machine with no card in it, which happens
    whenever someone installs the GPU requirements out of habit.
    """
    problem = cuda.problem()  # also registers the DLL dirs, before the import
    if problem:
        return problem, 0

    try:
        import ctranslate2
    except ImportError as exc:  # pragma: no cover - ctranslate2 is a hard dep
        return f"could not import ctranslate2: {exc}", 0

    try:
        return None, ctranslate2.get_cuda_device_count()
    except Exception as exc:
        # Querying the device count goes through the CUDA driver, which throws
        # rather than returning zero when it is absent or too old.
        return f"CUDA driver unavailable: {exc}", 0


def pick_model(key: str = AUTO, device: Device | None = None, language: str = "en") -> str:
    """Resolve a `model` setting, which may be "auto", to a registry key."""
    if key and key != AUTO:
        if key not in REGISTRY:
            known = ", ".join(sorted(REGISTRY))
            raise KeyError(f"Unknown model {key!r}. Known models: {known}")
        if REGISTRY[key].english_only and language != "en":
            # Not fatal, because it is an explicit choice and the model will
            # still produce output. It will be a bad English transcription of
            # another language, which is confusing enough to be worth naming.
            print(
                f"[models] {key} is English-only but language is {language!r}. "
                f"Use {key.removesuffix('.en')!r} for other languages."
            )
        return key

    device = device or resolve_device()
    if device.is_gpu:
        return CUDA_MODEL
    if language == "en":
        # The English-only checkpoint of the same size is both faster and more
        # accurate on English, which is the whole reason it exists.
        return f"{CPU_MODEL}.en"
    return CPU_MODEL


def get(key: str = AUTO, device: Device | None = None, language: str = "en") -> ModelSpec:
    return REGISTRY[pick_model(key, device, language)]
