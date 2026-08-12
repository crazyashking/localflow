"""Run openWakeWord's trainer, with six upstream incompatibilities patched.

Use this instead of `python -m openwakeword.train`. Arguments pass straight
through:

    training\\.venv\\Scripts\\python.exe training\\train.py \\
        --training_config training\\configs\\hey_flow.yml \\
        --generate_clips --augment_clips --train_model

Every patch here exists because openWakeWord 0.6.0 was written against older
versions of its own dependencies, and against Linux. None of them change what
the trainer computes. Without all six a run on Windows cannot reach the end:
each one was found by hitting it.

The first three are explained below. Patch 4 (the dataloader cannot ship
lambdas to spawned workers), patch 5 (the ONNX export splits weights into a
sibling file) and patch 6 (the phonemizer checkpoint will not load under
weights_only) are explained where they are defined.

--- 1. speechbrain's lazy modules break torch's op registration -------------

Without the guard, the run dies during augmentation with

    ImportError: Lazy import of LazyModule(
        package=None, target=speechbrain.integrations.k2_fsa, loaded=False) failed

which names neither library actually at fault. The chain:

  1. speechbrain 1.1.0 puts LazyModule placeholders in sys.modules. Their
     __getattr__ performs the real import on first attribute access.
  2. torch 2.13 registers custom ops (torch.distributed.tensor,
     torch.distributed.fsdp, and others) and records each one's source
     location while doing so.
  3. Recording it calls inspect.getmodule, which walks every entry in
     sys.modules doing hasattr(module, "__file__").
  4. That hasattr reaches a LazyModule, which tries to import
     speechbrain.integrations.k2_fsa. k2 has no Windows build and is not
     installed, so it raises.
  5. speechbrain re-raises as ImportError instead of letting hasattr answer
     False, so it escapes introspection and kills the run.

The bug is step 5: an object that cannot answer __file__ should raise
AttributeError, which is what hasattr is built to absorb. Pre-importing the
torch modules in a safe order was the first attempt and it does not hold,
because torch registers ops from several modules and any one of them loaded
after speechbrain trips the same wire. Guarding dunder access fixes the class
of bug rather than one instance, and ordinary lazy imports still work.

--- 2. torchaudio 2.13 has no built-in decoders -----------------------------

torchaudio.load now delegates to torchcodec, which needs FFmpeg shared
libraries present on the system. Its wheel installs on Windows but its DLLs
fail to load without them:

    OSError: Could not load this library: ...libtorchcodec_core4.dll

Requiring a system FFmpeg install would put an unpinnable dependency in the
middle of an otherwise reproducible setup. Every file this pipeline touches
is a WAV (the generated clips, the impulse responses, the extracted AudioSet
background), and soundfile reads those directly and is already required by
speechbrain and librosa. So load and info are routed through it instead.
Both openWakeWord and torch_audiomentations call these as module attributes,
so patching here covers both.

--- 3. trim_mmap unlinks a file that is still mapped ------------------------

openWakeWord's trim_mmap finishes with os.remove followed by os.rename. POSIX
allows unlinking a file that still has an open mapping; Windows does not, so
the run ends with

    PermissionError: [WinError 32] The process cannot access the file because
    it is being used by another process: ...positive_features_train.npy

Two mappings are open at that moment: the one trim_mmap makes itself, and the
`fp` its caller in compute_features_from_generator is still holding. The
caller's is a local in a frame further up the stack, so open_memmap is wrapped
to record what it hands out and the replacement closes both before touching
the filesystem.

The replacement also fixes two smaller faults in the original:
str.strip(".npy") strips *characters*, so the temp path for
"positive_features_train.npy" came out as "positive_features_trai2.npy" with
the trailing n of "train" eaten, and the scan for the last non-empty row runs
off the front of the array if every row is zero.
"""

import os
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import openwakeword.data
import openwakeword.utils
import soundfile as sf
import torch
import torchaudio
from numpy.lib.format import open_memmap
from speechbrain.utils.importutils import LazyModule
from tqdm import tqdm

# --- patch 1 ----------------------------------------------------------------

_original_getattr = LazyModule.__getattr__


def _guarded_getattr(self, attr):
    """Answer introspection without triggering the import behind the module."""
    if attr.startswith("__") and attr.endswith("__"):
        raise AttributeError(attr)
    return _original_getattr(self, attr)


LazyModule.__getattr__ = _guarded_getattr

# --- patch 2 ----------------------------------------------------------------


def _sf_info(uri, *_args, **_kwargs):
    info = sf.info(str(uri))
    return SimpleNamespace(
        sample_rate=info.samplerate,
        num_frames=info.frames,
        num_channels=info.channels,
    )


def _sf_load(uri, frame_offset=0, num_frames=-1, normalize=True,
             channels_first=True, **_kwargs):
    # soundfile and torchaudio agree that -1 frames means "to the end".
    data, sample_rate = sf.read(
        str(uri), start=frame_offset, frames=num_frames,
        dtype="float32", always_2d=True,
    )
    tensor = torch.from_numpy(np.ascontiguousarray(data.T))  # [channels, samples]
    return (tensor if channels_first else tensor.T), sample_rate


torchaudio.info = _sf_info
torchaudio.load = _sf_load

# --- patch 3 ----------------------------------------------------------------

_live_maps: dict[str, np.memmap] = {}
_original_open_memmap = openwakeword.utils.open_memmap


def _tracked_open_memmap(filename, *args, **kwargs):
    array = _original_open_memmap(filename, *args, **kwargs)
    _live_maps[os.path.abspath(filename)] = array
    return array


def _close_map(array) -> None:
    """Release the file handle behind a memmap, if it still holds one."""
    handle = getattr(array, "_mmap", None)
    if handle is not None and not handle.closed:
        handle.close()


def _trim_mmap(mmap_path) -> None:
    """Drop all-zero trailing rows. Windows-safe rewrite of the original."""
    mmap_path = os.path.abspath(mmap_path)
    source = np.load(mmap_path, mmap_mode="r")

    last = -1
    while last >= -source.shape[0] and np.all(source[last, :, :] == 0):
        last -= 1
    n_new = source.shape[0] + last + 1

    temp_path = mmap_path.removesuffix(".npy") + ".trimmed.npy"
    dest = open_memmap(temp_path, mode="w+", dtype=np.float32,
                       shape=(n_new, source.shape[1], source.shape[2]))
    for i in tqdm(range(0, n_new, 1024), total=max(1, n_new // 1024),
                  desc="Trimming empty rows"):
        dest[i:i + 1024] = source[i:i + 1024]
    dest.flush()

    _close_map(dest)
    _close_map(source)
    tracked = _live_maps.pop(mmap_path, None)
    if tracked is not None:
        tracked.flush()
        _close_map(tracked)
    _live_maps.pop(os.path.abspath(temp_path), None)

    os.remove(mmap_path)
    os.rename(temp_path, mmap_path)


openwakeword.utils.open_memmap = _tracked_open_memmap
openwakeword.data.trim_mmap = _trim_mmap

# --- patch 4 ----------------------------------------------------------------


class _SingleProcessDataLoader(torch.utils.data.DataLoader):
    """A DataLoader that never spawns workers.

    The training loader is built with num_workers=os.cpu_count()//2, which
    means Windows has to pickle the dataset and send it to each worker. It
    cannot:

        PicklingError: Can't pickle <function <lambda>>: it's not found as
        __main__.<lambda>

    The label transforms are lambdas, and the data transforms are functions
    defined inside a module executed through runpy, so neither can be looked
    up by qualified name in a spawned interpreter. Linux forks instead of
    pickling, which is why upstream never sees this.

    Loading single-process avoids the transfer entirely. The batch generator
    only slices memory-mapped arrays, so there is little for workers to
    overlap with in the first place.
    """

    def __init__(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs.pop("prefetch_factor", None)  # rejected outright when workers is 0
        super().__init__(*args, **kwargs)


torch.utils.data.DataLoader = _SingleProcessDataLoader

# --- patch 5 ----------------------------------------------------------------

_original_onnx_export = torch.onnx.export


def _single_file_onnx_export(*args, **kwargs):
    """Keep the weights inside the .onnx instead of beside it.

    torch 2.13 defaults to external_data=True, which splits the weights into a
    sibling .onnx.data file. onnxruntime then only loads the model when that
    file happens to be adjacent, so copying just the .onnx into
    models/wake/custom/ would produce a model that fails at load time.
    """
    kwargs.setdefault("external_data", False)
    return _original_onnx_export(*args, **kwargs)


torch.onnx.export = _single_file_onnx_export

# --- patch 6 ----------------------------------------------------------------

_PHONEMIZER_CHECKPOINT = "en_us_cmudict_forward"
_original_torch_load = torch.load


def _torch_load(f, *args, **kwargs):
    """Load the DeepPhonemizer checkpoint, which is not a bare state dict.

    Reached only by a phrase containing a word the CMU dictionary does not
    have, so "hey flow" never gets here and "hey localflow" always does.
    openWakeWord downloads the checkpoint itself and calls
    Phonemizer.from_checkpoint, which torch 2.6+ refuses:

        UnpicklingError: Unsupported global: GLOBAL
        dp.preprocessing.text.Preprocessor was not an allowed global by default

    weights_only=True is the right default and stays the default. It is
    relaxed for this one filename, which openWakeWord fetches over HTTPS from
    a fixed URL into its own package directory.
    """
    if _PHONEMIZER_CHECKPOINT in str(f):
        kwargs["weights_only"] = False
    return _original_torch_load(f, *args, **kwargs)


torch.load = _torch_load

if __name__ == "__main__":
    sys.argv[0] = "openwakeword.train"
    try:
        runpy.run_module("openwakeword.train", run_name="__main__", alter_sys=True)
    except ModuleNotFoundError as exc:
        # The very last statement of openwakeword/train.py converts the freshly
        # written .onnx to tflite through onnx_tf, which is unmaintained, needs
        # TensorFlow, and has no CPython 3.14 build. LocalFlow loads the .onnx,
        # so the model is complete before this line is reached. Anything else
        # missing is a real failure and still propagates.
        if exc.name != "onnx_tf":
            raise
        print("\nSkipped the tflite conversion: onnx_tf is unmaintained and needs "
              "TensorFlow. The .onnx is written and is what LocalFlow loads.")
