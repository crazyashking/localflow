# Set up the wake word training environment.
#
# Separate from the app in every way: its own virtualenv, its own torch, and
# its own data directory. Nothing here is imported by LocalFlow at runtime.
#
#   .\training\setup.ps1              # everything except the 16 GB file
#   .\training\setup.ps1 -Full        # including it
#
# Sizes below were measured with HEAD requests on 2026-08-11, not estimated.

param([switch]$Full)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root "training\.venv\Scripts\python.exe"
$data = Join-Path $root "training\data"

# --- 1. virtualenv ----------------------------------------------------------
if (-not (Test-Path $py)) {
    Write-Host "creating training venv (3.14)"
    py -3.14 -m venv (Join-Path $root "training\.venv")
}

Write-Host "installing the training stack"
& $py -m pip install --quiet --upgrade pip
# `acoustics` is missing from openwakeword's install metadata even though
# data.py imports it at module scope, so training dies on import without it.
# It is used for exactly one call: acoustics.generator.noise(), which makes the
# coloured noise mixed in during augmentation.
# onnxscript is not declared by anything either, and is only reached at the
# very end of a run: torch 2.13's ONNX exporter imports it, so without it the
# training finishes and then dies before writing the model.
#
# deep-phonemizer is needed only for a phrase containing a word the CMU
# pronunciation dictionary does not have. openWakeWord builds its adversarial
# negatives from phoneme overlap and falls back to DeepPhonemizer for any
# out-of-vocabulary word. "hey flow" is two real words and never reaches it;
# "hey localflow" dies with ModuleNotFoundError: No module named 'dp' after
# the positives are already generated.
& $py -m pip install --quiet torch torchaudio torchinfo torchmetrics `
    speechbrain audiomentations torch-audiomentations pronouncing mutagen `
    openwakeword datasets acoustics onnxscript deep-phonemizer

# acoustics 0.2.6 (the newest release) imports scipy.special.sph_harm, which
# scipy removed in 1.17. The import happens in acoustics/__init__.py by way of
# a submodule openWakeWord never touches, so the whole package is unusable on
# current scipy. Pinning below 1.17 here is safe because this venv exists only
# to train; the app keeps its own pinned scipy and is untouched.
& $py -m pip install --quiet "scipy<1.17"

# Two packages are deliberately absent.
#
# webrtcvad is listed in piper-sample-generator's pyproject but appears nowhere
# in its source, and it has no cp314 wheel.
#
# torchcodec is what torchaudio 2.13 delegates decoding to. Its wheel installs
# on Windows but its DLLs will not load without FFmpeg shared libraries on the
# system, which is an unpinnable dependency. training/train.py routes audio
# loading through soundfile instead; see patch 2 in that file.

# CUDA build, installed second so it replaces the CPU wheel PyPI serves on
# Windows. This is not optional: generating tens of thousands of clips on CPU
# takes many hours, and minutes on the GPU. cu130 is the only channel carrying
# torch 2.13, and Blackwell (sm_120, the RTX 50 series) needs cu128 or newer.
Write-Host "replacing CPU torch with the CUDA build (about 3 GB)"
& $py -m pip install --quiet --upgrade --index-url https://download.pytorch.org/whl/cu130 `
    torch torchaudio
& $py -c "import torch; print('  torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

# --- 2. piper sample generator ----------------------------------------------
$piper = Join-Path $root "training\piper-sample-generator"
if (-not (Test-Path $piper)) {
    Write-Host "cloning piper-sample-generator (default branch is master)"
    git clone --depth 1 https://github.com/rhasspy/piper-sample-generator $piper
}
& $py -m pip install --quiet piper-tts

# openWakeWord's train.py does `from generate_samples import generate_samples`
# after putting piper_sample_generator_path on sys.path, which only works
# against dscripka's fork layout. training/adapter/generate_samples.py bridges
# the two and resamples the output to 16 kHz; read its docstring for why both
# are necessary. The clone is gitignored, so the adapter is kept outside it and
# copied in, otherwise a fresh checkout would silently lose it.
Write-Host "installing the generate_samples adapter into the clone"
Copy-Item (Join-Path $root "training\adapter\generate_samples.py") `
    (Join-Path $piper "generate_samples.py") -Force

# The generator accepts either a .pt checkpoint or a Piper .onnx voice. The
# openWakeWord notebook uses the .pt, so that is what is fetched here.
$ckpt = Join-Path $piper "models\en_US-libritts_r-medium.pt"
if (-not (Test-Path $ckpt)) {
    Write-Host "downloading voice checkpoint (204,089,915 bytes)"
    curl.exe -sL -o $ckpt `
        "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt"
}

# --- 3. training data --------------------------------------------------------
New-Item -ItemType Directory -Force -Path $data | Out-Null

# train.py calls os.mkdir on output_dir, which is not recursive, so it fails
# with WinError 3 unless the parent exists. It creates the per-model directory
# under this one itself.
New-Item -ItemType Directory -Force -Path (Join-Path $root "training\output") | Out-Null

Write-Host "fetching impulse responses and validation features"
& $py -c @"
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download('davidscripka/MIT_environmental_impulse_responses',
                  repo_type='dataset', local_dir=r'$data\rirs')
hf_hub_download('davidscripka/openwakeword_features', 'validation_set_features.npy',
                repo_type='dataset', local_dir=r'$data')
"@

# Background audio. The openWakeWord notebook points at data/bal_train09.tar,
# which now returns HTTP 404: the dataset was reconverted to parquet and lives
# under data/bal_train/NN.parquet, 38 shards totalling 24.1 GB. The audio is
# bytes inside a column, so it has to be extracted to real files before
# torch_audiomentations can see it. prepare_background.py downloads and
# extracts five shards, the closest equivalent to the one tar the notebook used.
Write-Host "fetching and extracting AudioSet background audio (about 3.2 GB)"
& $py (Join-Path $root "training\prepare_background.py")

# openWakeWord's own melspectrogram and embedding models, which the feature
# stage loads from inside the package. They are not installed with the wheel,
# so training dies with NO_SUCHFILE without this. Separate from the app's copies
# under models/wake, which carry pinned digests.
Write-Host "downloading openwakeword's feature models"
& $py -c "import openwakeword.utils as u; u.download_models()"

if ($Full) {
    # 16.093 GiB. Memory-mapped during training rather than read into RAM, so
    # this costs disk and not memory. There is no smaller official version.
    Write-Host "fetching negative features (16.1 GiB, this takes a while)"
    & $py -c @"
from huggingface_hub import hf_hub_download
hf_hub_download('davidscripka/openwakeword_features',
                'openwakeword_features_ACAV100M_2000_hrs_16bit.npy',
                repo_type='dataset', local_dir=r'$data')
"@
} else {
    Write-Host "skipping the 16.1 GiB negative features. Re-run with -Full for them."
}

# training/train.py, never `-m openwakeword.train`. It applies six patches the
# run cannot complete without on Windows; each one is documented where it is
# defined.
Write-Host "`nsetup done. Train with:"
Write-Host "  training\.venv\Scripts\python.exe training\train.py ``"
Write-Host "      --training_config training\configs\hey_flow.yml ``"
Write-Host "      --generate_clips --augment_clips --train_model"
