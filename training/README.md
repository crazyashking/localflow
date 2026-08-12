# Training a wake word

The app ships with openWakeWord's pretrained phrases (`hey_jarvis`, `alexa`,
`hey_mycroft`). This directory is how a phrase that nobody has trained yet gets
made, and it is entirely separate from running LocalFlow: nothing here is
imported by the app, and none of it is needed to use dictation.

## Why this has its own virtualenv

The training stack pulls in torch, torchaudio, speechbrain, audiomentations and
their trees, and it wants a different numpy than the app is pinned to. The
whole point of the app's `requirements.txt` is that 42 packages install to an
exact SHA256 and nothing else comes along. Installing a training stack into it
would end that, for a capability the app never uses at runtime.

So: `training/.venv` for this, `.venv` for the app. They never meet.

## What you need before starting

| input | what it is | size |
|---|---|---|
| `piper-sample-generator` | GitHub clone plus a voice checkpoint. Synthesises the positive clips, since nobody is recording 30,000 examples by hand. | ~1 GB |
| negative features | openWakeWord's precomputed features over ~2,000 hours of general audio. Teaches the model what the phrase is *not*, which is most of the work. | tens of GB |
| room impulse responses | Real room acoustics, convolved onto the clean synthetic clips. | ~1 GB |
| background audio | Noise and music mixed in during augmentation. AudioSet ships as parquet now, so `prepare_background.py` extracts the audio to 16 kHz WAVs first. | ~3 GB |
| false-positive validation | Hours of speech that must never trigger the model. This is what `target_false_positives_per_hour` is scored against. | several GB |

Everything except the configs and `adapter/` is gitignored.

## The adapter, and why it exists

`training/adapter/generate_samples.py` is copied into the generator clone by
`setup.ps1`. openWakeWord's trainer hardcodes

```python
sys.path.insert(0, os.path.abspath(config["piper_sample_generator_path"]))
from generate_samples import generate_samples
```

which matches dscripka's fork, not the current upstream package. The adapter
bridges the two and, importantly, resamples the output from the voice's native
22050 Hz down to 16 kHz. `augment_clips` raises `ValueError` on anything that
is not already 16 kHz and does no conversion of its own, so without the adapter
the pipeline dies the moment generation finishes. It lives outside the clone
because the clone is gitignored.

## The pipeline

Four stages, run as flags on one command:

```powershell
training\.venv\Scripts\python.exe training\train.py `
    --training_config training\configs\hey_flow.yml `
    --generate_clips --augment_clips --train_model
```

Use `training\train.py`, not `-m openwakeword.train`. openWakeWord 0.6.0 was
written against older versions of its own dependencies and cannot finish a run
on Windows without five patches: speechbrain's lazy modules break torch's op
registration, torchaudio no longer decodes anything without FFmpeg, `trim_mmap`
unlinks a file it still has mapped, the training dataloader ships lambdas to
spawned workers, and the ONNX export splits weights into a sibling file. Each
is documented at the point it is applied.

1. **generate_clips** synthesises the positive examples across many voices,
   speeds and pitches. The long stage, and the one that uses the GPU.
2. **augment_clips** convolves room impulse responses and mixes background
   noise over them. Skipping this produces a model that works perfectly on
   clean audio and falls apart next to a desk fan.
3. **compute features** turns audio into the melspectrogram and embedding
   representation the classifier actually sees.
4. **train_model** fits the classifier head and exports ONNX.

## Installing the trained model

Drop the exported `.onnx` into `models/wake/custom/` and point `wake_phrase` at
its name in `settings.json`:

```json
{ "wake_word": true, "wake_phrase": "hey_flow" }
```

`localflow/wake.py` resolves an unknown phrase against that directory. A model
trained here carries no pinned digest, unlike the downloaded ones, because
there is no upstream release for it to be checked against.

## Measuring it, which is the part that matters

A wake word is only as good as its false-accept rate, and a model that scores
well on its own validation split can still fire twice an hour in your actual
room. Measure it where it will be used:

```powershell
.\.venv\Scripts\python.exe bench\demo_wake.py --minutes 60 --phrase hey_flow
```

Leave it running through a normal working hour without ever saying the phrase.
Anything it reports is a false accept. Only a number from that run belongs in
the main README.

## The two phrases here

`hey_flow.yml` is the phrase we want. `hey_localflow.yml` is the control.

"hey flow" is two syllables and "flow" is a common word with a very common
vowel shape, so it sits near "hello", "hey Joe" and "cash flow". "hey
localflow" is four syllables and much further from ordinary English. Rather
than settle that by argument, both are trained on identical settings and
measured on identical audio, and the loser is deleted.

The confusables for each phrase are listed in `custom_negative_phrases` in its
config. That list is the direct fix for the problem above: every entry is
something the model is explicitly taught to reject.
