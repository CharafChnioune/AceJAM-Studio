# AceJAM app

Local app runtime for the Pinokio launcher in the parent folder.

## Flow

1. A local Qwen GGUF composer turns a plain-English description into title, tags, bpm, language, and lyrics.
2. ACE-Step v1.5 generates the audio.
3. The frontend stores optional saved songs under `data/songs/`.

## Runtime Notes

- ACE-Step checkpoints are cached under `model_cache/`.
- Composer GGUF files are cached under `composer_models/`.
- The composer defaults to CPU-first execution so VRAM remains available for ACE-Step.
- `Auto` now escalates long non-instrumental requests to a stronger composer profile so longer songs get fuller lyrics.
- On Apple Silicon, `Auto` now prefers the lighter `acestep-v15-turbo` checkpoint and the `tiny` composer profile for lower latency and memory use.
- On Intel Mac with Python 3.10, install against `torch==2.2.2`, `numpy==1.26.4`, `diffusers==0.31.0`, `numba==0.61.2`, and `vector-quantize-pytorch==1.25.0`, and build `llama-cpp-python` from source; newer wheel lines used on other platforms are not available there.
- On MPS, ACE-Step now auto-selects a lower-precision dtype when supported. Override with `ACE_STEP_DTYPE=auto|bfloat16|float16|float32`.
- The frontend exposes the active ACE-Step song model and lets you switch between `Turbo`, `XL Turbo`, and `Auto`.
- Set `ACE_STEP_MODEL` to override the ACE-Step checkpoint if you want the larger XL model.

## Entry Point

Run `python app.py` from this directory after installing `requirements.txt` plus `llama-cpp-python==0.3.20`.
