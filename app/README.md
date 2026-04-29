# AceJAM App Runtime

This folder contains the self-contained AceJAM web app runtime. Pinokio launcher scripts live one level above this folder.

## Runtime Layout

- `app.py`: FastAPI/Gradio server and public API routes.
- `index.html`: Studio UI.
- `studio_core.py`: shared validation, task/model compatibility, and request helpers.
- `songwriting_toolkit.py`: album agent toolbelt for model advice, tags, lyric length, rhyme/flow, hooks, metaphors, cliche checks, and fallback album planning.
- `official_runner.py`: isolated official ACE-Step 1.5 inference bridge for LM/CoT, output formats, and post-processing controls.
- `lora_trainer.py`: official ACE-Step trainer job manager.
- `acestep/`: local ACE-Step inference runtime used by AceJAM generation.
- `vendor/ACE-Step-1.5/`: official ACE-Step 1.5 trainer clone created by Pinokio install.
- `data/`: uploads, generated results, local library, LoRA datasets, tensor outputs, training jobs, and adapters.
- `model_cache/checkpoints/`: ACE-Step DiT, VAE, text encoder, and LM checkpoints.

## Clean Studio UI

`index.html` keeps the main workflow visible and hides advanced controls by default. Custom shows song, lyrics, reference audio, BPM, key, and time first; Album shows concept, track count, duration, language, model strategy, and planning/generation actions first; Trainer shows dataset flow first. Advanced inference, LM/CoT, output, post-processing, album craft/tooling, and training hyperparameters stay in closed disclosure panels with modified badges when values differ from defaults.

## Model Advice Runtime

`studio_core.py` owns the model advice layer:

- `MODEL_PROFILES` covers all known ACE-Step 1.5 DiT checkpoints.
- `LM_MODEL_PROFILES` covers `auto`, `none`, and the 0.6B/1.7B/4B 5Hz LM choices.
- `/api/config` returns `model_profiles`, `lm_model_profiles`, `recommended_song_model`, `recommended_lm_model`, installed model lists, per-model capabilities, official runner status, and the Custom `ui_schema`.
- Unknown local `acestep-v15-*` folders get a fallback profile inferred from the model name, so locally discovered checkpoints still render safely.
- `acestep-v15-turbo-rl` is listed as an official-but-unreleased checkpoint. It is visible in parity/model advice as not downloadable until ACE-Step publishes it.

The UI renders this metadata in the model dropdown, workspace model guide, Custom model advice panel, Simple LM selector, Custom LM selector, and Album LM selector. Unsupported task/model combinations remain disabled through the same compatibility matrix used by the generation API. Selecting a missing known model automatically calls `/api/models/download`, shows queued/running/failed/installed state in the advice panel, and reloads config after the checkpoint lands in `model_cache/checkpoints`.

`GET /api/ace-step/parity` exposes the full official parity manifest used by Settings: official sources, DiT/LM models, tasks, endpoints, `GenerationParams`, `GenerationConfig`, runtime controls, trainer features, and per-item status (`supported`, `guarded`, `missing`, `unreleased`, `not_applicable`). `/api/status` and `/api/config` also expose UI/backend hashes so stale browser sessions can prompt a refresh after restart.

## Custom Generation Runtime

`/api/generate_advanced` keeps the fast local handler for standard runs. It routes to `official_runner.py` only when a request uses official-only controls:

- LM/CoT: `thinking`, `sample_mode`, `use_format`, LM temperature/CFG/top-k/top-p, CoT toggles, constrained decoding.
- Output/post: `mp3`, `opus`, `aac`, `wav32`, MP3 options, normalization, fades, latent shift/rescale.
- Edit extras: cover noise, repaint crossfades, chunk mask mode, repaint mode/strength, sampler mode, velocity controls.

The official bridge runs in a subprocess with `PYTHONPATH` pointed at `vendor/ACE-Step-1.5`, avoiding namespace conflicts with local `app/acestep`. Missing vendor files, missing models, missing LM checkpoints, and unsupported output combinations return explicit errors.

`official_runner.py` also handles official ACE-Step LM helper actions for `create_sample`, `format_sample`, and `understand_music` without loading the DiT model first. `/api/create_sample` and `/api/format_sample` keep the Ollama songwriter flow by default, while official-compatible endpoints and explicit API requests can use the 5Hz LM.

If the UI is opened as `file://.../app/index.html`, it shows a runtime warning and disables generation actions. The working entrypoint is the Pinokio-served HTTP URL.

## Album Agent Runtime

Album mode is now a two-stage agent studio:

1. `GET /api/songwriting_toolkit` returns the available craft tools, ACE-Step-style tag taxonomy, lyric meta tags, density presets, model strategies, installed model list, and artist-reference policy.
2. `POST /api/album/plan` builds an editable track plan with title, caption/tags, full lyrics, BPM, key, time signature, model, and tool reports.
3. `POST /api/generate_album` accepts the edited plan and sends each track through `_run_advanced_generation` with `task_type=text2music`, model choice, lyrics, BPM/key/time metadata, inference controls, variants, score/LRC/audio-code flags, and rich library metadata.

The model advisor never silently swaps to a different checkpoint. Album final render defaults to `xl_sft_final`, which locks every track to `acestep-v15-xl-sft`. If XL SFT is missing, the UI/API starts a download and retries with a fresh payload instead of rendering with a cheaper fallback. `best_installed`, `maximum_detail`, and `selected` remain available for planning/previews or direct API use.

Album planning now uses AceJAM Agents by default: direct local LLM calls for album bible, per-track writer, finalizer, and quality repair, with deterministic Python gates for tag coverage, caption integrity, lyric duration fit, and contract enforcement before audio rendering. The deterministic toolbelt remains available only when `toolbelt_only=true`.

## Trainer Flow

1. Pick a dataset folder with the browser/Finder folder picker.
2. Enter the trigger tag and fixed training language.
3. Click **Start training**.
4. AceJAM imports the files into `data/lora_imports/<dataset_id>/`, labels sidecar-first, saves the official dataset JSON, preprocesses tensors, trains LoRA by default, registers the final adapter, and tries to auto-load it for generation.
5. Advanced users can still edit labels, run preprocess/train separately, or switch to LoKr.

The trainer runs as a subprocess with `PYTHONPATH` pointed at `vendor/ACE-Step-1.5`, so the official trainer package does not collide with the local inference package.

## API Examples

Runtime status:

```bash
curl http://127.0.0.1:7860/api/status
curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7860/v1/models
```

Create a structured sample:

```bash
curl -X POST http://127.0.0.1:7860/api/create_sample \
  -H 'Content-Type: application/json' \
  -d '{"description":"Dutch club rap with bright synth hooks","duration":60}'
```

Generate a batch:

```bash
curl -X POST http://127.0.0.1:7860/api/generate_advanced \
  -H 'Content-Type: application/json' \
  -d '{
    "task_type":"text2music",
    "caption":"club rap, bright synth hook, punchy drums",
    "lyrics":"[Verse]\nWe move fast...\n\n[Chorus]\nLight it up...",
    "duration":60,
    "bpm":128,
    "key_scale":"C minor",
    "time_signature":"4",
    "batch_size":2,
    "song_model":"acestep-v15-turbo",
    "save_to_library":true
  }'
```

Official LM/output example:

```bash
curl -X POST http://127.0.0.1:7860/api/generate_advanced \
  -H 'Content-Type: application/json' \
  -d '{
    "task_type":"text2music",
    "caption":"warm cinematic pop, detailed drums, intimate vocal",
    "lyrics":"[Verse]\n...",
    "duration":90,
    "thinking":true,
    "ace_lm_model":"acestep-5Hz-lm-1.7B",
    "lm_temperature":0.85,
    "audio_format":"mp3",
    "mp3_bitrate":"192k"
  }'
```

Official-compatible status and parity:

```bash
curl http://127.0.0.1:7860/api/ace-step/parity
curl http://127.0.0.1:7860/v1/stats
curl -X POST http://127.0.0.1:7860/v1/init \
  -H 'Content-Type: application/json' \
  -d '{"model":"acestep-v15-xl-sft","init_llm":true,"lm_model_path":"acestep-5Hz-lm-1.7B"}'
```

If `ACESTEP_API_KEY` is set in the environment, official-compatible routes require a matching `Authorization: Bearer ...`, `x-api-key`, `api_key`, or `ai_token`. The normal Studio endpoints are not blocked by that key.

Plan and generate an album:

```python
import requests

base = "http://127.0.0.1:7860"
album_request = {
    "concept": "Dutch club rap album with cinematic hooks",
    "num_tracks": 2,
    "track_duration": 90,
    "song_model_strategy": "xl_sft_final",
    "tag_packs": ["genre_style", "production_style", "vocal_character"],
    "custom_tags": "male rap vocal, radio ready, punchy drums",
    "lyric_density": "rap_dense",
    "rhyme_density": 0.9,
    "hook_intensity": 0.9,
}
plan = requests.post(f"{base}/api/album/plan", json=album_request, timeout=300).json()
album = requests.post(
    f"{base}/api/generate_album",
    json={**album_request, "tracks": plan["tracks"], "track_variants": 2, "save_to_library": True},
    timeout=3600,
).json()
```

Start one-click LoRA training:

```python
import requests

base = "http://127.0.0.1:7860"
job = requests.post(
    f"{base}/api/lora/one-click-train",
    json={"dataset_id": "imported-id", "trigger_tag": "mytrigger", "language": "en"},
).json()
job_id = job["job"]["id"]

# Poll /api/lora/jobs/{job_id}; stages are import, label, save_dataset, preprocess, train, register, load.
```

Load an adapter:

```js
await fetch("/api/lora/load", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ path: "/path/to/adapter/final" })
});
```

## Tests

Run lightweight tests from this folder:

```bash
env/bin/python -m pytest tests -q
```

Heavy model and LoRA acceptance tests are intentionally gated by environment flags and should only be run on suitable hardware.
