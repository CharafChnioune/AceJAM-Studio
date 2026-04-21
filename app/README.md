# AceJAM App Runtime

This folder contains the self-contained AceJAM web app runtime. Pinokio launcher scripts live one level above this folder.

## Runtime Layout

- `app.py`: FastAPI/Gradio server and public API routes.
- `index.html`: Studio UI.
- `studio_core.py`: shared validation, task/model compatibility, and request helpers.
- `lora_trainer.py`: official ACE-Step trainer job manager.
- `acestep/`: local ACE-Step inference runtime used by AceJAM generation.
- `vendor/ACE-Step-1.5/`: official ACE-Step 1.5 trainer clone created by Pinokio install.
- `data/`: uploads, generated results, local library, LoRA datasets, tensor outputs, training jobs, and adapters.
- `model_cache/checkpoints/`: ACE-Step DiT, VAE, text encoder, and LM checkpoints.

## Model Advice Runtime

`studio_core.py` owns the model advice layer:

- `MODEL_PROFILES` covers all known ACE-Step 1.5 DiT checkpoints.
- `LM_MODEL_PROFILES` covers `auto`, `none`, and the 0.6B/1.7B/4B 5Hz LM choices.
- `/api/config` returns `model_profiles`, `lm_model_profiles`, `recommended_song_model`, `recommended_lm_model`, installed model lists, and per-model capabilities.
- Unknown local `acestep-v15-*` folders get a fallback profile inferred from the model name, so locally discovered checkpoints still render safely.

The UI renders this metadata in the model dropdown, model advice panel, Simple LM selector, and Album LM selector. Unsupported task/model combinations remain disabled through the same compatibility matrix used by the generation API.

## Trainer Flow

1. Scan a folder containing audio files.
2. Review/edit captions, lyrics, BPM, key, language, and dataset trigger metadata.
3. Save the official dataset JSON.
4. Preprocess into tensor files through the official ACE-Step `training_v2` pipeline.
5. Train LoRA or LoKr through the official `fixed` Side-Step trainer.
6. Load the final adapter into AceJAM generation.

The trainer runs as a subprocess with `PYTHONPATH` pointed at `vendor/ACE-Step-1.5`, so the official trainer package does not collide with the local inference package.

## API Examples

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
    "batch_size":2,
    "song_model":"acestep-v15-turbo",
    "save_to_library":true
  }'
```

Start LoRA training:

```python
import requests

base = "http://127.0.0.1:7860"
scan = requests.post(f"{base}/api/lora/dataset/scan", json={"path": "/path/to/dataset"}).json()
saved = requests.post(f"{base}/api/lora/dataset/save", json={"entries": scan["files"]}).json()
prep = requests.post(f"{base}/api/lora/preprocess", json={"dataset_json": saved["dataset_path"]}).json()
job_id = prep["job"]["id"]

# Poll /api/lora/jobs/{job_id} until succeeded, then train with the tensor output.
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
env/bin/python -m unittest discover -s tests
```

Heavy model and LoRA acceptance tests are intentionally gated by environment flags and should only be run on suitable hardware.
