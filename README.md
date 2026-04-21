# AceJAM Full ACE-Step Studio

AceJAM is a native Pinokio app for ACE-Step 1.5 music generation. It keeps the fast AceJAM prompt-to-song flow, Ollama songwriter support, album generation, and local library, while exposing the deeper ACE-Step studio tasks: custom text-to-music, cover/remix, repaint, extract, lego, complete, batches, scoring, LRC timestamps, audio codes, and LoRA/LoKr adapter workflows.

## Install And Run

1. In Pinokio, open AceJAM and run **Install**.
2. Install creates/uses `app/env`, installs `app/requirements.txt` with `uv pip install -r requirements.txt`, and syncs the official ACE-Step 1.5 trainer into `app/vendor/ACE-Step-1.5`.
3. Run **Start**. The Web UI URL is captured by `start.js` and shown by Pinokio.

Generated data lives under `app/data/` and model files under `app/model_cache/`. These folders are ignored by git.

## Studio Modes

- **Simple**: prompt-to-song with Ollama composition helpers.
- **Custom**: direct caption, lyrics, metadata, seed, batch, and inference controls.
- **Cover/Remix**: source audio plus cover strength.
- **Repaint**: source audio plus start/end repaint window.
- **Extract, Lego, Complete**: base-model tasks for stems/layers and arrangement completion.
- **Album**: multi-track generation through the same generation engine.
- **Trainer Studio**: dataset scan, official dataset JSON export, tensor preprocessing, LoRA/LoKr training, estimate, stop, export/load adapter.
- **Library**: local saved songs with metadata.
- **Settings**: advanced inference controls.

## ACE-Step Model Advice

AceJAM now shows official ACE-Step model guidance directly beside the model selector. The dropdown labels and advice panel include best use, quality, speed, VRAM class, steps, CFG support, supported tasks, and whether the checkpoint is already installed in `app/model_cache/checkpoints`.

DiT quick guide:

| Model | Best Use | Notes |
| --- | --- | --- |
| `acestep-v15-turbo` | Best default daily driver | 8 steps, no CFG, fastest proven choice for text2music, cover, and repaint. |
| `acestep-v15-turbo-shift3` | Clearer/richer timbre | Fast niche variant that can sound drier with simpler orchestration. |
| `acestep-v15-sft` | CFG/detail tuning without XL | 50 steps, slower, better when prompt adherence and detail matter. |
| `acestep-v15-base` | Extract, lego, complete, fine-tuning | Most flexible 2B model and required for the advanced context tasks. |
| `acestep-v15-xl-turbo` | Best 20GB+ quality daily driver | XL quality with 8-step turbo speed; 20GB+ VRAM is recommended without offload. |
| `acestep-v15-xl-sft` | Highest-detail XL standard tasks | 50 steps with CFG for slower, detailed tuning. |
| `acestep-v15-xl-base` | XL all-task model | Best XL choice for extract, lego, complete, and advanced control. |

LM quick guide:

| LM | Best Use |
| --- | --- |
| `none` | Manual metadata and fastest controlled runs. |
| `acestep-5Hz-lm-0.6B` | Low-VRAM prototyping. |
| `acestep-5Hz-lm-1.7B` | Best default planner. |
| `acestep-5Hz-lm-4B` | Strongest complex planning and audio understanding. |

The guidance is based on the official ACE-Step 1.5 [Hugging Face model card](https://huggingface.co/ACE-Step/Ace-Step1.5), [XL Turbo card](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo), [Ultimate Guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md), and [Inference docs](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md).

## Trainer Notes

The trainer uses the official ACE-Step 1.5 `training_v2` / Side-Step CLI as an isolated subprocess. AceJAM does not import that vendor package into the running app process, which prevents conflicts with AceJAM's local `app/acestep` runtime.

Training is hardware-dependent. The UI and API report missing vendor files or Python dependencies clearly. During preprocess/train/estimate, generation is locked and the loaded ACE-Step generation model is released to free memory.

## HTTP API Examples

JavaScript:

```js
const base = "http://127.0.0.1:7860";

const sample = await fetch(`${base}/api/create_sample`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ description: "bright garage pop anthem", duration: 60 })
}).then((r) => r.json());

const generated = await fetch(`${base}/api/generate_advanced`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    task_type: "text2music",
    caption: sample.tags,
    lyrics: sample.lyrics,
    duration: 60,
    song_model: "acestep-v15-turbo",
    batch_size: 2,
    auto_score: true,
    auto_lrc: true,
    save_to_library: true
  })
}).then((r) => r.json());
```

Python:

```python
import requests

base = "http://127.0.0.1:7860"
payload = {
    "task_type": "cover",
    "caption": "energetic synth rock cover",
    "lyrics": "[Instrumental]",
    "src_audio_id": "uploaded-audio-id",
    "audio_cover_strength": 0.65,
    "duration": 90,
}
result = requests.post(f"{base}/api/generate_advanced", json=payload, timeout=3600).json()
print(result["result_id"])
```

curl:

```bash
curl -X POST http://127.0.0.1:7860/api/lora/dataset/scan \
  -H 'Content-Type: application/json' \
  -d '{"path":"/path/to/audio/folder"}'

curl -X POST http://127.0.0.1:7860/api/lora/preprocess \
  -H 'Content-Type: application/json' \
  -d '{"dataset_json":"/path/to/app/data/lora_datasets/unit.json","song_model":"acestep-v15-turbo"}'

curl -X POST http://127.0.0.1:7860/api/lora/train \
  -H 'Content-Type: application/json' \
  -d '{"tensor_dir":"/path/to/app/data/lora_tensors/job","adapter_type":"lora","train_epochs":10}'
```

## Main Endpoints

- `GET /api/config` returns available models, installed flags, recommendations, model advice, and compatibility.
- `POST /api/compose`
- `POST /api/create_sample`
- `POST /api/format_sample`
- `POST /api/generate_advanced`
- `POST /api/generate_album`
- `POST /api/uploads`
- `GET /api/results/{id}`
- `POST /api/audio-codes`
- `POST /api/lrc`
- `POST /api/score`
- `GET /api/lora/status`
- `POST /api/lora/dataset/scan`
- `POST /api/lora/dataset/save`
- `POST /api/lora/preprocess`
- `POST /api/lora/train`
- `POST /api/lora/estimate`
- `GET /api/lora/jobs`
- `GET /api/lora/jobs/{id}`
- `GET /api/lora/jobs/{id}/log`
- `POST /api/lora/jobs/{id}/stop`
- `GET /api/lora/adapters`
- `POST /api/lora/load`
- `POST /api/lora/unload`
- `POST /api/lora/use`

## Development

App code stays in `app/`; launcher scripts stay in the project root. `start.js` must keep Pinokio's generic URL capture pattern:

```js
event: "/(http:\\/\\/[0-9.:]+)/"
url: "{{input.event[1]}}"
```
