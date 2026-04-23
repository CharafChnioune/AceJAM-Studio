# AceJAM Full ACE-Step Studio

AceJAM is a native Pinokio app for ACE-Step 1.5 music generation. It keeps the fast AceJAM prompt-to-song flow, Ollama songwriter support, album agents, and local library, while exposing the deeper ACE-Step studio tasks: custom text-to-music, cover/remix, repaint, extract, lego, complete, batches, scoring, LRC timestamps, audio codes, and LoRA/LoKr adapter workflows.

## Install And Run

1. In Pinokio, open AceJAM and run **Install**.
2. Install creates/uses `app/env`, installs `app/requirements.txt` with `uv pip install -r requirements.txt`, and syncs the official ACE-Step 1.5 trainer into `app/vendor/ACE-Step-1.5`.
3. Run **Start**. The Web UI URL is captured by `start.js` and shown by Pinokio.

Generated data lives under `app/data/` and model files under `app/model_cache/`. These folders are ignored by git.

## Studio Modes

AceJAM uses a Clean Studio layout by default: the everyday controls stay visible, while advanced inference, LM, output, post-processing, album craft, and trainer hyperparameters are tucked into closed disclosure panels until you need them.

- **Simple**: prompt-to-song with Ollama composition helpers.
- **Custom**: full ACE-Step 1.5 studio controls: caption, global caption, lyrics, reference audio, BPM, key, time signature, duration, batch/seeds, sampler/inference, LM/CoT, output format, post-processing, scoring, LRC, audio codes, cover/repaint extras, and save-to-library.
- **Cover/Remix**: source audio plus cover strength.
- **Repaint**: source audio plus start/end repaint window.
- **Extract, Lego, Complete**: base-model tasks for stems/layers and arrangement completion.
- **Album**: Hit Album Agent Studio with CrewAI/Ollama tools, model strategy, tag packs, lyric craft controls, editable track plans, variants, and generation through the same advanced engine.
- **Trainer Studio**: dataset scan, official dataset JSON export, tensor preprocessing, LoRA/LoKr training, estimate, stop, export/load adapter.
- **Library**: local saved songs with metadata.
- **Settings**: runtime/config inspection. Generation defaults now live directly in Custom so they are visible while composing.

## ACE-Step Model Advice

AceJAM now shows official ACE-Step model guidance directly in the workspace and under Custom. The dropdown labels and advice panels include best use, quality, speed, VRAM class, steps, CFG support, supported tasks, and whether the checkpoint is already installed in `app/model_cache/checkpoints`. When you select a missing known ACE-Step DiT or 5Hz LM model, AceJAM starts a background Hugging Face download automatically and refreshes the studio as soon as the checkpoint appears.

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

## Custom Studio Controls

Custom exposes both the fast AceJAM DiT path and official ACE-Step 1.5 controls:

- Fast path: `wav`, `flac`, `ogg`, text2music/custom generation, BPM/key/time, batches, seeds, CFG interval, ADG, LRC, scoring, audio codes, and library save.
- Official runner path: `thinking`, `sample_mode`, `use_format`, LM/CoT controls, MP3/OPUS/AAC/WAV32 output, normalization, fades, latent shift/rescale, cover noise, repaint crossfades, repaint mode/strength, sampler mode, and velocity controls.
- If a visible control needs the official runner, `/api/generate_advanced` routes through `app/official_runner.py` with `PYTHONPATH=app/vendor/ACE-Step-1.5`. If the vendor checkout, model, or LM is missing, the API returns a clear error instead of ignoring the setting.
- User BPM/key/time/duration always override LM-inferred metadata.

## Hit Album Agent Studio

Album mode now plans before it generates. The agent layer can use CrewAI with Ollama, but every plan is repaired by the deterministic AceJAM songwriting toolbelt so tracks always carry editable ACE-Step generation data.

Toolbelt highlights:

- `ModelAdvisorTool`: chooses only installed compatible ACE-Step models. Album default prefers `acestep-v15-xl-turbo` when installed, then `acestep-v15-turbo`; maximum-detail prefers installed SFT models.
- `TagLibraryTool`: exposes ACE-Step caption dimensions, lyric section tags, performance tags, stems, curated tag packs, custom tags, and negative tags.
- `LyricLengthTool`: scales word, line, and section targets for short clips through long tracks while keeping ACE-Step lyric limits in view.
- `RhymeFlowTool`, `MetaphorWorldTool`, `HookDoctorTool`, `ClicheGuardTool`, `AlbumArcTool`, `InspirationRadarTool`, `CaptionPolisherTool`, and `ConflictCheckerTool`: provide non-imitative craft reports for rhyme density, hooks, metaphor worlds, repeated lines, tag conflicts, inspiration notes, and caption polish.

Living artist names are converted into broad technique briefs rather than direct imitation prompts. For example, references to dense rap writers become internal rhyme, multisyllabic rhyme, narrative detail, punchline density, hook contrast, and breath-control guidance.

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
    bpm: 128,
    key_scale: "C minor",
    time_signature: "4",
    batch_size: 2,
    auto_score: true,
    auto_lrc: true,
    save_to_library: true
  })
}).then((r) => r.json());

const toolkit = await fetch(`${base}/api/songwriting_toolkit`).then((r) => r.json());

const albumRequest = {
  concept: "Dutch luxury rap album with cinematic hooks",
  num_tracks: 3,
  track_duration: 120,
  song_model_strategy: "best_installed",
  tag_packs: ["genre_style", "production_style", "vocal_character"],
  custom_tags: "male rap vocal, radio ready, punchy drums",
  lyric_density: "rap_dense",
  rhyme_density: 0.9,
  hook_intensity: 0.9
};

const albumPlan = await fetch(`${base}/api/album/plan`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(albumRequest)
}).then((r) => r.json());

const album = await fetch(`${base}/api/generate_album`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    ...albumRequest,
    track_variants: 2,
    tracks: albumPlan.tracks,
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

album_plan = requests.post(
    f"{base}/api/album/plan",
    json={
        "concept": "cinematic Dutch rap about discipline and victory",
        "num_tracks": 2,
        "track_duration": 90,
        "song_model_strategy": "best_installed",
        "lyric_density": "rap_dense",
        "tag_packs": ["genre_style", "speed_rhythm", "production_style"],
    },
    timeout=300,
).json()
album = requests.post(
    f"{base}/api/generate_album",
    json={
        "concept": "cinematic Dutch rap about discipline and victory",
        "num_tracks": 2,
        "track_duration": 90,
        "song_model_strategy": "best_installed",
        "tracks": album_plan["tracks"],
        "save_to_library": True,
    },
    timeout=3600,
).json()
```

curl:

```bash
curl -X POST http://127.0.0.1:7860/api/generate_advanced \
  -H 'Content-Type: application/json' \
  -d '{
    "task_type":"text2music",
    "caption":"cinematic afrohouse, bright vocals, deep percussion",
    "lyrics":"[Verse]\n...",
    "duration":75,
    "bpm":122,
    "keyscale":"D minor",
    "thinking":true,
    "lm_temperature":0.85,
    "audio_format":"mp3",
    "mp3_bitrate":"192k",
    "save_to_library":true
  }'

curl http://127.0.0.1:7860/api/songwriting_toolkit

curl -X POST http://127.0.0.1:7860/api/album/plan \
  -H 'Content-Type: application/json' \
  -d '{
    "concept":"Dutch club rap album with cinematic hooks",
    "num_tracks":2,
    "track_duration":90,
    "song_model_strategy":"best_installed",
    "tag_packs":["genre_style","production_style"],
    "custom_tags":"male rap vocal, radio ready, punchy drums",
    "lyric_density":"rap_dense"
  }'

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

- `GET /api/config` returns available models, installed flags, recommendations, model advice, compatibility, official runner status, and `ui_schema` for Custom controls.
- `GET /api/songwriting_toolkit` returns tag taxonomy, lyric meta tags, craft tools, density presets, and model strategy descriptions.
- `GET /api/models/downloads` returns active/missing/installed model download states.
- `GET /api/models/download/{model_name}` returns one model download state.
- `POST /api/models/download` starts a background download for a known ACE-Step DiT or 5Hz LM model.
- `POST /api/compose`
- `POST /api/create_sample`
- `POST /api/format_sample`
- `POST /api/generate_advanced`
- `POST /api/album/plan`
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
