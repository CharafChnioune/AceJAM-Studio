# MLX Media Full ACE-Step Studio

MLX Media is a native Pinokio app for ACE-Step 1.5 music generation, MFLUX image generation, and MLX-video clips on Apple Silicon. It keeps the fast prompt-to-song flow, Ollama songwriter support, album agents, and local library, while exposing deeper studio tasks: custom text-to-music, cover/remix, repaint, extract, lego, complete, batches, scoring, LRC timestamps, audio codes, LoRA/LoKr adapter workflows, image tools, and draft-first video rendering.

## Install And Run

1. In Pinokio, open MLX Media and run **Install**.
2. Install creates/uses `app/env`, installs `app/requirements.txt` with `uv pip install -r requirements.txt`, syncs the official ACE-Step 1.5 trainer into `app/vendor/ACE-Step-1.5`, creates `app/mflux-env` for MFLUX image tools, and creates `app/video-env` with Python 3.11 for `mlx-video`.
3. Run **Start**. The Web UI URL is captured by `start.js` and shown by Pinokio.

Generated data lives under `app/data/` and model files under `app/model_cache/`. These folders are ignored by git.

Do not open the built frontend bundle directly with a `file://` URL for real work, for example `app/web/dist/index.html`. The UI now shows a warning in that case because browser file views cannot call the local `/api` backend. Use Pinokio's **Open Web UI** HTTP link.

## Upstream Baseline

As of June 17, 2026 this repo tracks these upstream-safe media baselines:

- ACE-Step 1.5 vendor: `v0.1.8` / commit `dce621408bee8c31b4fcf4811682eb9359e1bc94`.
- MFLUX image runtime: `0.18.x`, aligned with the upstream `v.0.18.0` release.
- MLX-video vendor: pinned to commit `87db56a51758fefb748a359b90a5283bb8ba4837` so Pinokio installs stay reproducible instead of following a moving `main`.

Pinokio update flows now stay non-destructive for live runtimes: `update.js` pulls with `--ff-only`, ACE-Step vendor sync refuses unknown local drift instead of force-resetting, and MLX-video sync refuses to overwrite unmanaged changes while still reusing the managed PR #24/#27 patch set.

## Studio Modes

MLX Media uses a Clean Studio layout by default: the everyday controls stay visible, while advanced inference, LM, output, post-processing, album craft, and trainer hyperparameters are tucked into closed disclosure panels until you need them.

- **Simple**: prompt-to-song with Ollama composition helpers.
- **Custom**: full ACE-Step 1.5 studio controls: caption, global caption, lyrics, reference audio, BPM, key, time signature, duration, batch/seeds, sampler/inference, LM/CoT, output format, post-processing, scoring, LRC, audio codes, cover/repaint extras, and save-to-library.
- **Image Studio**: MFLUX-powered image generation, edit/img2img, fill/inpaint, upscale, depth, and multi-LoRA image rendering on Apple Silicon MLX.
- **Image LoRA Trainer**: MFLUX image-LoRA training with a dedicated image adapter registry under `app/data/mflux/loras`.
- **Video Studio**: `mlx-video` draft-first text/image/audio/song-to-video with LTX/Wan presets, video-LoRAs, MP4 playback, and Final/HQ rerenders.
- **Cover/Remix**: source audio plus cover strength.
- **Repaint**: source audio plus start/end repaint window.
- **Extract, Lego, Complete**: base-model tasks for stems/layers and arrangement completion.
- **Album**: Hit Album Agent Studio with CrewAI/Ollama tools, model strategy, tag packs, lyric craft controls, editable track plans, variants, a dedicated Album LoRA selector, and generation through the same advanced engine.
- **Trainer Studio**: dataset scan, official dataset JSON export, tensor preprocessing, LoRA/LoKr training, estimate, stop, export/load adapter. Finished LoRA folders are registered from the trigger tag.
- **Library**: local saved songs with metadata.
- **Settings**: runtime/config inspection, ACE-Step downloads, LoRA registry, MFLUX health/model presets, and MLX-video health/model-dir registration.

## MFLUX Image Studio

MLX Media no longer exposes Ollama as an image generator. Ollama and LM Studio remain text/planner providers only. Image generation, editing, album art, song art, image-LoRA selection, and image-LoRA training now use [MFLUX](https://github.com/filipstrand/mflux) in an isolated `app/mflux-env`. MFLUX requires `transformers>=5`, while ACE-Step currently needs the 4.x stack, so the environments stay separated on purpose.

The MFLUX API surface is:

- `GET /api/mflux/status`: Apple Silicon, MLX and MFLUX readiness.
- `GET /api/mflux/models`: model catalog and presets for Max Quality, LoRA/Training, Fast Draft, Edit, Upscale and Depth.
- `POST /api/mflux/uploads`: import a PNG/JPG/WEBP/BMP/TIFF source or mask image for edit, inpaint, upscale and depth jobs.
- `POST /api/mflux/jobs`: create a generation/edit/upscale/depth job.
- `GET /api/mflux/jobs/{id}`: inspect progress, logs and result image URLs.
- `DELETE /api/mflux/jobs/{id}`: remove a finished or failed MFLUX job record. Active jobs are rejected with HTTP 409.
- `POST /api/mflux/lora/train`: start MFLUX image-LoRA training.
- `GET /api/mflux/lora/adapters`: list loadable image-LoRA adapters.
- `POST /api/mflux/art/attach`: attach an MFLUX result to a song, generation result, album or album family.

MFLUX results live under `app/data/mflux/results`, source/mask uploads under `app/data/mflux/uploads`, and image-LoRA adapters under `app/data/mflux/loras`. The default image flow is Apple MLX-only; non-Apple or missing-MLX systems return a clear block message instead of falling back to CPU image generation. Image Studio tracks MFLUX `0.18.x` and uses action-specific MFLUX commands such as `mflux-generate-qwen`, `mflux-generate-qwen-edit`, `mflux-generate-flux2`, `mflux-generate-flux2-edit`, `mflux-generate-ernie-image`, `mflux-generate-ernie-image-turbo`, `mflux-generate-ideogram4`, `mflux-generate-z-image`, `mflux-generate-z-image-turbo`, `mflux-upscale-seedvr2`, `mflux-save-depth`, and `mflux-train`. That covers the current upstream `0.18.0` additions: ERNIE-Image, ERNIE-Image-Turbo, Ideogram 4 FP8, and the `flux2-klein-9b-kv` multi-reference edit path.

Relevant MLX image ecosystem notes from the current upstream window: MFLUX now points to [`mlx-taef`](https://github.com/IonDen/mlx-taef) for tiny-autoencoder live previews / lower-memory FLUX decode and [`mlx-teacache`](https://github.com/IonDen/mlx-teacache) for TeaCache step-skipping acceleration on FLUX, Qwen Image and Z-Image. They are not installed by default here, but they are the closest safe Apple-MLX add-ons to watch for future Image Studio speed work.

## MLX Video Studio

Video Studio uses [`Blaizzy/mlx-video`](https://github.com/Blaizzy/mlx-video) in an isolated `app/video-env` because upstream requires Python 3.11 while the music runtime stays on Python 3.10. The default workflow is draft-first: make a small LTX-2.3 preview (`512x320`, `33` frames), then render the same prompt/source/seed as a Final/HQ pass if the motion and composition are worth the time. Legacy LTX-2 presets remain available, but the default draft/final presets now target upstream LTX-2.3.

The MLX-video API surface is:

- `GET /api/mlx-video/status`: Apple Silicon, Python 3.11 video-env, MLX, `mlx-video`, command help, patch status, cache paths, and registered Wan model directories.
- `GET /api/mlx-video/models`: LTX/Wan preset catalog, default actions, capabilities, and Wan model-dir registry.
- `POST /api/mlx-video/uploads`: import source images/audio/video for image-to-video, audio-to-video, or song-to-video.
- `POST /api/mlx-video/jobs`: create text-to-video, image-to-video, audio-to-video, song-to-video, or Final/HQ rerender jobs.
- `GET /api/mlx-video/jobs` and `GET /api/mlx-video/jobs/{id}`: inspect progress, logs, MP4 URL, poster frame, and metadata.
- `DELETE /api/mlx-video/jobs/{id}`: remove a finished or failed video job record. Active jobs are rejected with HTTP 409.
- `GET /api/mlx-video/loras`: list loadable video-LoRAs under `app/data/mlx_video/loras`.
- `POST /api/mlx-video/model-dirs`: register local converted Wan MLX model directories.
- `POST /api/mlx-video/attach`: attach an MP4 result to song/album/library metadata.

Wan models are not downloaded silently. Register converted model folders in Settings -> Video. The installer vendors `mlx-video` under `app/vendor/mlx-video`, pins it to upstream commit `87db56a` (still the latest mainline commit as checked on June 19, 2026), reports JSON runtime status with `python install_mlx_video.py --status-only --json`, surfaces the pinned ref/current commit/drift in Settings, and attempts upstream patch application for known LTX-2.3 fixes while keeping Helios disabled until upstream is stable. The Video wizard now passes upstream-native tiling modes (`auto`, `none`, `default`, `aggressive`, `conservative`, `spatial`, `temporal`), exposes the documented LTX-2.3 spatial upscaler variants (`x2 v1.0`, `x2 v1.1`, `x1.5 v1.0`), preserves those values in result metadata, and lets Wan renders use upstream negative-prompt / no-negative-prompt behavior plus explicit high-noise vs low-noise LoRA roles.

Example `/api/mlx-video/jobs` payloads:

```json
{
  "action": "i2v",
  "model_id": "ltx23-fast-draft",
  "prompt": "cinematic camera move over album art, subtle parallax, moody stage lights",
  "image_path": "/media/mlx-video/uploads/abc123/cover.png",
  "end_image_path": "/media/mlx-video/uploads/def456/final-frame.png",
  "end_image_strength": 0.35,
  "audio_path": "/media/mlx-video/uploads/ghi789/song.wav",
  "audio_start_time": 4.0,
  "tiling": "spatial",
  "spatial_upscaler": "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"
}
```

```json
{
  "action": "t2v",
  "model_id": "wan22-lightning-draft",
  "model_dir": "/Volumes/Media/Wan2.2-T2V-A14B-MLX",
  "prompt": "stylized concert performance, punchy cuts, crowd energy",
  "negative_prompt": "blurry, low quality, text overlay",
  "trim_first_frames": 1,
  "guide_scale": "1",
  "lora_adapters": [
    { "path": "/Volumes/Media/Wan2.2-Lightning/high_noise_model.safetensors", "role": "high", "scale": 1.0 },
    { "path": "/Volumes/Media/Wan2.2-Lightning/low_noise_model.safetensors", "role": "low", "scale": 1.0 }
  ]
}
```

## MLX Ecosystem Watchlist

Two nearby local-only MLX projects are worth watching but are not enabled as default runtime dependencies in MLX Media yet:

- [`Blaizzy/mlx-audio`](https://github.com/Blaizzy/mlx-audio) for STT/TTS/STS tooling on Apple Silicon.
- [`Blaizzy/mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) for multimodal prompt/vision helpers and omni audio/video-capable models.

## ACE-Step Model Advice

MLX Media now shows official ACE-Step model guidance directly in the workspace and under Custom. The dropdown labels and advice panels include best use, quality, speed, VRAM class, steps, CFG support, supported tasks, and whether the checkpoint is already installed in `app/model_cache/checkpoints`. When you select a missing known ACE-Step DiT or 5Hz LM model, MLX Media starts a background Hugging Face download automatically and refreshes the studio as soon as the checkpoint appears.

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
| `acestep-v15-turbo-rl` | Officially mentioned RL variant | Marked as unreleased/not downloadable until ACE-Step publishes a checkpoint. |

LM quick guide:

| LM | Best Use |
| --- | --- |
| `none` | Manual metadata and fastest controlled runs. |
| `acestep-5Hz-lm-0.6B` | Low-VRAM prototyping. |
| `acestep-5Hz-lm-1.7B` | Best default planner. |
| `acestep-5Hz-lm-4B` | Strongest complex planning and audio understanding. |

The guidance is based on the official ACE-Step 1.5 [Hugging Face model card](https://huggingface.co/ACE-Step/Ace-Step1.5), [XL Turbo card](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo), [Ultimate Guide](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md), and [Inference docs](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md).

## Trainer Notes

The trainer uses the official ACE-Step 1.5 `training_v2` / Side-Step CLI as an isolated subprocess. MLX Media does not import that vendor package into the running app process, which prevents conflicts with MLX Media's local `app/acestep` runtime.

Training is hardware-dependent. The UI and API report missing vendor files or Python dependencies clearly. During preprocess/train/estimate, generation is locked and the loaded ACE-Step generation model is released to free memory.

Trainer device is separate from the generation runtime device. On Apple Silicon, `auto` uses the M-chip GPU through PyTorch MPS when available; otherwise it falls back to CUDA or CPU. CPU remains selectable for conservative runs.

Finished one-click and manual training jobs copy the final adapter into `app/data/loras/<trigger-tag>`. If that folder already exists, MLX Media keeps the older adapter and creates `<trigger-tag>-2`, `<trigger-tag>-3`, and so on. Each registered folder includes `acejam_adapter.json` with `display_name`, `trigger_tag`, trainer metadata, model variant, job id, timestamp, and source paths.

LoRA and LoKr training now saves a checkpoint every epoch. For PEFT LoRA jobs, Trainer runs a fixed 20-second, full-quality 64-step WAV epoch audition after each checkpoint by default. The Test genre selector lives next to the training tag and chooses clean standard test lyrics with a visible read-only lyrics preview; there is no free-form test-lyrics box, and the user's trigger tag is added to the caption only instead of being inserted into the sung text. Auditions are attached to `/api/lora/jobs/{id}` as `epoch_auditions` with checkpoint path, status, result id, audio URL, and errors; they are not saved to the Music Library. LoKr checkpoints are still saved every epoch, but auditions are skipped because ACE-Step generation loads PEFT LoRA adapters only.

The Trainer known-adapters menu shows LoRA and LoKr outputs. Generation menus only list loadable PEFT LoRA adapter folders because ACE-Step's standard generation loading expects PEFT files such as `adapter_config.json` and `adapter_model.safetensors`/`.bin`. The global generation adapter applies to Simple, Custom, Cover, Repaint, Lego and Complete. Album uses its own Album adapter selector and sends the selected LoRA path, name and scale to every track and every model render.

## Custom Studio Controls

Custom exposes both the fast MLX Media DiT path and official ACE-Step 1.5 controls:

- Fast path: `wav`, `flac`, `ogg`, text2music/custom generation, BPM/key/time, batches, seeds, CFG interval, ADG, LRC, scoring, audio codes, and library save.
- Official runner path: `thinking`, `sample_mode`, `use_format`, LM/CoT controls, MP3/OPUS/AAC/WAV32 output, normalization, fades, latent shift/rescale, cover noise, repaint crossfades, repaint mode/strength, sampler mode, and velocity controls.
- If a visible control needs the official runner, `/api/generate_advanced` routes through `app/official_runner.py` with `PYTHONPATH=app/vendor/ACE-Step-1.5`. If the vendor checkout, model, or LM is missing, the API returns a clear error instead of ignoring the setting.
- User BPM/key/time/duration always override LM-inferred metadata.

## Official Parity Layer

`GET /api/ace-step/parity` returns MLX Media's explicit ACE-Step parity manifest. It lists official DiT/LM models, tasks, public-compatible endpoints, `GenerationParams`, `GenerationConfig`, runtime knobs, trainer features, source links, and per-item status: `supported`, `guarded`, `missing`, `unreleased`, or `not_applicable`.

Official-compatible routes support optional `ACESTEP_API_KEY`. If the environment variable is set, pass `Authorization: Bearer <key>`, `x-api-key`, `api_key`, or `ai_token` depending on request shape. The local Studio routes stay open for normal Pinokio use.

Runtime controls exposed in Custom/Settings are guarded: device, dtype, flash attention, compile, DiT CPU offload, LM backend/device/dtype/offload, and LM repetition penalty. The official subprocess uses them where the installed ACE-Step vendor runtime supports them.

## Hit Album Agent Studio

Album mode now plans before it generates. The agent layer can use CrewAI with Ollama, but every plan is repaired by the deterministic MLX Media songwriting toolbelt so tracks always carry editable ACE-Step generation data.

Toolbelt highlights:

- `ModelAdvisorTool` and `XLModelPolicyTool`: album final generation is locked to `acestep-v15-xl-sft` by default (`xl_sft_final`). If XL SFT is missing, MLX Media starts a download and resumes instead of silently falling back. Other strategies remain available for planning/previews, but final album rendering uses XL SFT.
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
  song_model_strategy: "xl_sft_final",
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
        "song_model_strategy": "xl_sft_final",
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
        "song_model_strategy": "xl_sft_final",
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
    "song_model_strategy":"xl_sft_final",
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
  -d '{"tensor_dir":"/path/to/app/data/lora_tensors/job","trigger_tag":"mytrigger","adapter_type":"lora","train_epochs":10}'
```

## Main Endpoints

- `GET /api/status` returns runtime health, server URL hint, installed model counts, active downloads, active trainer job, Ollama status, official runner status, and trainer status.
- `GET /api/config` returns available models, installed flags, recommendations, model advice, compatibility, official runner status, backend/UI hashes, official parity, and `ui_schema` for Custom controls.
- `GET /api/ace-step/parity` returns the official parity manifest and runtime status snapshot.
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
- Official ACE-Step-compatible aliases: `GET /health`, `GET /v1/models`, `POST /v1/init`, `GET /v1/stats`, `GET /v1/audio`, `POST /create_random_sample`, `POST /format_input`, `POST /release_task`, `POST /query_result`, `POST /v1/training/start`, `POST /v1/training/start_lokr`.
- `POST /api/uploads`
- `GET /api/results/{id}`
- `POST /api/audio-codes`
- `POST /api/lrc`
- `POST /api/score`
- `GET /api/lora/status`
- `POST /api/lora/dataset/import-folder`
- `POST /api/lora/dataset/scan`
- `POST /api/lora/dataset/save`
- `POST /api/lora/one-click-train`
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
- `POST /api/lora/scale`
- `GET /api/mflux/status`, `GET /api/mflux/models`, `POST /api/mflux/uploads`, `POST /api/mflux/jobs`, `GET /api/mflux/jobs/{id}`, `POST /api/mflux/lora/train`, `GET /api/mflux/lora/adapters`, `POST /api/mflux/art/attach`
- `GET /api/mlx-video/status`, `GET /api/mlx-video/models`, `POST /api/mlx-video/uploads`, `POST /api/mlx-video/jobs`, `GET /api/mlx-video/jobs`, `GET /api/mlx-video/jobs/{id}`, `GET /api/mlx-video/loras`, `POST /api/mlx-video/model-dirs`, `POST /api/mlx-video/attach`

## Development

App code stays in `app/`; launcher scripts stay in the project root. `start.js` must keep Pinokio's generic URL capture pattern:

```js
event: "/(http:\\/\\/[0-9.:]+)/"
url: "{{input.event[1]}}"
```

Lightweight tests:

```bash
app/env/bin/python -m pytest app/tests -q
```
