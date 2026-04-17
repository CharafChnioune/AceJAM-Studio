# AceJAM

A Pinokio launcher for the fully local `AceJAM` app.

The app turns a plain-English song description into a full track:

- a local Qwen GGUF model writes the title, tags, and lyrics
- ACE-Step v1.5 generates the audio
- the custom frontend lets you play, download, and browse a local song feed

## What You Need

- A working GPU is strongly recommended for ACE-Step.
- No Hugging Face token is required.
- The first generation downloads ACE-Step checkpoints and the selected local composer model, so expect a long cold start.

## How To Use

1. Click `Install`.
2. Click `Start`.
3. Wait for the app URL, then open the web UI.
4. Describe the song you want, choose a composer profile if needed, and click `Generate`.

## Notes

- The launcher owns the app under `app/` and installs Python dependencies into `app/env`.
- PyTorch is installed by the launcher with platform/GPU-specific commands in `torch.js`.
- ACE-Step checkpoints live under `app/model_cache/`.
- Local composer GGUF files live under `app/composer_models/`.
- Shared download caches live under `cache/` in the launcher root.
- Saved songs live under `app/data/songs/`.
- `Auto` is the default composer mode. Use `Tiny` for the lowest-memory path, `Balanced` for a reasonable default, and `Quality` if the machine has the headroom.
- `Auto` now escalates long non-instrumental requests to a stronger composer profile so 2-minute songs are less likely to collapse into placeholder lyrics.
- The composer runs CPU-first by default so VRAM stays available for ACE-Step. You can override this with `ACE_STEP_COMPOSER_GPU_LAYERS`.
- On Windows x64 with Python 3.10, the launcher installs the official upstream `llama-cpp-python` `0.3.19` wheel directly to avoid slow local builds.
- On Apple Silicon, the launcher builds `llama-cpp-python` from source with Metal enabled because the hosted Metal wheel path has been unreliable here; other platforms use the CPU wheel by default so the composer stays independent of the ACE-Step GPU path.
- On Intel Mac with Python 3.10, the launcher uses the older `torch==2.2.2` stack, pins `numpy==1.26.4`, `diffusers==0.31.0`, `numba==0.61.2`, and `vector-quantize-pytorch==1.25.0`, and builds `llama-cpp-python` from source with Metal disabled because the newer wheel lines used elsewhere are no longer published for `macos x86_64` and Metal-backed composer init can stall there.
- On Apple Silicon, the launcher now defaults to `acestep-v15-turbo` plus the `Tiny` composer profile when `Auto` is selected, because that is much faster and lighter than the XL + quality path.
- On MPS, ACE-Step now uses lower precision automatically when the local PyTorch build supports it. Override with `ACE_STEP_DTYPE=auto|bfloat16|float16|float32`.
- The web UI now exposes the ACE-Step song model directly, so you can choose `Turbo`, `XL Turbo`, or leave it on `Auto`.
- Override the checkpoint with `ACE_STEP_MODEL`, for example `ACE_STEP_MODEL=acestep-v15-xl-turbo` if you explicitly want the larger model.

## API

Base URL: use the URL captured by Pinokio from `start.js`, usually something like `http://127.0.0.1:7860`.

The app exposes four public Gradio named APIs:

- `/create`
  - Parameters: `description`, `audio_duration`, `seed`, `community`, `composer_profile`, `song_model`, `instrumental`
  - Returns: a JSON string containing `audio`, `title`, `tags`, `lyrics`, `bpm`, `language`, `composer_profile`, `composer_model`, `song_model`, and optionally `community_url`
- `/generate`
  - Parameters: `prompt`, `lyrics`, `audio_duration`, `infer_step`, `guidance_scale`, `seed`, `lora_name_or_path`, `lora_weight`, `song_model`
  - Returns: a `data:audio/wav;base64,...` string
- `/community`
  - Parameters: none
  - Returns: a JSON string containing the current local feed
- `/config`
  - Parameters: none
  - Returns: a JSON string containing `active_song_model`, `default_song_model`, and `available_song_models`

These APIs are queued Gradio endpoints. For JavaScript and Python, use the Gradio client libraries. For raw HTTP, use the queue endpoints under `/gradio_api/call/<name>`.

### JavaScript

```js
import { Client } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:7860");

const result = await client.predict("/create", {
  description: "A glossy synth-pop breakup anthem with a huge chorus",
  audio_duration: 30,
  seed: -1,
  community: false,
  composer_profile: "auto",
  song_model: "auto",
  instrumental: false
});

const raw = result.data?.[0] ?? result.data ?? result;
const song = JSON.parse(typeof raw === "string" ? raw : JSON.stringify(raw));

console.log(song.title);
console.log(song.tags);
console.log(song.composer_model);
console.log(song.song_model);
console.log(song.audio);
```

To call the other public APIs, swap `"/create"` for `"/generate"` or `"/community"` and pass the matching parameters listed above.

### Python

```python
import json
from gradio_client import Client

client = Client("http://127.0.0.1:7860")

raw = client.predict(
    description="A cinematic electronic track about starting over",
    audio_duration=30.0,
    seed=-1,
    community=False,
    composer_profile="auto",
    song_model="auto",
    instrumental=False,
    api_name="/create",
)

song = json.loads(raw)

print(song["title"])
print(song["tags"])
print(song["composer_profile"])
print(song["song_model"])
print(song["audio"][:64])
```

### Curl

```bash
EVENT_ID=$(
  curl -X POST http://127.0.0.1:7860/gradio_api/call/create \
    -s \
    -H "Content-Type: application/json" \
    -d '{"data":["A moody trip-hop ballad with whispered vocals",30,-1,false,"auto","auto",false]}' \
  | python -c 'import sys, json; print(json.load(sys.stdin)["event_id"])'
)

curl -N http://127.0.0.1:7860/gradio_api/call/create/$EVENT_ID
```

For the other APIs, the queue payloads are:

- `/generate`: `{"data":["trip-hop, smoky, cinematic, downtempo","[Verse]\\nCity lights fade under silver rain",30,8,7.0,-1,"",0.8,"auto"]}`
- `/community`: `{"data":[]}`
- `/config`: `{"data":[]}`
