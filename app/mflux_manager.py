from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MFLUX_DIR = DATA_DIR / "mflux"
MFLUX_RESULTS_DIR = MFLUX_DIR / "results"
MFLUX_JOBS_DIR = MFLUX_DIR / "jobs"
MFLUX_LORAS_DIR = MFLUX_DIR / "loras"
MFLUX_DATASETS_DIR = MFLUX_DIR / "datasets"
MFLUX_UPLOADS_DIR = MFLUX_DIR / "uploads"
MFLUX_ENV_DIR = BASE_DIR / "mflux-env"

for _path in (MFLUX_RESULTS_DIR, MFLUX_JOBS_DIR, MFLUX_LORAS_DIR, MFLUX_DATASETS_DIR, MFLUX_UPLOADS_DIR):
    _path.mkdir(parents=True, exist_ok=True)


MFLUX_VERSION_RANGE = ">=0.17.5,<0.18"
MFLUX_RESULT_KEEP_LIMIT = 100
MFLUX_ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str | None) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip(".-")
    return text[:80] or uuid.uuid4().hex[:12]


def _safe_slug(value: str | None, fallback: str = "mflux") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return (text or fallback)[:80]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _env_python() -> Path:
    if sys.platform == "win32":
        return MFLUX_ENV_DIR / "Scripts" / "python.exe"
    return MFLUX_ENV_DIR / "bin" / "python"


def _env_bin(name: str) -> Path:
    if sys.platform == "win32":
        return MFLUX_ENV_DIR / "Scripts" / f"{name}.exe"
    return MFLUX_ENV_DIR / "bin" / name


def _command_path(command: str) -> str | None:
    local = _env_bin(command)
    if local.is_file():
        return str(local)
    return shutil.which(command)


def _run_probe(command: list[str], timeout: int = 8) -> tuple[bool, str]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except Exception as exc:
        return False, str(exc)
    text = (completed.stdout or completed.stderr or "").strip()
    return completed.returncode == 0, text[-1200:]


def _python_import_status(python: Path, module: str) -> dict[str, Any]:
    if not python.is_file():
        return {"available": False, "version": "", "reason": "mflux-env python is missing"}
    code = (
        "import importlib.metadata as md, importlib.util as util; "
        f"dist='{module.replace('_', '-')}'; module='{module}'; "
        "available = util.find_spec(module) is not None; "
        "version = ''; "
        "\ntry:\n"
        "    version = md.version(dist)\n"
        "except md.PackageNotFoundError:\n"
        "    pass\n"
        "print(('1' if available else '0') + '\\n' + version)"
    )
    ok, out = _run_probe([str(python), "-c", code], timeout=4)
    lines = [line.strip() for line in out.strip().splitlines() if line.strip()]
    available = bool(ok and lines and lines[0] == "1")
    return {
        "available": available,
        "version": lines[1] if available and len(lines) > 1 else "",
        "reason": "" if available else out,
    }


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    bin_dir = _env_bin("python").parent
    if bin_dir.is_dir():
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    return env


def _resolve_child(root: Path, *parts: str) -> Path:
    root_resolved = root.resolve()
    target = root.joinpath(*parts).resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise FileNotFoundError("Path escapes MFLUX data directory")
    return target


MFLUX_MODELS: list[dict[str, Any]] = [
    {
        "id": "qwen-image",
        "label": "Qwen Image",
        "preset": "max_quality",
        "family": "qwen-image",
        "family_label": "Qwen Image",
        "size": "20B",
        "quantization_default": 6,
        "default_steps": 30,
        "default_width": 1024,
        "default_height": 1024,
        "command": "mflux-generate-qwen",
        "edit_command": "mflux-generate-qwen-edit",
        "capabilities": ["generate", "edit", "in_context_edit", "lora"],
        "trainable": False,
        "description": "Max quality preset: strongest prompt understanding and image/edit quality, slower and larger.",
    },
    {
        "id": "flux2-klein-9b",
        "label": "FLUX.2 Klein 9B",
        "preset": "lora_training",
        "family": "flux2",
        "family_label": "FLUX.2",
        "size": "9B",
        "quantization_default": 8,
        "default_steps": 20,
        "default_width": 1024,
        "default_height": 1024,
        "command": "mflux-generate-flux2",
        "edit_command": "mflux-generate-flux2-edit",
        "capabilities": ["generate", "edit", "inpaint", "lora", "train_lora"],
        "trainable": True,
        "description": "High quality trainable preset for artwork LoRAs and final covers.",
    },
    {
        "id": "flux2-klein-4b",
        "label": "FLUX.2 Klein 4B",
        "preset": "balanced",
        "family": "flux2",
        "family_label": "FLUX.2",
        "size": "4B",
        "quantization_default": 8,
        "default_steps": 16,
        "default_width": 1024,
        "default_height": 1024,
        "command": "mflux-generate-flux2",
        "edit_command": "mflux-generate-flux2-edit",
        "capabilities": ["generate", "edit", "inpaint", "lora", "train_lora"],
        "trainable": True,
        "description": "Smaller FLUX.2 preset for good local quality with lower memory pressure.",
    },
    {
        "id": "z-image",
        "label": "Z-Image",
        "preset": "quality",
        "family": "z-image",
        "family_label": "Z-Image",
        "size": "6B",
        "quantization_default": 8,
        "default_steps": 20,
        "default_width": 1024,
        "default_height": 1024,
        "command": "mflux-generate-z-image",
        "capabilities": ["generate", "img2img", "lora", "train_lora"],
        "trainable": True,
        "description": "Z-Image base path for higher quality local image work and LoRA experiments.",
    },
    {
        "id": "z-image-turbo",
        "label": "Z-Image Turbo",
        "preset": "fast_draft",
        "family": "z-image",
        "family_label": "Z-Image",
        "size": "6B",
        "quantization_default": 8,
        "default_steps": 9,
        "default_width": 1280,
        "default_height": 768,
        "command": "mflux-generate-z-image-turbo",
        "capabilities": ["generate", "img2img", "lora", "train_lora"],
        "trainable": True,
        "description": "Fast draft and quick iteration preset with LoRA support.",
    },
    {
        "id": "fibo",
        "label": "FIBO",
        "preset": "edit",
        "family": "fibo",
        "family_label": "FIBO",
        "size": "8B",
        "quantization_default": 8,
        "default_steps": 20,
        "default_width": 1024,
        "default_height": 1024,
        "command": "mflux-generate-fibo",
        "edit_command": "mflux-generate-fibo",
        "capabilities": ["generate", "edit", "vlm_refine"],
        "trainable": False,
        "description": "Strong JSON and edit prompt understanding.",
    },
    {
        "id": "seedvr2",
        "label": "SeedVR2 Upscaler",
        "preset": "upscale",
        "family": "seedvr2",
        "family_label": "SeedVR2",
        "size": "3B/7B",
        "quantization_default": 8,
        "default_steps": 20,
        "command": "mflux-upscale-seedvr2",
        "capabilities": ["upscale"],
        "trainable": False,
        "description": "Dedicated high quality image upscaling.",
    },
    {
        "id": "depth-pro",
        "label": "Depth Pro",
        "preset": "depth",
        "family": "depth-pro",
        "family_label": "Depth Pro",
        "quantization_default": 8,
        "command": "mflux-save-depth",
        "capabilities": ["depth"],
        "trainable": False,
        "description": "Fast Apple depth map extraction for control workflows.",
    },
    {
        "id": "flux1-kontext",
        "label": "FLUX.1 Kontext",
        "preset": "legacy_edit",
        "family": "flux1",
        "family_label": "FLUX.1",
        "size": "12B",
        "quantization_default": 8,
        "default_steps": 20,
        "default_width": 1024,
        "default_height": 1024,
        "command": "mflux-generate-kontext",
        "edit_command": "mflux-generate-kontext",
        "capabilities": ["generate", "edit", "controlnet", "lora"],
        "trainable": False,
        "description": "Legacy FLUX edit/control path kept for compatibility.",
    },
]


MFLUX_ACTIONS: dict[str, dict[str, Any]] = {
    "generate": {
        "label": "Text to image",
        "command_key": "command",
        "requires_prompt": True,
        "requires_image": False,
        "default_model": "qwen-image",
    },
    "edit": {
        "label": "Image edit",
        "command_key": "edit_command",
        "fallback_command_key": "command",
        "requires_prompt": True,
        "requires_image": True,
        "default_model": "flux2-klein-9b",
    },
    "inpaint": {
        "label": "Inpaint / fill",
        "command_key": "edit_command",
        "fallback_command_key": "command",
        "requires_prompt": True,
        "requires_image": True,
        "requires_mask": True,
        "default_model": "flux2-klein-9b",
    },
    "upscale": {
        "label": "Upscale",
        "command_key": "command",
        "requires_prompt": False,
        "requires_image": True,
        "default_model": "seedvr2",
    },
    "depth": {
        "label": "Depth export",
        "command_key": "command",
        "requires_prompt": False,
        "requires_image": True,
        "default_model": "depth-pro",
    },
    "train_lora": {
        "label": "Image LoRA training",
        "command": "mflux-train",
        "requires_prompt": False,
        "requires_image": False,
        "default_model": "flux2-klein-9b",
    },
}

_HELP_CACHE: dict[str, dict[str, Any]] = {}


def model_by_id(model_id: str | None) -> dict[str, Any]:
    key = str(model_id or "").strip() or "qwen-image"
    for model in MFLUX_MODELS:
        if model["id"] == key:
            return dict(model)
    return dict(MFLUX_MODELS[0])


def _model_supports_action(model: dict[str, Any], action: str) -> bool:
    capabilities = set(model.get("capabilities") or [])
    if action == "edit":
        return bool({"edit", "img2img", "in_context_edit"} & capabilities)
    if action == "inpaint":
        return "inpaint" in capabilities
    return action in capabilities


def _command_name_for_action(model: dict[str, Any], action: str) -> str:
    if action == "train_lora":
        return "mflux-train"
    spec = MFLUX_ACTIONS.get(action) or MFLUX_ACTIONS["generate"]
    key = str(spec.get("command_key") or "command")
    command = str(model.get(key) or "").strip()
    if not command and spec.get("fallback_command_key"):
        command = str(model.get(str(spec["fallback_command_key"])) or "").strip()
    if not command:
        command = str(model.get("command") or "").strip()
    return command


def _all_commands() -> list[str]:
    commands: set[str] = {"mflux-train"}
    for model in MFLUX_MODELS:
        for key in ("command", "edit_command", "inpaint_command"):
            value = str(model.get(key) or "").strip()
            if value:
                commands.add(value)
    return sorted(commands)


def _command_help_status(command: str, path: str | None) -> dict[str, Any]:
    if not path:
        return {"available": False, "help_ok": False, "reason": "command not on PATH"}
    cached = _HELP_CACHE.get(path)
    if cached:
        return dict(cached)
    result = {"available": True, "help_ok": False, "reason": ""}
    try:
        completed = subprocess.run([path, "--help"], text=True, capture_output=True, timeout=8)
        result["help_ok"] = completed.returncode == 0
        if completed.returncode != 0:
            result["reason"] = (completed.stderr or completed.stdout or "help check failed")[-500:]
    except Exception as exc:
        result["reason"] = str(exc)
    _HELP_CACHE[path] = result
    return dict(result)


def mflux_models() -> dict[str, Any]:
    presets: dict[str, list[dict[str, Any]]] = {}
    by_action: dict[str, list[dict[str, Any]]] = {action: [] for action in MFLUX_ACTIONS if action != "train_lora"}
    for model in MFLUX_MODELS:
        item = dict(model)
        presets.setdefault(str(model["preset"]), []).append(item)
        for action in by_action:
            if _model_supports_action(model, action):
                by_action[action].append(dict(model))
    return {
        "success": True,
        "version_range": MFLUX_VERSION_RANGE,
        "models": [dict(model) for model in MFLUX_MODELS],
        "presets": presets,
        "actions": MFLUX_ACTIONS,
        "by_action": by_action,
        "defaults": {
            "generate": "qwen-image",
            "train_lora": "flux2-klein-9b",
            "fast_draft": "z-image-turbo",
            "edit": "flux2-klein-9b",
            "inpaint": "flux2-klein-9b",
            "upscale": "seedvr2",
            "depth": "depth-pro",
        },
    }


def mflux_status(check_help: bool = True) -> dict[str, Any]:
    is_apple_mlx_platform = sys.platform == "darwin" and platform.machine() == "arm64"
    current_mlx_spec = importlib.util.find_spec("mlx")
    current_mflux_spec = importlib.util.find_spec("mflux")
    env_python = _env_python()
    env_mlx_status = _python_import_status(env_python, "mlx")
    env_mflux_status = _python_import_status(env_python, "mflux")
    mlx_available = bool(env_mlx_status.get("available") or current_mlx_spec)
    mflux_available = bool(env_mflux_status.get("available") or current_mflux_spec)
    command_paths = {cmd: _command_path(cmd) for cmd in _all_commands()}
    command_help = {cmd: _command_help_status(cmd, path) for cmd, path in command_paths.items()} if check_help else {}
    cli_available = any(command_paths.values())
    ready = bool(is_apple_mlx_platform and mlx_available and (mflux_available or cli_available))
    action_readiness: dict[str, Any] = {}
    for action, spec in MFLUX_ACTIONS.items():
        models = [
            model
            for model in MFLUX_MODELS
            if (action == "train_lora" and model.get("trainable")) or (action != "train_lora" and _model_supports_action(model, action))
        ]
        commands = sorted({_command_name_for_action(model, action) for model in models if _command_name_for_action(model, action)})
        available = [cmd for cmd in commands if command_paths.get(cmd)]
        action_readiness[action] = {
            "label": spec.get("label") or action,
            "ready": bool(ready and available),
            "commands": commands,
            "available_commands": available,
            "missing_commands": [cmd for cmd in commands if not command_paths.get(cmd)],
            "models": [model["id"] for model in models],
            "reason": "" if available else f"No MFLUX command available for {action}.",
        }
    return {
        "success": True,
        "ready": ready,
        "platform": sys.platform,
        "arch": platform.machine(),
        "apple_silicon": is_apple_mlx_platform,
        "mlx_available": mlx_available,
        "mflux_available": mflux_available,
        "current_env_mlx_available": bool(current_mlx_spec),
        "current_env_mflux_available": bool(current_mflux_spec),
        "mflux_env_dir": str(MFLUX_ENV_DIR),
        "mflux_env_python": str(env_python),
        "mflux_env_mlx": env_mlx_status,
        "mflux_env_mflux": env_mflux_status,
        "cli_available": cli_available,
        "commands": command_paths,
        "command_help": command_help,
        "action_readiness": action_readiness,
        "data_dir": str(MFLUX_DIR),
        "results_dir": str(MFLUX_RESULTS_DIR),
        "uploads_dir": str(MFLUX_UPLOADS_DIR),
        "datasets_dir": str(MFLUX_DATASETS_DIR),
        "lora_dir": str(MFLUX_LORAS_DIR),
        "cache_home": os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or "",
        "hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") or "",
        "version_range": MFLUX_VERSION_RANGE,
        "blocking_reason": "" if ready else _mflux_blocking_reason(is_apple_mlx_platform, mlx_available, bool(mflux_available or cli_available)),
    }


def _mflux_blocking_reason(apple: bool, mlx: bool, mflux: bool) -> str:
    if not apple:
        return "MLX Media image runtime requires Apple Silicon (darwin arm64)."
    if not mlx:
        return "MLX is not importable in the app environment."
    if not mflux:
        return f"MFLUX is not installed. Re-run Install/Update so mflux{MFLUX_VERSION_RANGE} is available."
    return ""


_jobs_lock = threading.Lock()


def _job_file(job_id: str) -> Path:
    return MFLUX_JOBS_DIR / f"{_safe_id(job_id)}.json"


def _write_job(job: dict[str, Any]) -> dict[str, Any]:
    job["updated_at"] = _now()
    _job_file(str(job["id"])).write_text(json.dumps(_jsonable(job), indent=2), encoding="utf-8")
    return job


def _read_job(job_id: str) -> dict[str, Any] | None:
    path = _job_file(job_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _set_job(job_id: str, **patch: Any) -> dict[str, Any]:
    with _jobs_lock:
        job = _read_job(job_id) or {"id": _safe_id(job_id)}
        job.update(_jsonable(patch))
        return _write_job(job)


def mflux_get_job(job_id: str) -> dict[str, Any] | None:
    return _read_job(job_id)


def mflux_list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for path in sorted(MFLUX_JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        job = _read_job(path.stem)
        if job:
            jobs.append(job)
        if len(jobs) >= limit:
            break
    return jobs


def _result_dir(result_id: str) -> Path:
    path = MFLUX_RESULTS_DIR / _safe_id(result_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _image_url(result_id: str, filename: str) -> str:
    return f"/media/mflux/{_safe_id(result_id)}/{filename}"


def mflux_public_upload_url(upload_id: str, filename: str) -> str:
    return f"/media/mflux/uploads/{_safe_id(upload_id)}/{Path(filename).name}"


def mflux_validate_image_path(value: str | None, *, required: bool = False) -> Path | None:
    text = str(value or "").strip()
    if not text:
        if required:
            raise RuntimeError("Input image is required for this MFLUX action.")
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    else:
        path = path.resolve()
    allowed_roots = [MFLUX_UPLOADS_DIR.resolve(), MFLUX_RESULTS_DIR.resolve(), MFLUX_DATASETS_DIR.resolve(), DATA_DIR.resolve()]
    if path != DATA_DIR.resolve() and not any(root == path or root in path.parents for root in allowed_roots):
        raise RuntimeError("Image path must come from MLX Media uploads, results or data folders.")
    if path.suffix.lower() not in MFLUX_ALLOWED_IMAGE_EXTENSIONS:
        raise RuntimeError(f"Unsupported image file type: {path.suffix or '(none)'}")
    if not path.is_file():
        raise RuntimeError(f"Image file does not exist: {path}")
    return path


def _source_image_metadata(payload: dict[str, Any], image_path: Path | None, mask_path: Path | None) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    if image_path:
        images.append(
            {
                "role": "source",
                "path": str(image_path),
                "url": str(payload.get("image_url") or payload.get("input_image_url") or ""),
                "upload_id": payload.get("image_upload_id") or payload.get("input_upload_id") or "",
            }
        )
    if mask_path:
        images.append(
            {
                "role": "mask",
                "path": str(mask_path),
                "url": str(payload.get("mask_url") or ""),
                "upload_id": payload.get("mask_upload_id") or "",
            }
        )
    return images


def _normalize_lora_family(value: str | None) -> str:
    text = str(value or "").lower()
    if "flux2" in text or "flux.2" in text:
        return "flux2"
    if "z-image" in text or "z_image" in text or "zimage" in text:
        return "z-image"
    if "qwen" in text:
        return "qwen-image"
    if "flux1" in text or "flux.1" in text or "kontext" in text:
        return "flux1"
    return text.strip()


def _lora_args(payload: dict[str, Any], model: dict[str, Any]) -> list[str]:
    raw = payload.get("lora_adapters") or []
    adapters = raw if isinstance(raw, list) else []
    paths: list[str] = []
    scales: list[str] = []
    model_family = _normalize_lora_family(str(model.get("family") or model.get("id") or ""))
    for item in adapters:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or item.get("lora_path") or "").strip()
        if not path:
            continue
        adapter_family = _normalize_lora_family(str(item.get("model_id") or item.get("base_model") or item.get("family") or ""))
        if adapter_family and model_family and adapter_family != model_family:
            raise RuntimeError(f"Image LoRA '{item.get('display_name') or item.get('name') or path}' is for {adapter_family}, not {model_family}.")
        path_obj = Path(path).expanduser()
        if not path_obj.exists():
            raise RuntimeError(f"Image LoRA path does not exist: {path}")
        paths.append(str(path_obj.resolve()))
        try:
            scale = float(item.get("scale", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
        scales.append(str(max(0.0, min(2.0, scale))))
    args: list[str] = []
    if paths:
        args.extend(["--lora-paths", *paths])
        args.extend(["--lora-scales", *scales])
    return args


def _build_mflux_command(payload: dict[str, Any], output: Path) -> list[str]:
    action = str(payload.get("action") or "generate").strip().lower()
    model = model_by_id(payload.get("model_id") or payload.get("model") or MFLUX_ACTIONS.get(action, {}).get("default_model"))
    if not _model_supports_action(model, action):
        raise RuntimeError(f"Model {model['label']} does not support {action}.")
    command = _command_name_for_action(model, action)
    cmd_path = _command_path(command)
    if not cmd_path:
        raise RuntimeError(f"MFLUX command '{command}' is not available. Re-run Install/Update to install mflux{MFLUX_VERSION_RANGE}.")
    spec = MFLUX_ACTIONS.get(action) or MFLUX_ACTIONS["generate"]
    prompt = str(payload.get("prompt") or "").strip()
    if spec.get("requires_prompt") and not prompt:
        raise RuntimeError("Prompt is required for this MFLUX action.")
    image_path = mflux_validate_image_path(payload.get("image_path") or payload.get("input_image_path"), required=bool(spec.get("requires_image")))
    mask_path = mflux_validate_image_path(payload.get("mask_path") or payload.get("inpaint_mask_path"), required=bool(spec.get("requires_mask")))
    width = str(int(payload.get("width") or model.get("default_width") or 1024))
    height = str(int(payload.get("height") or model.get("default_height") or 1024))
    steps = str(int(payload.get("steps") or model.get("default_steps") or 20))
    seed = str(payload.get("seed") if payload.get("seed") not in {None, ""} else -1)
    quantize = str(int(payload.get("quantize") or model.get("quantization_default") or 8))

    args = [cmd_path]
    if action in {"upscale", "depth"}:
        args.extend(["--image-path", str(image_path), "-q", quantize, "--output", str(output)])
        if action == "upscale" and str(payload.get("upscale_factor") or "").strip():
            args.extend(["--upscale-factor", str(payload.get("upscale_factor"))])
        return args

    args.extend(["--prompt", prompt, "--width", width, "--height", height, "--seed", seed, "--steps", steps, "-q", quantize, "--output", str(output)])
    if image_path:
        args.extend(["--image-path", str(image_path)])
    if mask_path:
        args.extend(["--mask-path", str(mask_path)])
    if str(payload.get("guidance") or "").strip():
        args.extend(["--guidance", str(payload.get("guidance"))])
    if str(payload.get("strength") or "").strip():
        args.extend(["--strength", str(payload.get("strength"))])
    args.extend(_lora_args(payload, model))
    return args


def _run_cli_job(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    action = str(payload.get("action") or "generate").strip().lower()
    model = model_by_id(payload.get("model_id") or payload.get("model") or MFLUX_ACTIONS.get(action, {}).get("default_model"))
    result_id = _safe_id(str(payload.get("result_id") or f"mflux-{uuid.uuid4().hex[:12]}"))
    output_name = _safe_slug(payload.get("title") or payload.get("prompt") or f"{action}-{result_id}", "mflux-image") + ".png"
    output_path = _result_dir(result_id) / output_name
    command = _build_mflux_command(payload, output_path)
    image_path = mflux_validate_image_path(payload.get("image_path") or payload.get("input_image_path"), required=False)
    mask_path = mflux_validate_image_path(payload.get("mask_path") or payload.get("inpaint_mask_path"), required=False)
    _set_job(
        job_id,
        state="running",
        status="running",
        stage=f"MFLUX {action} running",
        progress=12,
        command=command,
        result_id=result_id,
        logs=[f"$ {' '.join(command)}"],
    )
    env = _runtime_env()
    completed = subprocess.run(
        command,
        cwd=str(BASE_DIR),
        env=env,
        text=True,
        capture_output=True,
        timeout=max(600, int(payload.get("timeout_seconds") or 7200)),
    )
    logs = [f"$ {' '.join(command)}"]
    if completed.stdout:
        logs.append(completed.stdout[-6000:])
    if completed.stderr:
        logs.append(completed.stderr[-6000:])
    if completed.returncode != 0:
        raise RuntimeError(f"MFLUX exited with code {completed.returncode}: {completed.stderr[-1200:] or completed.stdout[-1200:]}")
    if not output_path.is_file():
        candidates = sorted(output_path.parent.glob("*.png")) + sorted(output_path.parent.glob("*.jpg")) + sorted(output_path.parent.glob("*.jpeg")) + sorted(output_path.parent.glob("*.webp"))
        if not candidates:
            raise RuntimeError("MFLUX finished but did not write an image file.")
        output_path = candidates[-1]
    url = _image_url(result_id, output_path.name)
    metadata = {
        "result_id": result_id,
        "filename": output_path.name,
        "path": str(output_path),
        "url": url,
        "image_url": url,
        "thumbnail_url": url,
        "model_id": model["id"],
        "model_label": model["label"],
        "model_family": model.get("family"),
        "action": action,
        "prompt": payload.get("prompt") or "",
        "width": payload.get("width") or model.get("default_width"),
        "height": payload.get("height") or model.get("default_height"),
        "seed": payload.get("seed"),
        "steps": payload.get("steps") or model.get("default_steps"),
        "quantize": payload.get("quantize") or model.get("quantization_default"),
        "lora_adapters": payload.get("lora_adapters") or [],
        "source_images": _source_image_metadata(payload, image_path, mask_path),
        "attach_status": {},
        "created_at": _now(),
        "source": "mflux",
        "logs": logs,
    }
    (output_path.parent / "mflux_result.json").write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")
    return metadata


def mflux_create_job(payload: dict[str, Any], runner: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None) -> dict[str, Any]:
    status = mflux_status(check_help=False)
    if not status["apple_silicon"] or not status["mlx_available"]:
        raise RuntimeError(status["blocking_reason"])
    if not status["mflux_available"] and not status["cli_available"]:
        raise RuntimeError(status["blocking_reason"])
    action = str(payload.get("action") or "generate").strip().lower()
    if action not in MFLUX_ACTIONS or (action == "train_lora" and runner is None):
        raise RuntimeError(f"Unsupported MFLUX action: {action}")
    model = model_by_id(payload.get("model_id") or payload.get("model") or MFLUX_ACTIONS[action].get("default_model"))
    if action == "train_lora" and not model.get("trainable"):
        raise RuntimeError(f"Model {model['label']} does not support image LoRA training.")
    if action != "train_lora" and not _model_supports_action(model, action):
        raise RuntimeError(f"Model {model['label']} does not support {action}.")
    if MFLUX_ACTIONS[action].get("requires_image"):
        mflux_validate_image_path(payload.get("image_path") or payload.get("input_image_path"), required=True)
    if MFLUX_ACTIONS[action].get("requires_mask"):
        mflux_validate_image_path(payload.get("mask_path") or payload.get("inpaint_mask_path"), required=True)
    _lora_args(payload, model)
    job_id = _safe_id(str(payload.get("job_id") or f"mflux-{uuid.uuid4().hex[:12]}"))
    job = {
        "id": job_id,
        "kind": "mflux",
        "state": "queued",
        "status": "queued",
        "stage": "Queued",
        "progress": 0,
        "created_at": _now(),
        "payload": _jsonable({**payload, "action": action, "model_id": model["id"]}),
        "model": model,
        "logs": [],
    }
    _write_job(job)

    def worker() -> None:
        try:
            active = runner or _run_cli_job
            result = active(job_id, {**payload, "action": action, "model_id": model["id"]})
            _set_job(
                job_id,
                state="succeeded",
                status="succeeded",
                stage="Complete",
                progress=100,
                result=_jsonable(result),
                result_summary={
                    "result_id": result.get("result_id"),
                    "image_url": result.get("image_url") or result.get("url"),
                    "thumbnail_url": result.get("thumbnail_url") or result.get("image_url") or result.get("url"),
                    "model_id": result.get("model_id"),
                    "action": result.get("action"),
                },
                finished_at=_now(),
                logs=result.get("logs") or (_read_job(job_id) or {}).get("logs", []),
            )
        except Exception as exc:
            current = _read_job(job_id) or {}
            logs = list(current.get("logs") or [])
            logs.append(str(exc))
            _set_job(
                job_id,
                state="failed",
                status="failed",
                stage="Failed",
                progress=max(1, int(current.get("progress") or 0)),
                error=str(exc),
                logs=logs,
                finished_at=_now(),
            )

    threading.Thread(target=worker, name=f"mflux-job-{job_id}", daemon=True).start()
    return _read_job(job_id) or job


def mflux_list_lora_adapters() -> list[dict[str, Any]]:
    adapters: list[dict[str, Any]] = []
    for child in sorted(MFLUX_LORAS_DIR.iterdir() if MFLUX_LORAS_DIR.is_dir() else [], key=lambda p: p.name.lower()):
        if not child.exists():
            continue
        meta: dict[str, Any] = {}
        if child.is_dir():
            for meta_name in ("mflux_adapter.json", "adapter_config.json", "metadata.json"):
                path = child / meta_name
                if path.is_file():
                    try:
                        meta.update(json.loads(path.read_text(encoding="utf-8")))
                    except Exception:
                        pass
            files = [p for p in child.iterdir() if p.is_file() and p.suffix.lower() in {".safetensors", ".pt", ".ckpt"}]
            lora_path = files[0] if files else child
        elif child.suffix.lower() in {".safetensors", ".pt", ".ckpt"}:
            lora_path = child
        else:
            continue
        name = str(meta.get("display_name") or meta.get("trigger_tag") or child.stem)
        model_id = str(meta.get("model_id") or meta.get("base_model") or meta.get("model") or "")
        adapters.append(
            {
                "name": child.stem,
                "display_name": name,
                "trigger_tag": meta.get("trigger_tag") or child.stem,
                "path": str(lora_path),
                "adapter_type": "image_lora",
                "model_id": model_id,
                "family": _normalize_lora_family(model_id),
                "base_model": meta.get("base_model") or meta.get("model") or "",
                "generation_loadable": True,
                "is_loadable": True,
                "metadata": meta,
                "updated_at": datetime.fromtimestamp(child.stat().st_mtime, timezone.utc).isoformat(),
            }
        )
    return adapters


def mflux_summarize_training_dataset(dataset_path: str | None, dataset_type: str = "txt2img") -> dict[str, Any]:
    path = Path(str(dataset_path or "")).expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    else:
        path = path.resolve()
    if not path.is_dir():
        raise RuntimeError("Image LoRA training dataset folder does not exist.")
    image_files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in MFLUX_ALLOWED_IMAGE_EXTENSIONS]
    caption_files = [p for p in path.rglob("*.txt") if p.is_file()]
    edit_inputs = [p for p in image_files if "_in" in p.stem.lower()]
    edit_outputs = [p for p in image_files if "_out" in p.stem.lower()]
    missing_captions = [p.name for p in image_files if not (p.with_suffix(".txt")).is_file()]
    summary = {
        "path": str(path),
        "dataset_type": dataset_type,
        "image_count": len(image_files),
        "caption_count": len(caption_files),
        "edit_input_count": len(edit_inputs),
        "edit_output_count": len(edit_outputs),
        "missing_caption_count": len(missing_captions),
        "missing_caption_examples": missing_captions[:8],
    }
    if not image_files:
        raise RuntimeError("Image LoRA dataset has no image files.")
    if dataset_type == "edit" and (not edit_inputs or not edit_outputs):
        raise RuntimeError("Edit LoRA datasets need _in and _out image pairs.")
    return summary


def mflux_start_lora_training(payload: dict[str, Any]) -> dict[str, Any]:
    status = mflux_status(check_help=False)
    if not status["apple_silicon"] or not status["mlx_available"]:
        raise RuntimeError(status["blocking_reason"])
    train_command = _command_path("mflux-train")
    if not train_command:
        raise RuntimeError(f"MFLUX train command is not available. Re-run Install/Update to install mflux{MFLUX_VERSION_RANGE}.")
    model = model_by_id(payload.get("model_id") or "flux2-klein-9b")
    if not model.get("trainable"):
        raise RuntimeError(f"{model['label']} is not marked trainable in the MFLUX catalog.")
    dataset_type = str(payload.get("dataset_type") or "txt2img")
    dataset_summary = mflux_summarize_training_dataset(payload.get("dataset_path"), dataset_type)
    job_id = _safe_id(str(payload.get("job_id") or f"mflux-train-{uuid.uuid4().hex[:12]}"))
    trigger = _safe_slug(payload.get("trigger_tag") or payload.get("name") or job_id, "image-lora")
    out_dir = MFLUX_LORAS_DIR / trigger
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": model["id"],
        "data": dataset_summary["path"],
        "dataset_type": dataset_type,
        "seed": int(payload.get("seed") or 42),
        "steps": int(payload.get("steps") or model.get("default_steps") or 9),
        "guidance": float(payload.get("guidance") or 0.0),
        "quantize": int(payload.get("quantize") or model.get("quantization_default") or 8),
        "preview_prompt": str(payload.get("preview_prompt") or f"{trigger}, editorial album cover"),
        "training_loop": {
            "num_epochs": int(payload.get("epochs") or 1),
            "batch_size": int(payload.get("batch_size") or 1),
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(payload.get("learning_rate") or 1e-4),
        },
        "checkpoint": {
            "output_path": str(out_dir),
            "save_frequency": int(payload.get("save_frequency") or 30),
        },
        "monitoring": {
            "plot_frequency": 1,
            "generate_image_frequency": int(payload.get("generate_image_frequency") or 30),
        },
    }
    config_path = out_dir / "train.json"
    config_path.write_text(json.dumps(_jsonable(config), indent=2), encoding="utf-8")
    meta = {
        "display_name": str(payload.get("display_name") or trigger),
        "trigger_tag": trigger,
        "adapter_type": "image_lora",
        "model_id": model["id"],
        "family": model.get("family"),
        "base_model": model["label"],
        "job_id": job_id,
        "created_at": _now(),
        "config_path": str(config_path),
        "dataset_path": dataset_summary["path"],
        "dataset_summary": dataset_summary,
    }
    (out_dir / "mflux_adapter.json").write_text(json.dumps(_jsonable(meta), indent=2), encoding="utf-8")

    def runner(train_job_id: str, _payload: dict[str, Any]) -> dict[str, Any]:
        cmd = [train_command, "--config", str(config_path)]
        _set_job(
            train_job_id,
            state="running",
            status="running",
            stage="MFLUX LoRA training",
            progress=5,
            command=cmd,
            logs=[f"$ {' '.join(cmd)}"],
            dataset_summary=dataset_summary,
        )
        completed = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            env=_runtime_env(),
            text=True,
            capture_output=True,
            timeout=max(600, int(payload.get("timeout_seconds") or 86400)),
        )
        logs = [f"$ {' '.join(cmd)}", completed.stdout[-6000:], completed.stderr[-6000:]]
        if completed.returncode != 0:
            raise RuntimeError(f"mflux-train exited with code {completed.returncode}: {completed.stderr[-1200:] or completed.stdout[-1200:]}")
        adapters = mflux_list_lora_adapters()
        return {
            "adapter": next((item for item in adapters if item.get("trigger_tag") == trigger), meta),
            "output_dir": str(out_dir),
            "dataset_summary": dataset_summary,
            "logs": logs,
        }

    job = {
        "job_id": job_id,
        "action": "train_lora",
        "model_id": model["id"],
        "trigger_tag": trigger,
        "dataset_path": dataset_summary["path"],
        "dataset_summary": dataset_summary,
    }
    return mflux_create_job(job, runner=runner)
