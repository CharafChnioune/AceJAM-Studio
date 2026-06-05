"""AceJAM training bootstrap: patches torchaudio and variant map before ACE-Step training."""
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Resolve vendor paths relative to this script's location (works on any machine)
_SCRIPT_DIR = Path(__file__).resolve().parent
_VENDOR = _SCRIPT_DIR / "vendor" / "ACE-Step-1.5"
_NANO_VLLM = _VENDOR / "acestep" / "third_parts" / "nano-vllm"

# Force these paths to be FIRST in sys.path and invalidate any cached acestep imports
for p in [str(_NANO_VLLM), str(_VENDOR)]:
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

# Remove any cached acestep modules so they reload from our vendor path
for key in list(sys.modules.keys()):
    if key == "acestep" or key.startswith("acestep."):
        del sys.modules[key]

# Also set PYTHONPATH for any child processes
os.environ["PYTHONPATH"] = str(_VENDOR) + os.pathsep + str(_NANO_VLLM) + os.pathsep + os.environ.get("PYTHONPATH", "")

# Replace torchaudio.load with a resilient loader. torchaudio 2.9+ can demand
# TorchCodec, while some user MP3s only decode through ffmpeg. Keep the patch
# local to this training subprocess and try every practical backend before
# Side-Step marks a sample failed.
import soundfile
import torch
import torchaudio


_ORIGINAL_TORCHAUDIO_LOAD = torchaudio.load


def _frame_int(value, default):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(default if default >= 0 else parsed, parsed)


def _slice_frames(audio, frame_offset=0, num_frames=-1):
    offset = _frame_int(frame_offset, 0)
    count = _frame_int(num_frames, -1)
    if offset:
        audio = audio[:, offset:]
    if count is not None and count >= 0:
        audio = audio[:, :count]
    return audio


def _soundfile_load(filepath, *, frame_offset=0, num_frames=-1):
    data, sr = soundfile.read(str(filepath), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T.copy()).float()
    return _slice_frames(audio, frame_offset, num_frames), sr


def _ffmpeg_load(filepath, *, frame_offset=0, num_frames=-1):
    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not available for audio decode fallback")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(filepath),
        "-vn",
        "-ac",
        "2",
        "-ar",
        "48000",
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0 and not proc.stdout:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode audio: {detail}")
    array = np.frombuffer(proc.stdout, dtype="<f4")
    if array.size < 2:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg produced no audio samples: {detail}")
    if array.size % 2:
        array = array[:-1]
    audio = torch.from_numpy(array.reshape(-1, 2).T.copy()).float()
    return _slice_frames(audio, frame_offset, num_frames), 48000


def _robust_torchaudio_load(filepath, *args, **kwargs):
    frame_offset = kwargs.get("frame_offset", 0)
    num_frames = kwargs.get("num_frames", -1)
    attempts = [dict(kwargs)]
    if "backend" in kwargs:
        without_backend = dict(kwargs)
        without_backend.pop("backend", None)
        attempts.append(without_backend)
    errors = []
    for attempt in attempts:
        try:
            return _ORIGINAL_TORCHAUDIO_LOAD(filepath, *args, **attempt)
        except Exception as exc:
            errors.append(f"torchaudio: {exc}")
    try:
        return _soundfile_load(filepath, frame_offset=frame_offset, num_frames=num_frames)
    except Exception as exc:
        errors.append(f"soundfile: {exc}")
    try:
        audio, sr = _ffmpeg_load(filepath, frame_offset=frame_offset, num_frames=num_frames)
        print(
            f"[acejam] audio decode fallback used ffmpeg for {Path(str(filepath)).name}",
            file=sys.stderr,
            flush=True,
        )
        return audio, sr
    except Exception as exc:
        errors.append(f"ffmpeg: {exc}")
        print(
            f"[acejam] audio decode failed for {Path(str(filepath)).name}: {' | '.join(errors)}",
            file=sys.stderr,
            flush=True,
        )
        raise


torchaudio.load = _robust_torchaudio_load

# Patch VARIANT_DIR_MAP to include XL models
try:
    from acestep.training_v2.cli.args import VARIANT_DIR_MAP
    for key, value in {
        "turbo_shift3": "acestep-v15-turbo-shift3",
        "xl_turbo": "acestep-v15-xl-turbo",
        "xl_base": "acestep-v15-xl-base",
        "xl_sft": "acestep-v15-xl-sft",
    }.items():
        VARIANT_DIR_MAP.setdefault(key, value)
except ImportError:
    pass

# Override path safety root to allow access to app/data/ (outside vendor dir)
# The vendor's path_safety.py defaults to cwd (vendor/) but our data is in app/data/
try:
    from acestep.training.path_safety import set_safe_root
    # Set safe root to the app directory (parent of vendor) so data/ paths are allowed
    set_safe_root(str(_SCRIPT_DIR))
except ImportError:
    pass

# Fix MPS precision: MPS does not support FP16 GradScaler (unscale_grads fails)
# Patch _select_fabric_precision to use 32-true on MPS instead of 16-mixed
try:
    import acestep.training_v2.fixed_lora_module as _flm
    import acestep.training.trainer as _trainer

    def _fixed_precision(device_type: str) -> str:
        if device_type in ("cuda", "xpu"):
            return "bf16-mixed"
        return "32-true"  # MPS and CPU both use fp32

    def _fixed_compute_dtype(device_type: str) -> torch.dtype:
        if device_type in ("cuda", "xpu"):
            return torch.bfloat16
        return torch.float32  # MPS and CPU both use fp32

    _flm._select_fabric_precision = _fixed_precision
    _flm._select_compute_dtype = _fixed_compute_dtype
    _trainer._select_fabric_precision = _fixed_precision
except (ImportError, AttributeError):
    pass

# Fix MPS gradient clipping: MPS can produce NaN gradients on first steps
# Patch torch.nn.utils.clip_grad_norm_ to allow non-finite norms
try:
    import torch.nn.utils as _tnu
    _orig_clip = _tnu.clip_grad_norm_

    def _safe_clip(parameters, max_norm, **kwargs):
        kwargs["error_if_nonfinite"] = False
        return _orig_clip(parameters, max_norm, **kwargs)

    _tnu.clip_grad_norm_ = _safe_clip
except (ImportError, AttributeError):
    pass

# Patch preprocessing temp filenames before the CLI imports the module. Upstream
# uses the raw audio stem for "*.tmp.pt"; apostrophes/dots can create pass-2
# mismatches and collisions. Keep this patch local and idempotent.
try:
    preprocess_path = _VENDOR / "acestep" / "training_v2" / "preprocess.py"
    if preprocess_path.is_file():
        text = preprocess_path.read_text(encoding="utf-8")
        if "_acejam_safe_tensor_stem" not in text:
            text = text.replace("import logging\n", "import logging\nimport re\n", 1)
            text = text.replace(
                "logger = logging.getLogger(__name__)\n",
                (
                    "logger = logging.getLogger(__name__)\n\n\n"
                    "def _acejam_safe_tensor_stem(index, audio_path):\n"
                    "    stem = re.sub(r\"[^A-Za-z0-9._-]+\", \"-\", audio_path.stem).strip(\"-._\")\n"
                    "    return f\"{index + 1:06d}-{stem or 'sample'}\"\n"
                ),
                1,
            )
            text = text.replace(
                'tmp_path = out_path / f"{af.stem}.tmp.pt"',
                'tmp_path = out_path / f"{_acejam_safe_tensor_stem(i, af)}.tmp.pt"',
                1,
            )
            preprocess_path.write_text(text, encoding="utf-8")
except Exception:
    pass

# Run the actual ACE-Step training CLI by directly executing the file. Keep a
# real argv[0], because argparse reads sys.argv[1:] and would otherwise drop
# the first user flag. Also force the confirmation helper when --yes is present:
# some upstream Side-Step paths still reach the UI prompt during non-interactive
# background jobs.
_target = _VENDOR / "acestep" / "training_v2" / "cli" / "train_fixed.py"
if not _target.is_file():
    print(f"[FAIL] Training CLI not found: {_target}", file=sys.stderr)
    sys.exit(1)


def _normalize_mlx_training_args(args):
    normalized = list(args)
    if "--device" not in normalized:
        return normalized
    index = normalized.index("--device")
    if index + 1 >= len(normalized):
        return normalized
    requested = str(normalized[index + 1]).strip().lower()
    if requested not in {"mlx", "native_mlx", "mlx_training"}:
        return normalized
    if sys.platform != "darwin" or platform.machine() != "arm64":
        print("[FAIL] MLX LoRA training requires macOS Apple Silicon.", file=sys.stderr)
        sys.exit(2)
    try:
        import mlx.core  # noqa: F401
    except Exception as exc:
        print(f"[FAIL] MLX LoRA training requested but mlx is unavailable: {exc}", file=sys.stderr)
        sys.exit(2)
    os.environ["ACEJAM_TRAINING_BACKEND"] = "mlx"
    os.environ["ACESTEP_LM_BACKEND"] = "mlx"
    _install_mlx_training_compat()
    print(
        "[acejam] MLX LoRA training requested; ACE-Step trainer will use the Apple GPU compatibility path internally.",
        flush=True,
    )
    return normalized


def _install_mlx_training_compat():
    # ACE-Step's current LoRA trainer is Lightning/Torch based. It accepts
    # device strings through detect_gpu(), so keep AceJAM's public contract as
    # "mlx" while mapping to the Apple GPU device at the lowest possible layer.
    try:
        import acestep.training_v2.gpu_utils as _gpu_utils

        _original_detect_gpu = _gpu_utils.detect_gpu

        def _acejam_detect_gpu(requested_device="auto", requested_precision="auto"):
            requested = str(requested_device or "auto").strip().lower()
            if requested in {"mlx", "native_mlx", "mlx_training"}:
                info = _original_detect_gpu(requested_device="mps", requested_precision=requested_precision)
                info.name = "Apple MLX"
                return info
            return _original_detect_gpu(requested_device=requested_device, requested_precision=requested_precision)

        _gpu_utils.detect_gpu = _acejam_detect_gpu
    except Exception as exc:
        print(f"[FAIL] Could not install MLX LoRA training compatibility patch: {exc}", file=sys.stderr)
        sys.exit(2)


_user_args = _normalize_mlx_training_args(sys.argv[1:])
sys.argv = [str(_target)] + _user_args
if "--yes" in _user_args or "-y" in _user_args:
    try:
        import acestep.training_v2.ui.config_panel as _config_panel

        _config_panel.confirm_start = lambda skip=False: True
    except Exception:
        pass
exec(compile(_target.read_text(encoding="utf-8"), str(_target), "exec"), {"__name__": "__main__", "__file__": str(_target)})
