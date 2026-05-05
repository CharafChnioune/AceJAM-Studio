"""AceJAM training bootstrap: patches torchaudio and variant map before ACE-Step training."""
import os
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

# Replace torchaudio.load with soundfile.read
# torchaudio 2.9+ ignores backend= parameter and demands torchcodec/FFmpeg
import soundfile
import torch
import torchaudio


def _soundfile_load(filepath, *args, **kwargs):
    kwargs.pop("backend", None)
    kwargs.pop("frame_offset", None)
    kwargs.pop("num_frames", None)
    data, sr = soundfile.read(str(filepath), dtype="float32")
    t = torch.from_numpy(data if len(data.shape) == 1 else data.T).float()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t, sr


torchaudio.load = _soundfile_load

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

# Run the actual ACE-Step training CLI by directly executing the file
sys.argv = sys.argv[1:]
_target = _VENDOR / "acestep" / "training_v2" / "cli" / "train_fixed.py"
if not _target.is_file():
    print(f"[FAIL] Training CLI not found: {_target}", file=sys.stderr)
    sys.exit(1)
exec(compile(_target.read_text(encoding="utf-8"), str(_target), "exec"), {"__name__": "__main__", "__file__": str(_target)})
