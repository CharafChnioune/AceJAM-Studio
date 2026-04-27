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

# Run the actual ACE-Step training CLI by directly executing the file
sys.argv = sys.argv[1:]
_target = _VENDOR / "acestep" / "training_v2" / "cli" / "train_fixed.py"
if not _target.is_file():
    print(f"[FAIL] Training CLI not found: {_target}", file=sys.stderr)
    sys.exit(1)
exec(compile(_target.read_text(encoding="utf-8"), str(_target), "exec"), {"__name__": "__main__", "__file__": str(_target)})
