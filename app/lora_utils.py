from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def safe_peft_adapter_name(value: Any, fallback: str = "adapter") -> str:
    """Return a PEFT/PyTorch module-safe adapter name."""
    raw = str(value or "").strip()
    if not raw:
        raw = fallback
    name = Path(raw).name or fallback
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        safe = fallback
    if safe[0].isdigit():
        safe = f"adapter_{safe}"
    return safe[:90] or fallback
