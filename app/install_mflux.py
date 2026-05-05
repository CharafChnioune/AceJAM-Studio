from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MFLUX_ENV_DIR = BASE_DIR / "mflux-env"
REQUIREMENTS_PATH = BASE_DIR / "requirements-mflux.txt"
MFLUX_VERSION_RANGE = ">=0.17.5,<0.18"


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(command)}")
    return subprocess.run(command, cwd=str(cwd or BASE_DIR), text=True, check=True)


def _mflux_python() -> Path:
    if sys.platform == "win32":
        return MFLUX_ENV_DIR / "Scripts" / "python.exe"
    return MFLUX_ENV_DIR / "bin" / "python"


def _skip_runtime_install() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        print("[mflux] Apple Silicon is required; skipping mflux-env install on this platform.")
        return True
    return False


def _ensure_mflux_env() -> None:
    if not REQUIREMENTS_PATH.is_file():
        raise RuntimeError(f"Missing {REQUIREMENTS_PATH.name}; cannot install MFLUX runtime.")
    _run(["uv", "venv", str(MFLUX_ENV_DIR), "--python", "3.10"])
    python = _mflux_python()
    if not python.is_file():
        raise RuntimeError(f"mflux-env python missing after uv venv: {python}")
    _run(["uv", "pip", "install", "--python", str(python), "-r", str(REQUIREMENTS_PATH)])
    _run(
        [
            str(python),
            "-c",
            (
                "import importlib.metadata as md, mlx, mflux; "
                "print('mflux ready', md.version('mflux'), 'mlx', md.version('mlx'))"
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Install MLX Media's isolated MFLUX image runtime.")
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    if _skip_runtime_install():
        return 0
    python = _mflux_python()
    if args.status_only:
        print(f"[mflux] requirements: {REQUIREMENTS_PATH} exists={REQUIREMENTS_PATH.exists()}")
        print(f"[mflux] python: {python} exists={python.exists()}")
        return 0
    _ensure_mflux_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
