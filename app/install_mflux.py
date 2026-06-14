from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MFLUX_ENV_DIR = BASE_DIR / "mflux-env"
REQUIREMENTS_PATH = BASE_DIR / "requirements-mflux.txt"
MFLUX_VERSION_RANGE = ">=0.18,<0.19"


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


def _probe(command: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=12)
    except Exception as exc:
        return False, str(exc)
    text = (completed.stdout or completed.stderr or "").strip()
    return completed.returncode == 0, text


def _package_status(python: Path, package_name: str, module_name: str | None = None) -> dict[str, object]:
    module_name = module_name or package_name.replace("-", "_")
    if not python.is_file():
        return {"available": False, "version": "", "reason": "mflux-env python missing"}
    code = (
        "import importlib.metadata as md, importlib.util as util\n"
        f"dist = {package_name!r}\n"
        f"module = {module_name!r}\n"
        "available = util.find_spec(module) is not None\n"
        "version = ''\n"
        "try:\n"
        "    version = md.version(dist)\n"
        "except md.PackageNotFoundError:\n"
        "    pass\n"
        "print(('1' if available else '0') + '\\n' + version)\n"
    )
    ok, out = _probe([str(python), "-c", code])
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    available = bool(ok and lines and lines[0] == "1")
    return {
        "available": available,
        "version": lines[1] if available and len(lines) > 1 else "",
        "reason": "" if available else out[-400:],
    }


def _python_status(python: Path) -> dict[str, object]:
    if not python.is_file():
        return {"exists": False, "version": "", "ok": False, "reason": "mflux-env python missing"}
    ok, out = _probe([str(python), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"])
    version = out.strip().splitlines()[-1] if ok and out.strip() else ""
    return {"exists": True, "version": version, "ok": ok, "reason": "" if ok else out[-400:]}


def runtime_status() -> dict[str, object]:
    python = _mflux_python()
    mflux_status = _package_status(python, "mflux")
    mlx_status = _package_status(python, "mlx")
    transformers_status = _package_status(python, "transformers")
    return {
        "platform": sys.platform,
        "arch": platform.machine(),
        "apple_silicon": sys.platform == "darwin" and platform.machine() == "arm64",
        "requirements_path": str(REQUIREMENTS_PATH),
        "requirements_exists": REQUIREMENTS_PATH.exists(),
        "version_range": MFLUX_VERSION_RANGE,
        "env_dir": str(MFLUX_ENV_DIR),
        "python": _python_status(python),
        "packages": {
            "mflux": mflux_status,
            "mlx": mlx_status,
            "transformers": transformers_status,
        },
        "ready": bool(mflux_status.get("available") and mlx_status.get("available")),
    }


def _ensure_mflux_env() -> None:
    if not REQUIREMENTS_PATH.is_file():
        raise RuntimeError(f"Missing {REQUIREMENTS_PATH.name}; cannot install MFLUX runtime.")
    python = _mflux_python()
    if not python.is_file():
        _run(["uv", "venv", str(MFLUX_ENV_DIR), "--python", "3.10"])
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
    parser.add_argument("--json", action="store_true", help="Print runtime status as JSON.")
    args = parser.parse_args()
    if _skip_runtime_install():
        return 0
    if args.status_only:
        status = runtime_status()
        if args.json:
            print(json.dumps(status, indent=2, sort_keys=True))
        else:
            print(f"[mflux] requirements: {status['requirements_path']} exists={status['requirements_exists']}")
            print(f"[mflux] python: {status['python']}")
            print(f"[mflux] packages: {status['packages']}")
            print(f"[mflux] version range: {status['version_range']}")
        return 0
    _ensure_mflux_env()
    if args.json:
        print(json.dumps(runtime_status(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
