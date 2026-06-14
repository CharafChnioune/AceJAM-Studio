from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = BASE_DIR / "vendor" / "mlx-video"
VIDEO_ENV_DIR = BASE_DIR / "video-env"
REPO_URL = "https://github.com/Blaizzy/mlx-video.git"
VAE_PATCH_URLS = [
    ("PR #27 LTX-2.3 VAE channel cap", "https://github.com/Blaizzy/mlx-video/pull/27.patch"),
    ("PR #24 LTX-2.3 sampling fallback", "https://github.com/Blaizzy/mlx-video/pull/24.patch"),
]
FEATURE_PATCH_URLS = [
    ("PR #23 LTX first+last-frame I2V", "https://github.com/Blaizzy/mlx-video/pull/23.patch"),
]


def _run(command: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(command)}")
    return subprocess.run(command, cwd=str(cwd or BASE_DIR), text=True, check=check)


def _video_python() -> Path:
    if sys.platform == "win32":
        return VIDEO_ENV_DIR / "Scripts" / "python.exe"
    return VIDEO_ENV_DIR / "bin" / "python"


def _skip_runtime_install() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        print("[mlx-video] Apple Silicon is required; skipping video-env install on this platform.")
        return True
    return False


def _probe(command: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=12)
    except Exception as exc:
        return False, str(exc)
    text = (completed.stdout or completed.stderr or "").strip()
    return completed.returncode == 0, text


def _git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, cwd=str(VENDOR_DIR)).strip()
    except Exception:
        return ""


def _sync_vendor() -> None:
    VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
    if (VENDOR_DIR / ".git").is_dir():
        _run(["git", "-C", str(VENDOR_DIR), "fetch", "--depth", "1", "origin", "main"])
        _run(["git", "-C", str(VENDOR_DIR), "checkout", "main"])
        _run(["git", "-C", str(VENDOR_DIR), "pull", "--ff-only"])
        return
    _run(["git", "clone", "--depth", "1", REPO_URL, str(VENDOR_DIR)])


def _vae_patch_already_present() -> bool:
    video_vae = VENDOR_DIR / "mlx_video" / "models" / "ltx_2" / "video_vae" / "video_vae.py"
    if not video_vae.is_file():
        return False
    text = video_vae.read_text(encoding="utf-8", errors="ignore")
    return "max_channels" in text and "min(in_channels * multiplier" in text


def _sampling_patch_already_present() -> bool:
    sampling = VENDOR_DIR / "mlx_video" / "models" / "ltx_2" / "video_vae" / "sampling.py"
    if not sampling.is_file():
        return False
    text = sampling.read_text(encoding="utf-8", errors="ignore")
    return "conv branch" in text.lower() and "x_conv" in text and "x_in" in text


def _end_image_patch_already_present() -> bool:
    generate = VENDOR_DIR / "mlx_video" / "models" / "ltx_2" / "generate.py"
    if not generate.is_file():
        return False
    text = generate.read_text(encoding="utf-8", errors="ignore")
    return "--end-image" in text and "end_image_strength" in text


def _vendor_status() -> dict[str, object]:
    return {
        "exists": VENDOR_DIR.exists(),
        "is_git": (VENDOR_DIR / ".git").is_dir(),
        "commit": _git_output(["git", "rev-parse", "--short", "HEAD"]) if (VENDOR_DIR / ".git").is_dir() else "",
        "branch": _git_output(["git", "branch", "--show-current"]) if (VENDOR_DIR / ".git").is_dir() else "",
        "remote": _git_output(["git", "remote", "get-url", "origin"]) if (VENDOR_DIR / ".git").is_dir() else "",
        "dirty_files": _git_output(["git", "status", "--short"]) .splitlines() if (VENDOR_DIR / ".git").is_dir() else [],
    }


def _python_status(python: Path) -> dict[str, object]:
    if not python.is_file():
        return {"exists": False, "version": "", "ok": False, "reason": "video-env python missing"}
    ok, out = _probe([str(python), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"])
    version = out.strip().splitlines()[-1] if ok and out.strip() else ""
    return {"exists": True, "version": version, "ok": ok, "reason": "" if ok else out[-400:]}


def _package_status(python: Path, package_name: str, module_name: str | None = None) -> dict[str, object]:
    module_name = module_name or package_name.replace("-", "_")
    if not python.is_file():
        return {"available": False, "version": "", "reason": "video-env python missing"}
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
    return {"available": available, "version": lines[1] if available and len(lines) > 1 else "", "reason": "" if available else out[-400:]}


def _command_help(command: list[str]) -> dict[str, object]:
    ok, out = _probe(command + ["--help"])
    return {"command": command, "help_ok": ok, "reason": "" if ok else out[-500:]}


def runtime_status() -> dict[str, object]:
    python = _video_python()
    command_matrix = {
        "ltx": _command_help([str(python), "-m", "mlx_video.models.ltx_2.generate"]) if python.is_file() else {"command": [], "help_ok": False, "reason": "video-env python missing"},
        "wan": _command_help([str(python), "-m", "mlx_video.models.wan_2.generate"]) if python.is_file() else {"command": [], "help_ok": False, "reason": "video-env python missing"},
    }
    return {
        "platform": sys.platform,
        "arch": platform.machine(),
        "apple_silicon": sys.platform == "darwin" and platform.machine() == "arm64",
        "vendor": _vendor_status(),
        "python": _python_status(python),
        "packages": {
            "mlx": _package_status(python, "mlx"),
            "mlx-video": _package_status(python, "mlx-video", "mlx_video"),
            "mlx-vlm": _package_status(python, "mlx-vlm", "mlx_vlm"),
            "hf_transfer": _package_status(python, "hf_transfer"),
        },
        "patch_status": {
            "vae_fix_active": _vae_patch_already_present(),
            "sampling_fix_active": _sampling_patch_already_present(),
            "pr23_ltx_i2v_end_frame": _end_image_patch_already_present(),
        },
        "command_help": command_matrix,
        "ready": all(bool(info.get("help_ok")) for info in command_matrix.values()) and python.is_file(),
    }


def _try_apply_upstream_patch(label: str, url: str, already_present) -> bool:
    if already_present():
        print(f"[mlx-video] {label} already present.")
        return True
    try:
        with tempfile.NamedTemporaryFile(suffix=".patch", delete=False) as handle:
            patch_path = Path(handle.name)
        urllib.request.urlretrieve(url, patch_path)
        check = subprocess.run(
            ["git", "-C", str(VENDOR_DIR), "apply", "--check", str(patch_path)],
            text=True,
            capture_output=True,
        )
        if check.returncode != 0:
            print(f"[mlx-video] {label} does not apply cleanly: {(check.stderr or check.stdout).strip()[-500:]}")
            return False
        _run(["git", "-C", str(VENDOR_DIR), "apply", str(patch_path)])
        print(f"[mlx-video] applied {label}")
        return True
    except Exception as exc:
        print(f"[mlx-video] could not apply {label}: {exc}")
        return False
    finally:
        try:
            patch_path.unlink(missing_ok=True)  # type: ignore[name-defined]
        except Exception:
            pass


def _apply_upstream_fixes() -> None:
    if not (VENDOR_DIR / ".git").is_dir():
        print("[mlx-video] vendor clone missing; skipping patch step.")
        return
    vae_active = False
    for label, url in VAE_PATCH_URLS:
        detector = _vae_patch_already_present if "#27" in label else _sampling_patch_already_present
        if _try_apply_upstream_patch(label, url, detector):
            print("[mlx-video] LTX-2.3 VAE fix active.")
            vae_active = True
            break
    if not vae_active:
        print("[mlx-video] upstream LTX-2.3 patch not applied; runtime keeps tokenizer/VAE warnings visible.")
    for label, url in FEATURE_PATCH_URLS:
        if _try_apply_upstream_patch(label, url, _end_image_patch_already_present):
            print("[mlx-video] LTX end-frame conditioning support active.")
            break


def _ensure_video_env() -> None:
    python = _video_python()
    if not python.is_file():
        _run(["uv", "venv", str(VIDEO_ENV_DIR), "--python", "3.11"])
    if not python.is_file():
        raise RuntimeError(f"video-env python missing after uv venv: {python}")
    _run(["uv", "pip", "install", "--python", str(python), "hf_transfer"])
    _run(["uv", "pip", "install", "--python", str(python), "-e", str(VENDOR_DIR)])
    _run([str(python), "-c", "import sys, mlx, mlx_video; assert sys.version_info >= (3, 11); print('mlx-video ready')"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Install MLX Media's isolated mlx-video runtime.")
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
            print(f"[mlx-video] vendor: {status['vendor']}")
            print(f"[mlx-video] python: {status['python']}")
            print(f"[mlx-video] packages: {status['packages']}")
            print(f"[mlx-video] patch_status: {status['patch_status']}")
        return 0
    _sync_vendor()
    _apply_upstream_fixes()
    _ensure_video_env()
    if args.json:
        print(json.dumps(runtime_status(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
