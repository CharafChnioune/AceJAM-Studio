from __future__ import annotations

import argparse
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
    _run(["uv", "venv", str(VIDEO_ENV_DIR), "--python", "3.11"])
    python = _video_python()
    if not python.is_file():
        raise RuntimeError(f"video-env python missing after uv venv: {python}")
    _run(["uv", "pip", "install", "--python", str(python), "hf_transfer"])
    _run(["uv", "pip", "install", "--python", str(python), "-e", str(VENDOR_DIR)])
    _run([str(python), "-c", "import sys, mlx, mlx_video; assert sys.version_info >= (3, 11); print('mlx-video ready')"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Install MLX Media's isolated mlx-video runtime.")
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    if _skip_runtime_install():
        return 0
    if args.status_only:
        python = _video_python()
        print(f"[mlx-video] vendor: {VENDOR_DIR} exists={VENDOR_DIR.exists()}")
        print(f"[mlx-video] python: {python} exists={python.exists()}")
        return 0
    _sync_vendor()
    _apply_upstream_fixes()
    _ensure_video_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
