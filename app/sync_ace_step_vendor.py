from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
VENDOR_DIR = APP_DIR / "vendor" / "ACE-Step-1.5"
ACE_STEP_REPO = "https://github.com/ace-step/ACE-Step-1.5"
ACE_STEP_VENDOR_RELEASE = "v0.1.8"
ACE_STEP_VENDOR_COMMIT = "dce621408bee8c31b4fcf4811682eb9359e1bc94"
KNOWN_PATCH_FILES = {
    "acestep/core/generation/handler/lora/controls.py",
    "acestep/core/generation/handler/lora/lifecycle.py",
    "acestep/core/generation/handler/mlx_dit_init.py",
    "acestep/llm_inference.py",
    "acestep/models/mlx/dit_convert.py",
    "acestep/training/dataset_builder_modules/preprocess_audio.py",
    "acestep/training/trainer.py",
    "acestep/training_v2/cli/args.py",
    "acestep/training_v2/cli/config_builder.py",
    "acestep/training_v2/configs.py",
    "acestep/training_v2/fixed_lora_module.py",
    "acestep/training_v2/gpu_utils.py",
    "acestep/training_v2/preprocess.py",
    "acestep/training_v2/trainer_basic_loop.py",
    "acestep/training_v2/trainer_fixed.py",
}


def run(args: list[str], *, cwd: Path | None = None) -> None:
    subprocess.check_call(args, cwd=str(cwd) if cwd else None)


def capture(args: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=str(cwd) if cwd else None, text=True).strip()


def _normalize_dirty_paths(dirty_lines: list[str]) -> list[str]:
    normalized: list[str] = []
    for line in dirty_lines:
        parts = line.split(maxsplit=1)
        normalized.append(parts[1] if len(parts) == 2 else line)
    return normalized

def ensure_vendor_checkout() -> None:
    VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not (VENDOR_DIR / ".git").is_dir():
        if VENDOR_DIR.exists() and any(VENDOR_DIR.iterdir()):
            raise SystemExit(f"ACE-Step vendor directory exists but is not a git checkout: {VENDOR_DIR}")
        run(["git", "clone", ACE_STEP_REPO, str(VENDOR_DIR)])

    run(["git", "remote", "set-url", "origin", ACE_STEP_REPO], cwd=VENDOR_DIR)
    run(["git", "fetch", "origin", "--tags", "--prune"], cwd=VENDOR_DIR)
    run(["git", "cat-file", "-e", f"{ACE_STEP_VENDOR_COMMIT}^{{commit}}"], cwd=VENDOR_DIR)
    status = vendor_status()
    if status["vendor_matches_pin"] and not status["unknown_drift_files"]:
        print(f"ACE-Step vendor already pinned to {ACE_STEP_VENDOR_RELEASE} ({ACE_STEP_VENDOR_COMMIT})")
        return
    if status["unknown_drift_files"]:
        raise SystemExit(
            "ACE-Step vendor has local drift outside the managed patch set; "
            f"refusing to overwrite {status['unknown_drift_files']}"
        )
    if status["dirty_files"] and not status["vendor_matches_pin"]:
        raise SystemExit(
            "ACE-Step vendor has local patch drift while not on the pinned commit; "
            "refusing to move HEAD without a clean checkout."
        )
    run(["git", "switch", "--detach", ACE_STEP_VENDOR_COMMIT], cwd=VENDOR_DIR)


def vendor_status() -> dict[str, object]:
    exists = (VENDOR_DIR / ".git").is_dir()
    upstream_head = ""
    vendor_head = ""
    dirty_files: list[str] = []
    known_patch_files: list[str] = []
    unknown_drift_files: list[str] = []
    if exists:
        try:
            run(["git", "remote", "set-url", "origin", ACE_STEP_REPO], cwd=VENDOR_DIR)
            run(["git", "fetch", "origin", "--tags", "--prune"], cwd=VENDOR_DIR)
            upstream_head = capture(["git", "rev-parse", "origin/main"], cwd=VENDOR_DIR)
            vendor_head = capture(["git", "rev-parse", "HEAD"], cwd=VENDOR_DIR)
            dirty_output = capture(["git", "status", "--short"], cwd=VENDOR_DIR)
            dirty_files = [line.strip() for line in dirty_output.splitlines() if line.strip()]
            normalized_dirty = []
            for line in dirty_files:
                parts = line.split(maxsplit=1)
                normalized_dirty.append(parts[1] if len(parts) == 2 else line)
            known_patch_files = sorted(path for path in normalized_dirty if path in KNOWN_PATCH_FILES)
            unknown_drift_files = sorted(path for path in normalized_dirty if path not in KNOWN_PATCH_FILES)
        except Exception:
            pass
    return {
        "vendor_dir": str(VENDOR_DIR),
        "repo": ACE_STEP_REPO,
        "pinned_commit": ACE_STEP_VENDOR_COMMIT,
        "vendor_exists": exists,
        "vendor_head": vendor_head,
        "upstream_head": upstream_head,
        "pinned_matches_upstream": bool(upstream_head and upstream_head == ACE_STEP_VENDOR_COMMIT),
        "vendor_matches_pin": bool(vendor_head and vendor_head == ACE_STEP_VENDOR_COMMIT),
        "dirty_files": dirty_files,
        "known_patch_files": known_patch_files,
        "unknown_drift_files": unknown_drift_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync the embedded ACE-Step vendor checkout to the pinned upstream commit.")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print vendor status as JSON.")
    args = parser.parse_args()
    if args.status_only:
        status = vendor_status()
        if args.json:
            print(json.dumps(status, indent=2, sort_keys=True))
        else:
            print(f"ACE-Step vendor status: {status}")
        return
    ensure_vendor_checkout()
    status = vendor_status()
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print(f"ACE-Step vendor pinned to {ACE_STEP_VENDOR_RELEASE} ({ACE_STEP_VENDOR_COMMIT})")


if __name__ == "__main__":
    main()
