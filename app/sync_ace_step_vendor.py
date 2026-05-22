from __future__ import annotations

import subprocess
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
VENDOR_DIR = APP_DIR / "vendor" / "ACE-Step-1.5"
ACE_STEP_REPO = "https://github.com/ace-step/ACE-Step-1.5"
ACE_STEP_VENDOR_COMMIT = "dce621408bee8c31b4fcf4811682eb9359e1bc94"


def run(args: list[str], *, cwd: Path | None = None) -> None:
    subprocess.check_call(args, cwd=str(cwd) if cwd else None)


def ensure_vendor_checkout() -> None:
    VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not (VENDOR_DIR / ".git").is_dir():
        if VENDOR_DIR.exists() and any(VENDOR_DIR.iterdir()):
            raise SystemExit(f"ACE-Step vendor directory exists but is not a git checkout: {VENDOR_DIR}")
        run(["git", "clone", ACE_STEP_REPO, str(VENDOR_DIR)])

    run(["git", "remote", "set-url", "origin", ACE_STEP_REPO], cwd=VENDOR_DIR)
    run(["git", "fetch", "origin", "--tags", "--prune"], cwd=VENDOR_DIR)
    run(["git", "cat-file", "-e", f"{ACE_STEP_VENDOR_COMMIT}^{{commit}}"], cwd=VENDOR_DIR)
    run(["git", "checkout", "--force", ACE_STEP_VENDOR_COMMIT], cwd=VENDOR_DIR)
    run(["git", "reset", "--hard", ACE_STEP_VENDOR_COMMIT], cwd=VENDOR_DIR)


def main() -> None:
    ensure_vendor_checkout()
    print(f"ACE-Step vendor pinned to {ACE_STEP_VENDOR_COMMIT}")


if __name__ == "__main__":
    main()
