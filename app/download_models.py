from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from acestep.handler import AceStepHandler
from studio_core import (
    ACE_STEP_LM_MODELS,
    KNOWN_ACE_STEP_MODELS,
    OFFICIAL_ACE_STEP_MODEL_REGISTRY,
    OFFICIAL_CORE_MODEL_ID,
    OFFICIAL_MAIN_MODEL_COMPONENTS,
    OFFICIAL_MAIN_MODEL_REPO,
    OFFICIAL_UNRELEASED_MODELS,
    diffusers_pipeline_dir_ready,
    diffusers_pipeline_missing_reasons,
    official_boot_model_ids,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = BASE_DIR / "model_cache"
CHECKPOINT_DIR = MODEL_CACHE_DIR / "checkpoints"
SHARED_RUNTIME_COMPONENTS = ("vae", "Qwen3-Embedding-0.6B")
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".ckpt"}


def _is_diffusers_export(name: str) -> bool:
    return str((OFFICIAL_ACE_STEP_MODEL_REGISTRY.get(name) or {}).get("role") or "") == "diffusers_export"


def checkpoint_dir_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    if _is_diffusers_export(path.name):
        return diffusers_pipeline_dir_ready(path)
    if not (path / "config.json").is_file():
        return False
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map") if isinstance(index, dict) else {}
            shards = sorted({str(name) for name in weight_map.values()}) if isinstance(weight_map, dict) else []
            if shards:
                return all((path / shard).is_file() and (path / shard).stat().st_size > 0 for shard in shards)
        except Exception:
            return False
    return any(child.is_file() and child.stat().st_size > 0 and child.suffix in WEIGHT_SUFFIXES for child in path.iterdir())


def missing_reason(name: str) -> str:
    path = CHECKPOINT_DIR / name
    if not path.exists():
        return f"{name}: missing directory"
    if not path.is_dir():
        return f"{name}: not a directory"
    if _is_diffusers_export(name):
        reasons = diffusers_pipeline_missing_reasons(path)
        if reasons:
            return f"{name}: {', '.join(reasons)}"
        return f"{name}: missing usable Diffusers pipeline files"
    if not (path / "config.json").is_file():
        return f"{name}: missing config.json"
    return f"{name}: missing usable weight files"


def default_download_models() -> list[str]:
    names: list[str] = []

    def add(model_name: str) -> None:
        if model_name and model_name not in names:
            names.append(model_name)

    for model_name in official_boot_model_ids():
        add(model_name)
    add("acestep-v15-turbo")
    for component in SHARED_RUNTIME_COMPONENTS:
        add(component)
    add("acestep-5Hz-lm-1.7B")
    for model_name in KNOWN_ACE_STEP_MODELS:
        if model_name not in OFFICIAL_UNRELEASED_MODELS:
            add(model_name)
    for model_name in ACE_STEP_LM_MODELS:
        if model_name not in {"auto", "none"}:
            add(model_name)
    return names


def verify_runtime_components() -> list[str]:
    failures: list[str] = []
    for component in SHARED_RUNTIME_COMPONENTS:
        if not checkpoint_dir_ready(CHECKPOINT_DIR / component):
            failures.append(missing_reason(component))
    return failures


def download_models(models: Iterable[str], check_only: bool = False) -> list[str]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    handler = AceStepHandler(persistent_storage_path=str(MODEL_CACHE_DIR))
    requested = list(models)
    failures: list[str] = []
    for model_name in requested:
        if model_name == OFFICIAL_CORE_MODEL_ID:
            missing_components = [
                component
                for component in OFFICIAL_MAIN_MODEL_COMPONENTS
                if not checkpoint_dir_ready(CHECKPOINT_DIR / component)
            ]
            if not missing_components:
                print(f"[models] ready: {model_name}", flush=True)
                continue
            if check_only:
                failures.extend(missing_reason(component) for component in missing_components)
                continue
            print(f"[models] downloading: {model_name} ({OFFICIAL_MAIN_MODEL_REPO})", flush=True)
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=OFFICIAL_MAIN_MODEL_REPO,
                local_dir=str(CHECKPOINT_DIR),
                local_dir_use_symlinks=False,
            )
            missing_components = [
                component
                for component in OFFICIAL_MAIN_MODEL_COMPONENTS
                if not checkpoint_dir_ready(CHECKPOINT_DIR / component)
            ]
            if missing_components:
                failures.extend(missing_reason(component) for component in missing_components)
                continue
            print(f"[models] installed: {model_name}", flush=True)
            continue
        path = CHECKPOINT_DIR / model_name
        if checkpoint_dir_ready(path):
            print(f"[models] ready: {model_name}", flush=True)
            continue
        if check_only:
            failures.append(missing_reason(model_name))
            continue
        print(f"[models] downloading: {model_name}", flush=True)
        handler._ensure_model_downloaded(model_name, str(CHECKPOINT_DIR))
        if not checkpoint_dir_ready(path):
            failures.append(missing_reason(model_name))
            continue
        print(f"[models] installed: {model_name}", flush=True)
    if any(model.startswith("acestep-v15-") or model in SHARED_RUNTIME_COMPONENTS for model in requested):
        failures.extend(verify_runtime_components())
    return sorted(set(failures))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and verify AceJAM ACE-Step checkpoints.")
    parser.add_argument("--all", action="store_true", help="Download every AceJAM-supported ACE-Step model.")
    parser.add_argument("--model", action="append", default=[], help="Download one model/component. Can be repeated.")
    parser.add_argument("--check-only", action="store_true", help="Only verify local checkpoint readiness.")
    args = parser.parse_args(argv)

    models = list(args.model) if args.model else default_download_models()
    if not args.all and not args.model:
        models = ["acestep-v15-turbo", *SHARED_RUNTIME_COMPONENTS, "acestep-5Hz-lm-1.7B"]

    print(f"[models] target count: {len(models)}", flush=True)
    failures = download_models(models, check_only=bool(args.check_only))
    if failures:
        print("[models] verification failed:", flush=True)
        for failure in failures:
            print(f"  - {failure}", flush=True)
        return 1
    print("[models] all requested checkpoints are ready.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
