from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CLUSTER_ROOT = Path("/Volumes/4tb/lora_clusters")
DEFAULT_MIRROR_ROOT = Path("/Volumes/4tb/loras")
DEFAULT_SCRATCH_ROOT = Path("/Volumes/4tb/tmp/acejam-lora-batch")
DEFAULT_STATE_FILE = BASE_DIR / "data" / "lora_cluster_batch_state.json"
DEFAULT_LOG_FILE = BASE_DIR / "data" / "lora_cluster_batch.log"
ACTIVE_JOB_STATES = {"queued", "running", "stopping"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def derive_trigger_tag(folder_name: str) -> str:
    text = str(folder_name or "").strip()
    if text.lower().startswith("lora_"):
        text = text[5:]
    text = text.strip(" _-")
    return text or str(folder_name or "").strip() or "lora-cluster"


def list_cluster_folders(cluster_root: Path) -> list[Path]:
    root = cluster_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cluster root not found: {root}")
    return sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name.lower())


def load_state(state_file: Path) -> dict[str, Any]:
    path = state_file.expanduser()
    if not path.is_file():
        return {"entries": {}, "updated_at": ""}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"entries": {}, "updated_at": ""}
    entries = data.get("entries")
    if not isinstance(entries, dict):
        data["entries"] = {}
    return data


def save_state(state_file: Path, state: dict[str, Any]) -> None:
    path = state_file.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["updated_at"] = utc_now()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _path_exists(value: str | None) -> bool:
    return bool(value) and Path(str(value)).expanduser().exists()


def should_skip_completed_entry(entry: dict[str, Any], *, force: bool = False) -> bool:
    if force:
        return False
    if str(entry.get("status") or "").strip().lower() != "completed":
        return False
    return _path_exists(entry.get("registered_path")) and _path_exists(entry.get("mirror_path"))


def mirror_adapter(registered_path: Path, mirror_root: Path) -> Path:
    source = registered_path.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Registered adapter not found: {source}")
    target = mirror_root.expanduser().resolve() / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    shutil.copytree(source, target)
    return target


def cleanup_training_dir(path_value: str | None, *, logger: logging.Logger) -> None:
    if not path_value:
        return
    path = Path(path_value).expanduser()
    if not path.exists():
        return
    shutil.rmtree(path)
    logger.info("cleanup training_dir=%s", path)


def cleanup_path(path_value: str | None, *, logger: logging.Logger, label: str) -> None:
    if not path_value:
        return
    path = Path(path_value).expanduser()
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    logger.info("cleanup %s=%s", label, path)


def cleanup_result_ids(result_ids: list[str], *, logger: logging.Logger) -> None:
    for result_id in result_ids:
        safe_id = str(result_id or "").strip()
        if not safe_id:
            continue
        cleanup_path(str(BASE_DIR / "data" / "results" / safe_id), logger=logger, label="result_dir")


def collect_result_ids(result: dict[str, Any], finished_job: dict[str, Any]) -> list[str]:
    result_ids: list[str] = []

    def _add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in result_ids:
            result_ids.append(text)

    _add(result.get("result_id"))
    for item in list(result.get("epoch_auditions") or []):
        if isinstance(item, dict):
            _add(item.get("result_id"))
    for item in list((finished_job.get("result") or {}).get("epoch_auditions") or []):
        if isinstance(item, dict):
            _add(item.get("result_id"))
    return result_ids


def cleanup_entry_artifacts(entry: dict[str, Any], *, logger: logging.Logger) -> None:
    cleanup_training_dir(entry.get("training_output_dir"), logger=logger)
    cleanup_path(entry.get("tensor_output_dir"), logger=logger, label="tensor_output_dir")
    cleanup_result_ids(list(entry.get("result_ids") or []), logger=logger)


def configure_batch_runtime(scratch_root: Path, *, logger: logging.Logger) -> None:
    scratch = scratch_root.expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    os.environ["TMP"] = str(scratch)
    os.environ["TEMP"] = str(scratch)
    logger.info("runtime scratch_root=%s", scratch)


def reclaim_batch_artifacts(state: dict[str, Any], *, logger: logging.Logger) -> None:
    entries = state.get("entries") or {}
    if not isinstance(entries, dict):
        return
    for entry in entries.values():
        if isinstance(entry, dict):
            cleanup_entry_artifacts(entry, logger=logger)


def build_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("lora_cluster_batch")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.propagate = False
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def get_default_training_manager():
    try:
        from app import training_manager

        return training_manager
    except Exception:
        from lora_trainer import AceTrainingManager

        return AceTrainingManager(
            base_dir=BASE_DIR,
            data_dir=BASE_DIR / "data",
            model_cache_dir=BASE_DIR / "model_cache",
        )


def wait_for_job_completion(
    manager: Any,
    job_id: str,
    *,
    poll_interval: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    last_state = ""
    last_stage = ""
    while True:
        job = dict(manager.get_job(job_id))
        state = str(job.get("state") or "")
        stage = str(job.get("stage") or "")
        if state != last_state or stage != last_stage:
            logger.info("job=%s state=%s stage=%s progress=%s", job_id, state, stage, job.get("progress"))
            last_state = state
            last_stage = stage
        if state not in ACTIVE_JOB_STATES:
            return job
        time.sleep(max(0.1, float(poll_interval)))


def run_batch(
    *,
    manager: Any,
    cluster_root: Path,
    mirror_root: Path,
    state_file: Path,
    log_file: Path,
    language: str = "en",
    limit: int | None = None,
    force: bool = False,
    poll_interval: float = 30.0,
    cleanup_training_artifacts: bool = True,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    logger = build_logger(log_file)
    configure_batch_runtime(scratch_root, logger=logger)
    active_job = manager.active_job() if callable(getattr(manager, "active_job", None)) else None
    if active_job:
        raise RuntimeError(f"Training job already active: {active_job.get('id') or active_job}")

    folders = list_cluster_folders(cluster_root)
    if limit is not None:
        folders = folders[: max(0, int(limit))]

    state = load_state(state_file)
    entries = state.setdefault("entries", {})
    reclaim_batch_artifacts(state, logger=logger)
    summary = {"completed": 0, "failed": 0, "skipped": 0, "total": len(folders)}

    for folder in folders:
        folder_name = folder.name
        trigger_tag = derive_trigger_tag(folder_name)
        entry = dict(entries.get(folder_name) or {})
        entry.update(
            {
                "folder_name": folder_name,
                "source_path": str(folder),
                "trigger_tag": trigger_tag,
            }
        )

        if should_skip_completed_entry(entry, force=force):
            summary["skipped"] += 1
            logger.info("skip folder=%s trigger=%s registered=%s mirror=%s", folder_name, trigger_tag, entry.get("registered_path"), entry.get("mirror_path"))
            entries[folder_name] = entry
            save_state(state_file, state)
            continue

        entry.update(
            {
                "status": "running",
                "job_id": "",
                "started_at": utc_now(),
                "finished_at": "",
                "registered_path": "",
                "mirror_path": "",
                "error": "",
            }
        )
        entries[folder_name] = entry
        save_state(state_file, state)
        logger.info("start folder=%s trigger=%s source=%s", folder_name, trigger_tag, folder)

        payload = {
            "dataset_id": folder_name,
            "import_root": str(folder),
            "trigger_tag": trigger_tag,
            "custom_tag": trigger_tag,
            "language": language,
            "auto_load": False,
            "save_every_n_epochs": 10,
            "epoch_audition_every_n_epochs": 10,
        }

        try:
            started_job = dict(manager.start_one_click_train(payload))
            job_id = str(started_job.get("id") or "")
            if not job_id:
                raise RuntimeError(f"Could not determine job id from start_one_click_train response: {started_job}")
            entry["job_id"] = job_id
            entry["tensor_output_dir"] = str(((started_job.get("paths") or {}).get("tensor_output") or "")).strip()
            entries[folder_name] = entry
            save_state(state_file, state)

            finished_job = wait_for_job_completion(
                manager,
                job_id,
                poll_interval=poll_interval,
                logger=logger,
            )
            result = dict(finished_job.get("result") or {})
            job_state = str(finished_job.get("state") or "")
            entry["result_ids"] = collect_result_ids(result, finished_job)
            entry["finished_at"] = utc_now()

            if job_state == "succeeded":
                registered_path = Path(str(result.get("registered_adapter_path") or "")).expanduser().resolve()
                if not registered_path.exists():
                    raise FileNotFoundError(f"Succeeded job missing registered adapter: {registered_path}")
                mirrored_path = mirror_adapter(registered_path, mirror_root)
                output_dir = str(
                    result.get("output_dir")
                    or finished_job.get("paths", {}).get("output_dir")
                    or entry.get("training_output_dir")
                    or ""
                ).strip()
                entry.update(
                    {
                        "status": "completed",
                        "registered_path": str(registered_path),
                        "mirror_path": str(mirrored_path),
                        "generation_trigger_tag": str(result.get("generation_trigger_tag") or result.get("trigger_tag") or trigger_tag),
                        "training_output_dir": output_dir,
                        "tensor_output_dir": str(result.get("tensor_output") or finished_job.get("paths", {}).get("tensor_output") or entry.get("tensor_output_dir") or "").strip(),
                        "error": "",
                    }
                )
                summary["completed"] += 1
                logger.info(
                    "completed folder=%s trigger=%s registered=%s mirror=%s",
                    folder_name,
                    trigger_tag,
                    registered_path,
                    mirrored_path,
                )
                if cleanup_training_artifacts:
                    cleanup_entry_artifacts(entry, logger=logger)
            else:
                error = str(finished_job.get("error") or result.get("error") or f"job ended with state={job_state}")
                entry.update(
                    {
                        "status": "failed",
                        "training_output_dir": str(result.get("output_dir") or finished_job.get("paths", {}).get("output_dir") or entry.get("training_output_dir") or "").strip(),
                        "tensor_output_dir": str(result.get("tensor_output") or finished_job.get("paths", {}).get("tensor_output") or entry.get("tensor_output_dir") or "").strip(),
                        "error": error,
                    }
                )
                summary["failed"] += 1
                logger.error("failed folder=%s trigger=%s job=%s error=%s", folder_name, trigger_tag, job_id, error)
                if cleanup_training_artifacts:
                    cleanup_entry_artifacts(entry, logger=logger)
        except Exception as exc:
            entry["finished_at"] = utc_now()
            entry["status"] = "failed"
            entry["error"] = str(exc)
            summary["failed"] += 1
            logger.exception("batch exception folder=%s trigger=%s", folder_name, trigger_tag)
            if cleanup_training_artifacts:
                cleanup_entry_artifacts(entry, logger=logger)
        finally:
            entries[folder_name] = entry
            save_state(state_file, state)

    logger.info("summary total=%s completed=%s failed=%s skipped=%s", summary["total"], summary["completed"], summary["failed"], summary["skipped"])
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch train LoRA cluster folders through AceJAM one-click training.")
    parser.add_argument("--cluster-root", default=str(DEFAULT_CLUSTER_ROOT), help="Root folder containing direct cluster subfolders.")
    parser.add_argument("--mirror-root", default=str(DEFAULT_MIRROR_ROOT), help="Where successful registered adapters are mirrored.")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE), help="Batch progress state JSON file.")
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE), help="Batch log file.")
    parser.add_argument("--scratch-root", default=str(DEFAULT_SCRATCH_ROOT), help="Scratch/temp root used for TMPDIR/TMP/TEMP during batch jobs.")
    parser.add_argument("--language", default="en", help="Training language sent to one-click train.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N cluster folders.")
    parser.add_argument("--force", action="store_true", help="Retrain folders even if state says completed and artifacts exist.")
    parser.add_argument("--poll-interval", type=float, default=30.0, help="Seconds between job status polls.")
    parser.add_argument(
        "--keep-training-artifacts",
        action="store_true",
        help="Keep per-job training output directories after a successful mirror instead of cleaning them up.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    manager = get_default_training_manager()
    run_batch(
        manager=manager,
        cluster_root=Path(args.cluster_root),
        mirror_root=Path(args.mirror_root),
        state_file=Path(args.state_file),
        log_file=Path(args.log_file),
        scratch_root=Path(args.scratch_root),
        language=str(args.language or "en").strip() or "en",
        limit=args.limit,
        force=bool(args.force),
        poll_interval=float(args.poll_interval),
        cleanup_training_artifacts=not bool(args.keep_training_artifacts),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
