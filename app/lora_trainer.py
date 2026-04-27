from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import soundfile as sf
except Exception:  # pragma: no cover - dependency is installed in the app env.
    sf = None


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".aac", ".m4a"}
JOB_ACTIVE_STATES = {"queued", "running", "stopping"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str, fallback: str = "item") -> str:
    import re

    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or fallback)).strip("-._")
    return text[:90] or fallback


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def parse_int(value: Any, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def parse_float(value: Any, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def model_to_variant(model_name: str | None) -> str:
    name = (model_name or "acestep-v15-turbo").strip()
    aliases = {
        "auto": "turbo",
        "acestep-v15-turbo": "turbo",
        "acestep-v15-base": "base",
        "acestep-v15-sft": "sft",
        "acestep-v15-xl-turbo": "xl_turbo",
        "acestep-v15-xl-base": "xl_base",
        "acestep-v15-xl-sft": "xl_sft",
    }
    return aliases.get(name, name)


@dataclass
class TrainingJob:
    id: str
    kind: str
    state: str
    created_at: str
    updated_at: str
    command: list[str]
    params: dict[str, Any]
    paths: dict[str, str]
    log_path: str
    pid: int | None = None
    return_code: int | None = None
    error: str = ""
    stage: str = ""
    progress: float = 0.0
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "command": self.command,
            "params": self.params,
            "paths": self.paths,
            "log_path": self.log_path,
            "pid": self.pid,
            "return_code": self.return_code,
            "error": self.error,
            "stage": self.stage,
            "progress": self.progress,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingJob":
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            state=str(data["state"]),
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
            command=[str(part) for part in data.get("command", [])],
            params=dict(data.get("params") or {}),
            paths={str(k): str(v) for k, v in dict(data.get("paths") or {}).items()},
            log_path=str(data.get("log_path") or ""),
            pid=data.get("pid"),
            return_code=data.get("return_code"),
            error=str(data.get("error") or ""),
            stage=str(data.get("stage") or ""),
            progress=parse_float(data.get("progress"), 0.0, 0.0, 100.0),
            result=dict(data.get("result") or {}),
        )


class AceTrainingManager:
    def __init__(
        self,
        *,
        base_dir: Path,
        data_dir: Path,
        model_cache_dir: Path,
        release_models: Callable[[], None] | None = None,
        adapter_ready: Callable[[Path, float], dict[str, Any]] | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.model_cache_dir = model_cache_dir
        self.vendor_dir = base_dir / "vendor" / "ACE-Step-1.5"
        self.checkpoint_dir = model_cache_dir / "checkpoints"
        self.datasets_dir = data_dir / "lora_datasets"
        self.tensor_dir = data_dir / "lora_tensors"
        self.training_dir = data_dir / "lora_training"
        self.exports_dir = data_dir / "loras"
        self.imports_dir = data_dir / "lora_imports"
        self.jobs_dir = data_dir / "lora_jobs"
        self.release_models = release_models
        self.adapter_ready = adapter_ready
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen[str]] = {}

        for directory in [
            self.datasets_dir,
            self.tensor_dir,
            self.training_dir,
            self.exports_dir,
            self.imports_dir,
            self.jobs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        self._mark_stale_jobs()

    def status(self) -> dict[str, Any]:
        missing = self.missing_dependencies()
        vendor_ready = self.vendor_ready()
        active = self.active_job()
        return {
            "vendor_ready": vendor_ready,
            "vendor_path": str(self.vendor_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "missing_dependencies": missing,
            "ready": vendor_ready and not missing,
            "active_job": active,
            "adapter_count": len(self.list_adapters()),
            "tensorboard_runs": self.tensorboard_runs(),
        }

    def vendor_ready(self) -> bool:
        return (self.vendor_dir / "train.py").is_file() and (self.vendor_dir / "acestep" / "training_v2").is_dir()

    def missing_dependencies(self) -> list[str]:
        modules = {
            "lightning": "lightning",
            "lycoris-lora": "lycoris",
            "tensorboard": "tensorboard",
            "toml": "toml",
            "modelscope": "modelscope",
            "typer-slim": "typer",
            "peft": "peft",
            "torchao": "torchao",
        }
        missing = []
        for package, module in modules.items():
            if importlib.util.find_spec(module) is None:
                missing.append(package)
        return missing

    def require_ready(self) -> None:
        if not self.vendor_ready():
            raise RuntimeError(
                "Official ACE-Step trainer is not installed. Run Pinokio Install first so app/vendor/ACE-Step-1.5 is cloned."
            )
        missing = self.missing_dependencies()
        if missing:
            raise RuntimeError(
                "Trainer dependencies are missing: "
                + ", ".join(missing)
                + ". Run Pinokio Install/Update to install training extras."
            )
        # Patch vendor VARIANT_DIR_MAP to include XL models (upstream only has turbo/base/sft)
        try:
            from acestep.training_v2.cli.args import VARIANT_DIR_MAP
            VARIANT_DIR_MAP.setdefault("turbo_shift3", "acestep-v15-turbo-shift3")
            VARIANT_DIR_MAP.setdefault("xl_turbo", "acestep-v15-xl-turbo")
            VARIANT_DIR_MAP.setdefault("xl_base", "acestep-v15-xl-base")
            VARIANT_DIR_MAP.setdefault("xl_sft", "acestep-v15-xl-sft")
        except ImportError:
            pass

    def active_job(self) -> dict[str, Any] | None:
        with self._lock:
            for job in self._load_jobs_unlocked():
                if job.state in JOB_ACTIVE_STATES:
                    return self._public_job(job)
        return None

    def is_busy(self) -> bool:
        return self.active_job() is not None

    def scan_dataset(self, root: Path) -> dict[str, Any]:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {root}")
        csv_meta = self._load_csv_metadata(root)
        files = []
        for index, audio_path in enumerate(
            path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ):
            files.append(self._sample_from_audio(audio_path, root, csv_meta, index))
        dataset_id = uuid.uuid4().hex[:12]
        scan_path = self.datasets_dir / f"{dataset_id}.scan.json"
        scan_path.write_text(
            json.dumps({"id": dataset_id, "root": str(root), "files": files}, indent=2),
            encoding="utf-8",
        )
        return {"dataset_id": dataset_id, "root": str(root), "files": files, "dataset_path": str(scan_path)}

    def import_root_for(self, dataset_id: str) -> Path:
        return self.imports_dir / slug(dataset_id, "dataset")

    def label_entries(
        self,
        entries: list[dict[str, Any]],
        *,
        trigger_tag: str,
        language: str,
        tag_position: str = "prepend",
        genre_ratio: int | float | str = 0,
    ) -> list[dict[str, Any]]:
        """Create ACE-Step training labels without language detection or LM guessing."""
        trigger = str(trigger_tag or "").strip()
        fixed_language = str(language or "unknown").strip() or "unknown"
        position = str(tag_position or "prepend").strip().lower()
        if position not in {"prepend", "append", "replace"}:
            position = "prepend"
        ratio = parse_int(genre_ratio, 0, 0, 100)
        labeled: list[dict[str, Any]] = []
        for item in entries:
            entry = dict(item or {})
            fallback_caption = self._caption_fallback(entry)
            caption = str(entry.get("caption") or fallback_caption).strip() or fallback_caption
            caption = self._apply_trigger_tag(caption, trigger, position)
            lyrics = str(entry.get("lyrics") or "").strip() or "[Instrumental]"
            entry.update(
                {
                    "caption": caption,
                    "lyrics": lyrics,
                    "language": fixed_language,
                    "custom_tag": trigger,
                    "trigger_tag": trigger,
                    "tag_position": position,
                    "genre_ratio": ratio,
                    "label_source": entry.get("label_source")
                    or ("sidecar_metadata" if entry.get("caption_path") or entry.get("metadata_path") else "deterministic_filename"),
                    "is_instrumental": parse_bool(entry.get("is_instrumental"), lyrics.strip().lower() == "[instrumental]"),
                    "labeled": True,
                }
            )
            labeled.append(entry)
        return labeled

    def auto_epochs(self, sample_count: int) -> int:
        count = max(0, int(sample_count or 0))
        if count <= 20:
            return 800
        if count <= 100:
            return 500
        return 300

    def start_one_click_train(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.require_ready()
        trigger_tag = str(payload.get("trigger_tag") or payload.get("custom_tag") or "").strip()
        language = str(payload.get("language") or payload.get("vocal_language") or "").strip()
        if not trigger_tag:
            raise ValueError("trigger_tag is required")
        if not language:
            raise ValueError("language is required")

        dataset_id = slug(str(payload.get("dataset_id") or payload.get("import_id") or uuid.uuid4().hex[:12]), "dataset")
        import_root = Path(str(payload.get("import_root") or "")).expanduser() if payload.get("import_root") else self.import_root_for(dataset_id)
        if not import_root.is_dir():
            raise FileNotFoundError(f"Imported dataset folder not found: {import_root}")

        with self._lock:
            active = next((job for job in self._load_jobs_unlocked() if job.state in JOB_ACTIVE_STATES), None)
            if active:
                raise RuntimeError(f"Training job already active: {active.id}")
            job_id = uuid.uuid4().hex[:12]
            job_dir = self.jobs_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            params = self._one_click_params(payload, dataset_id=dataset_id, import_root=import_root)
            job = TrainingJob(
                id=job_id,
                kind="one_click_train",
                state="queued",
                created_at=utc_now(),
                updated_at=utc_now(),
                command=["acejam-one-click-lora"],
                params=params,
                paths={"import_root": str(import_root)},
                log_path=str(job_dir / "job.log"),
                stage="queued",
                progress=0.0,
            )
            self._write_job_unlocked(job)

        thread = threading.Thread(target=self._run_one_click_job, args=(job,), daemon=True)
        thread.start()
        return self.get_job(job.id)

    def save_dataset(
        self,
        entries: list[dict[str, Any]],
        *,
        dataset_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not entries:
            raise ValueError("No dataset entries supplied")

        dataset_id = slug(dataset_id or f"dataset-{uuid.uuid4().hex[:8]}", "dataset")
        samples = [self._official_sample(entry) for entry in entries]
        samples = [sample for sample in samples if sample.get("audio_path")]
        if not samples:
            raise ValueError("Dataset contains no valid audio paths")

        dataset = {
            "metadata": {
                "tag_position": str((metadata or {}).get("tag_position") or "prepend"),
                "genre_ratio": parse_int((metadata or {}).get("genre_ratio"), 0, 0, 100),
                "custom_tag": str((metadata or {}).get("custom_tag") or ""),
                "language": str((metadata or {}).get("language") or "unknown"),
                "one_click_train": parse_bool((metadata or {}).get("one_click_train"), False),
            },
            "samples": samples,
        }
        dataset_path = self.datasets_dir / f"{dataset_id}.json"
        dataset_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
        return {
            "dataset_id": dataset_id,
            "dataset_path": str(dataset_path),
            "sample_count": len(samples),
            "metadata": dataset["metadata"],
            "samples": samples,
        }

    def start_preprocess(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.require_ready()
        dataset_json = self._resolve_dataset_json(payload)
        audio_dir = str(Path(str(payload.get("audio_dir") or payload.get("path") or "")).expanduser()) if payload.get("audio_dir") or payload.get("path") else ""
        if not dataset_json and not audio_dir:
            raise ValueError("Preprocess requires dataset_json, dataset_id, or audio_dir")

        job_id = uuid.uuid4().hex[:12]
        tensor_output = Path(str(payload.get("tensor_output") or "")).expanduser() if payload.get("tensor_output") else self.tensor_dir / job_id
        variant = model_to_variant(str(payload.get("model_variant") or payload.get("song_model") or "acestep-v15-turbo"))
        command = [
            sys.executable,
            "-m",
            "acestep.training_v2.cli.train_fixed",
            "--plain",
            "--yes",
            "--preprocess",
            "--checkpoint-dir",
            str(self.checkpoint_dir),
            "--model-variant",
            variant,
            "--tensor-output",
            str(tensor_output),
            "--max-duration",
            str(parse_float(payload.get("max_duration"), 240.0, 10.0, 600.0)),
            "--device",
            str(payload.get("device") or "auto"),
            "--precision",
            str(payload.get("precision") or "auto"),
        ]
        if dataset_json:
            command.extend(["--dataset-json", dataset_json])
        if audio_dir:
            command.extend(["--audio-dir", audio_dir])
        return self._start_job(
            kind="preprocess",
            command=command,
            params={"model_variant": variant, "dataset_json": dataset_json, "audio_dir": audio_dir},
            paths={"tensor_output": str(tensor_output), "dataset_json": dataset_json, "audio_dir": audio_dir},
        )

    def start_train(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.require_ready()
        dataset_dir = Path(str(payload.get("dataset_dir") or payload.get("tensor_dir") or "")).expanduser()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Tensor dataset directory not found: {dataset_dir}")
        job_id = uuid.uuid4().hex[:12]
        adapter_type = str(payload.get("adapter_type") or "lora").lower()
        if adapter_type not in {"lora", "lokr"}:
            raise ValueError("adapter_type must be lora or lokr")
        variant = model_to_variant(str(payload.get("model_variant") or payload.get("song_model") or "acestep-v15-turbo"))
        output_dir = Path(str(payload.get("output_dir") or "")).expanduser() if payload.get("output_dir") else self.training_dir / f"{slug(adapter_type)}-{job_id}"
        log_dir = output_dir / "runs"
        command = [
            sys.executable,
            "train.py",
            "--plain",
            "--yes",
            "fixed",
            "--checkpoint-dir",
            str(self.checkpoint_dir),
            "--model-variant",
            variant,
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--adapter-type",
            adapter_type,
            "--batch-size",
            str(parse_int(payload.get("train_batch_size", payload.get("batch_size")), 1, 1, 64)),
            "--gradient-accumulation",
            str(parse_int(payload.get("gradient_accumulation"), 4, 1, 128)),
            "--epochs",
            str(parse_int(payload.get("train_epochs", payload.get("epochs")), 10, 1, 10000)),
            "--save-every",
            str(parse_int(payload.get("save_every_n_epochs", payload.get("save_every")), 5, 1, 10000)),
            "--lr",
            str(parse_float(payload.get("learning_rate"), 1e-4, 1e-7, 1.0)),
            "--shift",
            str(parse_float(payload.get("training_shift", payload.get("shift")), 3.0, 0.1, 10.0)),
            "--seed",
            str(parse_int(payload.get("training_seed", payload.get("seed")), 42, 0, 2**31 - 1)),
            "--num-inference-steps",
            str(parse_int(payload.get("num_inference_steps"), 8, 1, 200)),
            "--warmup-steps",
            str(parse_int(payload.get("warmup_steps"), 100, 0, 100000)),
            "--weight-decay",
            str(parse_float(payload.get("weight_decay"), 0.01, 0.0, 1.0)),
            "--max-grad-norm",
            str(parse_float(payload.get("max_grad_norm"), 1.0, 0.0, 100.0)),
            "--optimizer-type",
            str(payload.get("optimizer_type") or "adamw"),
            "--scheduler-type",
            str(payload.get("scheduler_type") or "cosine"),
            "--log-dir",
            str(log_dir),
            "--log-every",
            str(parse_int(payload.get("log_every"), 10, 1, 100000)),
            "--log-heavy-every",
            str(parse_int(payload.get("log_heavy_every"), 50, 1, 100000)),
            "--sample-every-n-epochs",
            str(parse_int(payload.get("sample_every_n_epochs"), 0, 0, 10000)),
            "--device",
            str(payload.get("device") or "auto"),
            "--precision",
            str(payload.get("precision") or "auto"),
        ]
        if parse_bool(payload.get("offload_encoder"), False):
            command.append("--offload-encoder")
        else:
            command.append("--no-offload-encoder")
        if parse_bool(payload.get("gradient_checkpointing"), True):
            command.append("--gradient-checkpointing")
        else:
            command.append("--no-gradient-checkpointing")

        if adapter_type == "lokr":
            command.extend(
                [
                    "--lokr-linear-dim",
                    str(parse_int(payload.get("lokr_linear_dim"), 64, 1, 256)),
                    "--lokr-linear-alpha",
                    str(parse_int(payload.get("lokr_linear_alpha"), 128, 1, 512)),
                    "--lokr-factor",
                    str(parse_int(payload.get("lokr_factor"), -1, -1, 64)),
                ]
            )
            if parse_bool(payload.get("lokr_decompose_both"), False):
                command.append("--lokr-decompose-both")
            if parse_bool(payload.get("lokr_use_tucker"), False):
                command.append("--lokr-use-tucker")
            if parse_bool(payload.get("lokr_use_scalar"), False):
                command.append("--lokr-use-scalar")
            if parse_bool(payload.get("lokr_weight_decompose"), True):
                command.append("--lokr-weight-decompose")
        else:
            command.extend(
                [
                    "--rank",
                    str(parse_int(payload.get("rank"), 64, 1, 512)),
                    "--alpha",
                    str(parse_int(payload.get("alpha"), 128, 1, 1024)),
                    "--dropout",
                    str(parse_float(payload.get("dropout"), 0.1, 0.0, 1.0)),
                    "--attention-type",
                    str(payload.get("attention_type") or "both"),
                ]
            )

        return self._start_job(
            kind="train",
            command=command,
            params={"adapter_type": adapter_type, "model_variant": variant},
            paths={"dataset_dir": str(dataset_dir), "output_dir": str(output_dir), "final_adapter": str(output_dir / "final"), "log_dir": str(log_dir)},
        )

    def start_estimate(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.require_ready()
        dataset_dir = Path(str(payload.get("dataset_dir") or payload.get("tensor_dir") or "")).expanduser()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Tensor dataset directory not found: {dataset_dir}")
        job_id = uuid.uuid4().hex[:12]
        variant = model_to_variant(str(payload.get("model_variant") or payload.get("song_model") or "acestep-v15-turbo"))
        output = self.training_dir / f"estimate-{job_id}.json"
        command = [
            sys.executable,
            "train.py",
            "--plain",
            "--yes",
            "estimate",
            "--checkpoint-dir",
            str(self.checkpoint_dir),
            "--model-variant",
            variant,
            "--dataset-dir",
            str(dataset_dir),
            "--batch-size",
            str(parse_int(payload.get("batch_size"), 1, 1, 64)),
            "--estimate-batches",
            str(parse_int(payload.get("estimate_batches"), 5, 1, 1000)),
            "--top-k",
            str(parse_int(payload.get("top_k"), 16, 1, 256)),
            "--granularity",
            str(payload.get("granularity") or "module"),
            "--output",
            str(output),
            "--device",
            str(payload.get("device") or "auto"),
            "--precision",
            str(payload.get("precision") or "auto"),
        ]
        return self._start_job(
            kind="estimate",
            command=command,
            params={"model_variant": variant},
            paths={"dataset_dir": str(dataset_dir), "estimate_output": str(output)},
        )

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._public_job(job) for job in self._load_jobs_unlocked()]

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return self._public_job(self._read_job_unlocked(job_id))

    def read_log(self, job_id: str, tail: int = 400) -> dict[str, Any]:
        with self._lock:
            job = self._read_job_unlocked(job_id)
        log_path = Path(job.log_path)
        if not log_path.is_file():
            return {"job_id": job_id, "log": ""}
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"job_id": job_id, "log": "\n".join(lines[-max(1, tail) :])}

    def stop_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._read_job_unlocked(job_id)
            process = self._processes.get(job_id)
            if job.state not in JOB_ACTIVE_STATES:
                return self._public_job(job)
            job.state = "stopping"
            job.updated_at = utc_now()
            self._write_job_unlocked(job)
        if process and process.poll() is None:
            process.terminate()
        return self.get_job(job_id)

    def list_adapters(self) -> list[dict[str, Any]]:
        roots = [self.exports_dir, self.training_dir]
        adapters: list[dict[str, Any]] = []
        for root in roots:
            if not root.exists():
                continue
            for child in sorted(root.rglob("*")):
                if not child.is_dir():
                    continue
                has_lora = (child / "adapter_config.json").is_file() and (
                    (child / "adapter_model.safetensors").is_file() or (child / "adapter_model.bin").is_file()
                )
                has_lokr = (child / "lokr_weights.safetensors").is_file()
                if not (has_lora or has_lokr):
                    continue
                meta: dict[str, Any] = {}
                meta_path = child / "acejam_adapter.json"
                if meta_path.is_file():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        meta = {}
                adapters.append(
                    {
                        "name": child.name,
                        "path": str(child),
                        "adapter_type": "lokr" if has_lokr and not has_lora else "lora",
                        "source": "exports" if root == self.exports_dir else "training",
                        "updated_at": datetime.fromtimestamp(child.stat().st_mtime, timezone.utc).isoformat(),
                        "trigger_tag": meta.get("trigger_tag", ""),
                        "language": meta.get("language", ""),
                        "model_variant": meta.get("model_variant", ""),
                        "song_model": meta.get("song_model", ""),
                        "sample_count": meta.get("sample_count"),
                        "metadata": meta,
                    }
                )
        return adapters

    def tensorboard_runs(self) -> list[dict[str, str]]:
        runs: list[dict[str, str]] = []
        for run_dir in sorted(self.training_dir.rglob("runs")):
            if not run_dir.is_dir():
                continue
            if not any(path.name.startswith("events.out.tfevents") for path in run_dir.rglob("*") if path.is_file()):
                continue
            runs.append(
                {
                    "path": str(run_dir),
                    "name": run_dir.parent.name,
                    "updated_at": datetime.fromtimestamp(run_dir.stat().st_mtime, timezone.utc).isoformat(),
                }
            )
        return runs[:20]

    def export_adapter(self, source: Path, name: str | None = None) -> dict[str, Any]:
        source = source.expanduser()
        if not source.exists():
            raise FileNotFoundError(f"LoRA source path not found: {source}")
        export_id = slug(name or source.name, "adapter")
        target = self.exports_dir / export_id
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target / source.name)
        return {"success": True, "path": str(target), "adapter": target.name}

    def _one_click_params(self, payload: dict[str, Any], *, dataset_id: str, import_root: Path) -> dict[str, Any]:
        sample_count = parse_int(payload.get("sample_count"), 0, 0, None)
        epochs = payload.get("train_epochs", payload.get("epochs"))
        return {
            "dataset_id": dataset_id,
            "import_root": str(import_root),
            "trigger_tag": str(payload.get("trigger_tag") or payload.get("custom_tag") or "").strip(),
            "language": str(payload.get("language") or payload.get("vocal_language") or "").strip(),
            "adapter_type": str(payload.get("adapter_type") or "lora").strip().lower(),
            "tag_position": str(payload.get("tag_position") or "prepend").strip().lower(),
            "genre_ratio": parse_int(payload.get("genre_ratio"), 0, 0, 100),
            "song_model": str(payload.get("song_model") or "acestep-v15-sft").strip(),
            "model_variant": model_to_variant(str(payload.get("model_variant") or payload.get("song_model") or "acestep-v15-sft")),
            "train_batch_size": parse_int(payload.get("train_batch_size", payload.get("batch_size")), 1, 1, 64),
            "gradient_accumulation": parse_int(payload.get("gradient_accumulation"), 4, 1, 128),
            "rank": parse_int(payload.get("rank"), 64, 1, 512),
            "alpha": parse_int(payload.get("alpha"), 128, 1, 1024),
            "dropout": parse_float(payload.get("dropout"), 0.1, 0.0, 1.0),
            "training_shift": parse_float(payload.get("training_shift", payload.get("shift")), 3.0, 0.1, 10.0),
            "training_seed": parse_int(payload.get("training_seed", payload.get("seed")), 42, 0, 2**31 - 1),
            "train_epochs": parse_int(epochs, self.auto_epochs(sample_count), 1, 10000) if epochs not in [None, "", "auto"] else None,
            "save_every_n_epochs": parse_int(payload.get("save_every_n_epochs", payload.get("save_every")), 25, 1, 10000),
            "learning_rate": parse_float(payload.get("learning_rate"), 1e-4, 1e-7, 1.0),
            "max_duration": parse_float(payload.get("max_duration"), 240.0, 10.0, 600.0),
            "device": str(payload.get("device") or "auto"),
            "precision": str(payload.get("precision") or "auto"),
            "auto_load": parse_bool(payload.get("auto_load"), True),
            "lora_scale": parse_float(payload.get("lora_scale"), 1.0, 0.0, 1.0),
            "use_official_lm_labels": parse_bool(payload.get("use_official_lm_labels"), False),
        }

    def _run_one_click_job(self, job: TrainingJob) -> None:
        log_path = Path(job.log_path)
        try:
            if self.release_models is not None:
                self.release_models()
            params = dict(job.params)
            import_root = Path(params["import_root"]).expanduser()
            dataset_id = str(params["dataset_id"])
            trigger = str(params["trigger_tag"])
            language = str(params["language"])
            adapter_type = str(params.get("adapter_type") or "lora").lower()
            if adapter_type not in {"lora", "lokr"}:
                raise ValueError("adapter_type must be lora or lokr")

            self._set_job_state(job.id, state="running", stage="import", progress=5)
            self._append_log(log_path, f"[import] scanning {import_root}\n")
            scanned = self.scan_dataset(import_root)
            files = list(scanned.get("files") or [])
            if not files:
                raise ValueError("No supported audio files found in the imported dataset")

            self._set_job_state(job.id, stage="label", progress=16)
            labels = self.label_entries(
                files,
                trigger_tag=trigger,
                language=language,
                tag_position=str(params.get("tag_position") or "prepend"),
                genre_ratio=params.get("genre_ratio", 0),
            )
            epochs = params.get("train_epochs") or self.auto_epochs(len(labels))
            params["train_epochs"] = epochs

            self._set_job_state(job.id, stage="save_dataset", progress=24)
            saved = self.save_dataset(
                labels,
                dataset_id=dataset_id,
                metadata={
                    "custom_tag": trigger,
                    "tag_position": params.get("tag_position") or "prepend",
                    "genre_ratio": params.get("genre_ratio") or 0,
                    "language": language,
                    "one_click_train": True,
                },
            )
            dataset_json = saved["dataset_path"]
            tensor_output = self.tensor_dir / job.id
            preprocess_command = [
                sys.executable,
                "-m",
                "acestep.training_v2.cli.train_fixed",
                "--plain",
                "--yes",
                "--preprocess",
                "--checkpoint-dir",
                str(self.checkpoint_dir),
                "--model-variant",
                str(params["model_variant"]),
                "--tensor-output",
                str(tensor_output),
                "--max-duration",
                str(params["max_duration"]),
                "--device",
                str(params["device"]),
                "--precision",
                str(params["precision"]),
                "--dataset-json",
                dataset_json,
                "--audio-dir",
                str(import_root),
            ]
            self._set_job_state(
                job.id,
                stage="preprocess",
                progress=34,
                paths={"import_root": str(import_root), "dataset_json": dataset_json, "tensor_output": str(tensor_output)},
                result={"sample_count": len(labels), "epochs": epochs},
            )
            self._run_command_step(job.id, preprocess_command, log_path, stage="preprocess")

            output_dir = self.training_dir / f"{slug(trigger or adapter_type)}-{job.id}"
            log_dir = output_dir / "runs"
            train_command = [
                sys.executable,
                "train.py",
                "--plain",
                "--yes",
                "fixed",
                "--checkpoint-dir",
                str(self.checkpoint_dir),
                "--model-variant",
                str(params["model_variant"]),
                "--dataset-dir",
                str(tensor_output),
                "--output-dir",
                str(output_dir),
                "--adapter-type",
                adapter_type,
                "--batch-size",
                str(params["train_batch_size"]),
                "--gradient-accumulation",
                str(params["gradient_accumulation"]),
                "--epochs",
                str(epochs),
                "--save-every",
                str(params["save_every_n_epochs"]),
                "--lr",
                str(params["learning_rate"]),
                "--shift",
                str(params["training_shift"]),
                "--seed",
                str(params["training_seed"]),
                "--num-inference-steps",
                "8",
                "--warmup-steps",
                "100",
                "--weight-decay",
                "0.01",
                "--max-grad-norm",
                "1.0",
                "--optimizer-type",
                "adamw",
                "--scheduler-type",
                "cosine",
                "--log-dir",
                str(log_dir),
                "--log-every",
                "10",
                "--log-heavy-every",
                "50",
                "--sample-every-n-epochs",
                "0",
                "--device",
                str(params["device"]),
                "--precision",
                str(params["precision"]),
                "--gradient-checkpointing",
                "--no-offload-encoder",
            ]
            if adapter_type == "lokr":
                train_command.extend(["--lokr-linear-dim", "64", "--lokr-linear-alpha", "128", "--lokr-factor", "-1", "--lokr-weight-decompose"])
            else:
                train_command.extend(
                    [
                        "--rank",
                        str(params["rank"]),
                        "--alpha",
                        str(params["alpha"]),
                        "--dropout",
                        str(params["dropout"]),
                        "--attention-type",
                        "both",
                    ]
                )
            self._set_job_state(
                job.id,
                stage="train",
                progress=58,
                paths={
                    "import_root": str(import_root),
                    "dataset_json": dataset_json,
                    "tensor_output": str(tensor_output),
                    "output_dir": str(output_dir),
                    "final_adapter": str(output_dir / "final"),
                    "log_dir": str(log_dir),
                },
                result={"sample_count": len(labels), "epochs": epochs},
            )
            self._run_command_step(job.id, train_command, log_path, stage="train")

            final_adapter = output_dir / "final"
            if not final_adapter.exists():
                raise FileNotFoundError(f"Training finished but no final adapter was found at {final_adapter}")

            self._set_job_state(job.id, stage="register", progress=90)
            export = self.export_adapter(final_adapter, f"{slug(trigger, 'adapter')}-{job.id}")
            registered = Path(str(export["path"]))
            metadata = {
                "trigger_tag": trigger,
                "language": language,
                "adapter_type": adapter_type,
                "model_variant": params["model_variant"],
                "song_model": params["song_model"],
                "dataset_id": dataset_id,
                "sample_count": len(labels),
                "epochs": epochs,
                "trained_at": utc_now(),
            }
            (registered / "acejam_adapter.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            load_status: dict[str, Any] = {"requested": False}
            if params.get("auto_load") and self.adapter_ready is not None:
                self._set_job_state(job.id, stage="load", progress=96)
                load_status = self.adapter_ready(registered, float(params.get("lora_scale") or 1.0))

            with self._lock:
                current = self._read_job_unlocked(job.id)
                current.state = "succeeded"
                current.stage = "complete"
                current.progress = 100.0
                current.return_code = 0
                current.updated_at = utc_now()
                current.result = {
                    "dataset_id": dataset_id,
                    "dataset_json": dataset_json,
                    "tensor_output": str(tensor_output),
                    "output_dir": str(output_dir),
                    "final_adapter": str(final_adapter),
                    "registered_adapter_path": str(registered),
                    "sample_count": len(labels),
                    "epochs": epochs,
                    "auto_load": bool(params.get("auto_load")),
                    "use_lora": bool(load_status.get("success", False)),
                    "load_status": load_status,
                    "labels": labels[:5],
                }
                self._write_job_unlocked(current)
            self._append_log(log_path, f"\n[complete] adapter registered at {registered}\n")
        except Exception as exc:
            with self._lock:
                try:
                    current = self._read_job_unlocked(job.id)
                except Exception:
                    current = job
                current.state = "failed"
                current.error = str(exc)
                current.updated_at = utc_now()
                self._write_job_unlocked(current)
            self._append_log(log_path, f"\n[failed] {exc}\n")

    def _run_command_step(self, job_id: str, command: list[str], log_path: Path, *, stage: str) -> None:
        env = self._training_env()
        self._append_log(log_path, f"\n[{stage}] $ {' '.join(command)}\n\n")
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=str(self.vendor_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self._lock:
                self._processes[job_id] = process
                current = self._read_job_unlocked(job_id)
                current.pid = process.pid
                current.updated_at = utc_now()
                self._write_job_unlocked(current)
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
            return_code = process.wait()
        with self._lock:
            self._processes.pop(job_id, None)
            current = self._read_job_unlocked(job_id)
            current.pid = None
            current.return_code = return_code
            current.updated_at = utc_now()
            self._write_job_unlocked(current)
        if return_code != 0:
            raise RuntimeError(f"{stage} exited with code {return_code}")

    def _training_env(self) -> dict[str, str]:
        env = os.environ.copy()
        py_paths = [
            str(self.vendor_dir),
            str(self.vendor_dir / "acestep" / "third_parts" / "nano-vllm"),
        ]
        if env.get("PYTHONPATH"):
            py_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(py_paths)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
        env.setdefault("HF_MODULES_CACHE", str(self.model_cache_dir / "hf_modules"))
        env.setdefault("MPLCONFIGDIR", str(self.model_cache_dir / "matplotlib"))
        return env

    def _set_job_state(
        self,
        job_id: str,
        *,
        state: str | None = None,
        stage: str | None = None,
        progress: float | None = None,
        paths: dict[str, str] | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            current = self._read_job_unlocked(job_id)
            if state is not None:
                current.state = state
            if stage is not None:
                current.stage = stage
            if progress is not None:
                current.progress = parse_float(progress, 0.0, 0.0, 100.0)
            if paths:
                current.paths.update({str(k): str(v) for k, v in paths.items()})
            if result:
                current.result.update(result)
            current.updated_at = utc_now()
            self._write_job_unlocked(current)

    def _append_log(self, log_path: Path, text: str) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text)

    def _caption_fallback(self, entry: dict[str, Any]) -> str:
        relative = str(entry.get("relative_path") or entry.get("filename") or entry.get("path") or "sample")
        path = Path(relative)
        parts = [part for part in path.with_suffix("").parts if part not in {".", ""}]
        text = " ".join(parts[-3:]) if parts else path.stem
        return text.replace("_", " ").replace("-", " ").strip() or "training sample"

    def _apply_trigger_tag(self, caption: str, trigger_tag: str, tag_position: str) -> str:
        caption = str(caption or "").strip()
        trigger = str(trigger_tag or "").strip()
        if not trigger:
            return caption
        if trigger.lower() in caption.lower():
            return caption
        if tag_position == "replace":
            return trigger
        if tag_position == "append":
            return f"{caption}, {trigger}" if caption else trigger
        return f"{trigger}, {caption}" if caption else trigger

    def _start_job(
        self,
        *,
        kind: str,
        command: list[str],
        params: dict[str, Any],
        paths: dict[str, str],
    ) -> dict[str, Any]:
        with self._lock:
            active = next((job for job in self._load_jobs_unlocked() if job.state in JOB_ACTIVE_STATES), None)
            if active:
                raise RuntimeError(f"Training job already active: {active.id}")
            job_id = uuid.uuid4().hex[:12]
            job_dir = self.jobs_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            job = TrainingJob(
                id=job_id,
                kind=kind,
                state="queued",
                created_at=utc_now(),
                updated_at=utc_now(),
                command=command,
                params=params,
                paths=paths,
                log_path=str(job_dir / "job.log"),
            )
            self._write_job_unlocked(job)

        thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        thread.start()
        return self.get_job(job.id)

    def _run_job(self, job: TrainingJob) -> None:
        try:
            if self.release_models is not None:
                self.release_models()
            env = os.environ.copy()
            py_paths = [
                str(self.vendor_dir),
                str(self.vendor_dir / "acestep" / "third_parts" / "nano-vllm"),
            ]
            if env.get("PYTHONPATH"):
                py_paths.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(py_paths)
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
            env.setdefault("HF_MODULES_CACHE", str(self.model_cache_dir / "hf_modules"))
            env.setdefault("MPLCONFIGDIR", str(self.model_cache_dir / "matplotlib"))

            with self._lock:
                job.state = "running"
                job.updated_at = utc_now()
                self._write_job_unlocked(job)

            log_path = Path(job.log_path)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"$ {' '.join(job.command)}\n\n")
                log.flush()
                process = subprocess.Popen(
                    job.command,
                    cwd=str(self.vendor_dir),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                with self._lock:
                    self._processes[job.id] = process
                    job.pid = process.pid
                    job.updated_at = utc_now()
                    self._write_job_unlocked(job)

                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()

                return_code = process.wait()

            with self._lock:
                self._processes.pop(job.id, None)
                current = self._read_job_unlocked(job.id)
                current.return_code = return_code
                current.updated_at = utc_now()
                if current.state == "stopping":
                    current.state = "stopped"
                elif return_code == 0:
                    current.state = "succeeded"
                    current.result = self._job_result(current)
                else:
                    current.state = "failed"
                    current.error = f"Trainer exited with code {return_code}"
                self._write_job_unlocked(current)
        except Exception as exc:
            with self._lock:
                try:
                    current = self._read_job_unlocked(job.id)
                except Exception:
                    current = job
                current.state = "failed"
                current.error = str(exc)
                current.updated_at = utc_now()
                self._write_job_unlocked(current)

    def _job_result(self, job: TrainingJob) -> dict[str, Any]:
        if job.kind == "train":
            final_adapter = job.paths.get("final_adapter", "")
            return {"final_adapter": final_adapter, "adapter_exists": Path(final_adapter).exists()}
        if job.kind == "preprocess":
            output = Path(job.paths.get("tensor_output", ""))
            tensors = len(list(output.glob("*.pt"))) if output.is_dir() else 0
            return {"tensor_output": str(output), "tensor_count": tensors}
        if job.kind == "estimate":
            output = Path(job.paths.get("estimate_output", ""))
            return {"estimate_output": str(output), "exists": output.is_file()}
        return {}

    def _public_job(self, job: TrainingJob) -> dict[str, Any]:
        data = job.to_dict()
        data["log_url"] = f"/api/lora/jobs/{job.id}/log"
        return data

    def _mark_stale_jobs(self) -> None:
        with self._lock:
            for job in self._load_jobs_unlocked():
                if job.state in JOB_ACTIVE_STATES:
                    job.state = "failed"
                    job.error = "Job was interrupted by an app restart"
                    job.updated_at = utc_now()
                    self._write_job_unlocked(job)

    def _load_jobs_unlocked(self) -> list[TrainingJob]:
        jobs = []
        for job_path in sorted(self.jobs_dir.glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                jobs.append(TrainingJob.from_dict(json.loads(job_path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return jobs

    def _read_job_unlocked(self, job_id: str) -> TrainingJob:
        job_id = slug(job_id, "job")
        path = self.jobs_dir / job_id / "job.json"
        if not path.is_file():
            raise FileNotFoundError(f"Job not found: {job_id}")
        return TrainingJob.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _write_job_unlocked(self, job: TrainingJob) -> None:
        job_dir = self.jobs_dir / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")

    def _resolve_dataset_json(self, payload: dict[str, Any]) -> str:
        if payload.get("dataset_json"):
            path = Path(str(payload["dataset_json"])).expanduser()
            if path.is_file():
                return str(path)
        if payload.get("dataset_id"):
            dataset_id = slug(str(payload["dataset_id"]), "dataset")
            path = self.datasets_dir / f"{dataset_id}.json"
            if path.is_file():
                return str(path)
        return ""

    def _load_csv_metadata(self, root: Path) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for csv_path in sorted(root.glob("*.csv")):
            try:
                with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        keys = {key.lower().strip(): key for key in row.keys() if key}
                        filename = row.get(keys.get("file", ""), "") or row.get(keys.get("filename", ""), "")
                        if not filename:
                            continue
                        rows[Path(filename).name] = row
            except Exception:
                continue
        return rows

    def _sample_from_audio(self, audio_path: Path, root: Path, csv_meta: dict[str, dict[str, Any]], index: int) -> dict[str, Any]:
        stem = audio_path.stem
        lyrics_path = audio_path.with_name(f"{stem}.lyrics.txt")
        legacy_lyrics_path = audio_path.with_suffix(".txt")
        caption_path = audio_path.with_name(f"{stem}.caption.txt")
        json_path = audio_path.with_suffix(".json")
        metadata: dict[str, Any] = {}
        if json_path.is_file():
            try:
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        csv_row = csv_meta.get(audio_path.name, {})
        lyrics = ""
        if lyrics_path.is_file():
            lyrics = lyrics_path.read_text(encoding="utf-8", errors="replace").strip()
        elif legacy_lyrics_path.is_file():
            lyrics = legacy_lyrics_path.read_text(encoding="utf-8", errors="replace").strip()
        else:
            lyrics = str(metadata.get("lyrics") or "[Instrumental]")
        caption = ""
        if caption_path.is_file():
            caption = caption_path.read_text(encoding="utf-8", errors="replace").strip()
        caption = caption or str(metadata.get("caption") or csv_row.get("Caption") or stem.replace("_", " ").replace("-", " "))
        duration = None
        if sf is not None:
            try:
                info = sf.info(str(audio_path))
                duration = round(info.frames / info.samplerate, 3)
            except Exception:
                duration = None
        return {
            "id": f"{index + 1}-{slug(stem)}",
            "filename": audio_path.name,
            "path": str(audio_path),
            "relative_path": str(audio_path.relative_to(root)) if root in audio_path.parents else audio_path.name,
            "lyrics_path": str(lyrics_path if lyrics_path.is_file() else legacy_lyrics_path if legacy_lyrics_path.is_file() else ""),
            "caption_path": str(caption_path if caption_path.is_file() else ""),
            "metadata_path": str(json_path if json_path.is_file() else ""),
            "caption": caption,
            "lyrics": lyrics,
            "genre": str(metadata.get("genre") or csv_row.get("Genre") or ""),
            "bpm": metadata.get("bpm") or csv_row.get("BPM") or None,
            "keyscale": metadata.get("keyscale") or metadata.get("key_scale") or csv_row.get("Key") or "",
            "timesignature": metadata.get("timesignature") or metadata.get("time_signature") or "4",
            "language": metadata.get("language") or metadata.get("vocal_language") or "unknown",
            "duration": duration or metadata.get("duration") or 0,
            "is_instrumental": parse_bool(metadata.get("is_instrumental"), lyrics.strip().lower() == "[instrumental]"),
            "labeled": bool(caption),
        }

    def _official_sample(self, entry: dict[str, Any]) -> dict[str, Any]:
        audio_path = Path(str(entry.get("audio_path") or entry.get("path") or entry.get("filename") or "")).expanduser()
        if not audio_path.is_file():
            return {}
        lyrics = str(entry.get("lyrics") or "[Instrumental]").strip() or "[Instrumental]"
        return {
            "filename": audio_path.name,
            "audio_path": str(audio_path),
            "caption": str(entry.get("caption") or audio_path.stem.replace("_", " ").replace("-", " ")),
            "lyrics": lyrics,
            "genre": str(entry.get("genre") or ""),
            "bpm": None if entry.get("bpm") in [None, "", "auto"] else parse_int(entry.get("bpm"), 120, 30, 300),
            "keyscale": str(entry.get("keyscale") or entry.get("key_scale") or ""),
            "timesignature": str(entry.get("timesignature") or entry.get("time_signature") or "4"),
            "language": str(entry.get("language") or entry.get("vocal_language") or "unknown"),
            "duration": parse_float(entry.get("duration"), 0.0, 0.0, None),
            "is_instrumental": parse_bool(entry.get("is_instrumental"), lyrics.lower() == "[instrumental]"),
            "custom_tag": str(entry.get("custom_tag") or ""),
            "prompt_override": entry.get("prompt_override") or None,
        }
