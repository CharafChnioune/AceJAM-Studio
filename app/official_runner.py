from __future__ import annotations

import json
import os
import inspect
import platform
import sys
import traceback
from pathlib import Path
from typing import Any


def _prepare_vendor_imports(vendor_dir: Path) -> None:
    app_dir = Path(__file__).resolve().parent
    filtered = []
    for entry in sys.path:
        try:
            if Path(entry or ".").resolve() == app_dir:
                continue
        except Exception:
            pass
        filtered.append(entry)
    sys.path[:] = [str(vendor_dir)] + filtered


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if k != "tensor"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "shape"):
        return {"shape": list(value.shape)}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _parse_seeds(value: Any) -> list[int] | None:
    if value in (None, "", "-1"):
        return None
    if isinstance(value, list):
        raw = value
    else:
        raw = [item.strip() for item in str(value).split(",")]
    seeds = []
    for item in raw:
        if str(item).strip():
            seeds.append(int(float(item)))
    return seeds or None


def _resolve_backend(requested: str) -> str:
    value = (requested or "auto").strip().lower()
    if value in {"pt", "vllm"}:
        return value
    return "pt"

def _disable_acestep_mlx_backends(handler_cls: Any) -> None:
    def _disabled_mlx_backends(self: Any, *args: Any, **kwargs: Any) -> tuple[str, str]:
        self.mlx_decoder = None
        self.use_mlx_dit = False
        self.mlx_dit_compiled = False
        self.mlx_vae = None
        self.use_mlx_vae = False
        return "Disabled by AceJAM (PyTorch/MPS)", "Disabled by AceJAM (PyTorch/MPS)"

    handler_cls._initialize_mlx_backends = _disabled_mlx_backends


def _patch_mlx_thread_stream(handler_cls: Any) -> None:
    """Patch the generation thread to initialize the MLX stream for child threads.

    MLX Metal streams are thread-local. The diffusion timeout mechanism runs
    the actual generation in a threading.Thread, but MLX was initialized in the
    main thread. This patches _run_generate_music_service_with_progress to set
    mx.set_default_device(mx.gpu) inside the child thread before running.
    """
    try:
        from acestep.core.generation.handler.generate_music_execute import GenerateMusicExecuteMixin
    except ImportError:
        return

    _orig = getattr(GenerateMusicExecuteMixin, "_run_generate_music_service_with_progress", None)
    if _orig is None:
        return

    import threading as _threading

    def _patched(self, *args, **kwargs):
        # Monkey-patch threading.Thread to inject MLX stream setup
        _OrigThread = _threading.Thread

        class _MLXThread(_OrigThread):
            def run(self):
                if getattr(self._target, "__name__", "") == "_service_target":
                    try:
                        import mlx.core as mx
                        mx.set_default_device(mx.gpu)
                    except Exception:
                        pass
                super().run()

        _threading.Thread = _MLXThread
        try:
            return _orig(self, *args, **kwargs)
        finally:
            _threading.Thread = _OrigThread

    GenerateMusicExecuteMixin._run_generate_music_service_with_progress = _patched


def _none_if_auto(value: Any) -> Any:
    text = str(value or "auto").strip().lower()
    return None if text in {"", "auto", "none"} else value


def _bool_or_auto(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "auto").strip().lower()
    if text == "auto":
        return False
    return text in {"1", "true", "yes", "on"}


def _filter_generation_params(params_data: dict[str, Any], generation_params_cls: Any) -> dict[str, Any]:
    fields = getattr(generation_params_cls, "__dataclass_fields__", None) or {}
    if not fields:
        return dict(params_data)
    allowed = set(fields.keys())
    dropped = sorted(key for key in params_data if key not in allowed)
    if dropped:
        print(f"[official_runner] dropping unsupported GenerationParams fields: {', '.join(dropped)}")
    return {key: value for key, value in params_data.items() if key in allowed}


def _call_compat(method: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(method)
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
        allowed = set(signature.parameters)
        if not accepts_kwargs:
            dropped = sorted(key for key in kwargs if key not in allowed)
            if dropped:
                print(f"[official_runner] dropping unsupported {getattr(method, '__name__', 'method')} kwargs: {', '.join(dropped)}")
            kwargs = {key: value for key, value in kwargs.items() if key in allowed}
    except (TypeError, ValueError):
        pass
    return method(**kwargs)


def _run(request_path: Path, response_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    vendor_dir = Path(request["vendor_dir"]).resolve()
    _prepare_vendor_imports(vendor_dir)

    from acestep.handler import AceStepHandler
    from acestep.inference import (
        GenerationConfig,
        GenerationParams,
        create_sample,
        format_sample,
        generate_music,
        understand_music,
    )
    from acestep.llm_inference import LLMHandler

    _is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    if not _is_apple_silicon:
        _disable_acestep_mlx_backends(AceStepHandler)
    else:
        # MLX Metal streams are thread-local. The diffusion timeout mechanism runs
        # generation in a child thread, but MLX is initialized in the main thread.
        # Monkey-patch the generation thread to set up the MLX default device.
        _patch_mlx_thread_stream(AceStepHandler)

    save_dir = Path(request.get("save_dir") or response_path.parent)
    save_dir.mkdir(parents=True, exist_ok=True)
    action = str(request.get("action") or "generate")

    def _init_llm() -> Any:
        llm = LLMHandler(persistent_storage_path=request["model_cache_dir"])
        lm_status, lm_ready = llm.initialize(
            checkpoint_dir=request["checkpoint_dir"],
            lm_model_path=request["lm_model"],
            backend=_resolve_backend(request.get("lm_backend", "auto")),
            device=str(request.get("lm_device") or request.get("device") or "auto"),
            offload_to_cpu=_bool_or_auto(request.get("lm_offload_to_cpu", False)),
            dtype=_none_if_auto(request.get("lm_dtype") or request.get("dtype")),
        )
        if not lm_ready:
            raise RuntimeError(lm_status)
        return llm

    if action == "create_sample":
        llm_handler = _init_llm()
        params_data = dict(request.get("params") or {})
        sample = create_sample(
            llm_handler=llm_handler,
            query=request.get("query") or params_data.get("sample_query") or params_data.get("caption") or "NO USER INPUT",
            instrumental=bool(params_data.get("instrumental")),
            vocal_language=params_data.get("vocal_language") or "unknown",
            temperature=float(params_data.get("lm_temperature", 0.85)),
            top_k=int(params_data.get("lm_top_k") or 0),
            top_p=float(params_data.get("lm_top_p") or 0.9),
            repetition_penalty=float(params_data.get("repetition_penalty") or 1.0),
            use_constrained_decoding=bool(params_data.get("use_constrained_decoding", True)),
            constrained_decoding_debug=bool(params_data.get("constrained_decoding_debug")),
        )
        response_path.write_text(json.dumps(_jsonable(sample.to_dict()), indent=2), encoding="utf-8")
        return

    if action == "format_sample":
        llm_handler = _init_llm()
        params_data = dict(request.get("params") or {})
        formatted = format_sample(
            llm_handler=llm_handler,
            caption=params_data.get("caption") or "",
            lyrics=params_data.get("lyrics") or "",
            user_metadata=params_data.get("user_metadata") or {},
            temperature=float(params_data.get("lm_temperature", 0.85)),
            top_k=int(params_data.get("lm_top_k") or 0),
            top_p=float(params_data.get("lm_top_p") or 0.9),
            repetition_penalty=float(params_data.get("repetition_penalty") or 1.0),
            use_constrained_decoding=bool(params_data.get("use_constrained_decoding", True)),
            constrained_decoding_debug=bool(params_data.get("constrained_decoding_debug")),
        )
        response_path.write_text(json.dumps(_jsonable(formatted.to_dict()), indent=2), encoding="utf-8")
        return

    if action == "understand_music":
        llm_handler = _init_llm()
        params_data = dict(request.get("params") or {})
        understood = understand_music(
            llm_handler=llm_handler,
            audio_codes=params_data.get("audio_codes") or "NO USER INPUT",
            temperature=float(params_data.get("lm_temperature", 0.85)),
            top_k=int(params_data.get("lm_top_k") or 0),
            top_p=float(params_data.get("lm_top_p") or 0.9),
            repetition_penalty=float(params_data.get("repetition_penalty") or 1.0),
            use_constrained_decoding=bool(params_data.get("use_constrained_decoding", True)),
            constrained_decoding_debug=bool(params_data.get("constrained_decoding_debug")),
        )
        response_path.write_text(json.dumps(_jsonable(understood.to_dict()), indent=2), encoding="utf-8")
        return

    dit_handler = AceStepHandler()
    flash_request = request.get("use_flash_attention", "auto")
    use_flash = dit_handler.is_flash_attention_available("auto") if str(flash_request).lower() == "auto" else _bool_or_auto(flash_request)
    status, ready = _call_compat(
        dit_handler.initialize_service,
        project_root=request["base_dir"],
        config_path=request["song_model"],
        device=str(request.get("device") or "auto"),
        use_flash_attention=use_flash,
        compile_model=_bool_or_auto(request.get("compile_model", False)),
        offload_to_cpu=_bool_or_auto(request.get("offload_to_cpu", False)),
        offload_dit_to_cpu=_bool_or_auto(request.get("offload_dit_to_cpu", False)),
        use_mlx_dit=_is_apple_silicon,
    )
    if not ready:
        raise RuntimeError(status)

    params_data = dict(request["params"])
    config_data = dict(request["config"])
    llm_handler = None
    if request.get("requires_lm"):
        llm_handler = LLMHandler(persistent_storage_path=request["model_cache_dir"])
        lm_status, lm_ready = llm_handler.initialize(
            checkpoint_dir=request["checkpoint_dir"],
            lm_model_path=request["lm_model"],
            backend=_resolve_backend(request.get("lm_backend", "auto")),
            device=str(request.get("lm_device") or request.get("device") or "auto"),
            offload_to_cpu=_bool_or_auto(request.get("lm_offload_to_cpu", False)),
            dtype=_none_if_auto(request.get("lm_dtype") or request.get("dtype")),
        )
        if not lm_ready:
            raise RuntimeError(lm_status)

    if params_data.pop("sample_mode", False) or str(params_data.pop("sample_query", "") or "").strip():
        if llm_handler is None:
            raise RuntimeError("sample_mode requires an initialized ACE-Step LM")
        sample = create_sample(
            llm_handler=llm_handler,
            query=request["params"].get("sample_query") or params_data.get("caption") or "NO USER INPUT",
            instrumental=bool(params_data.get("instrumental")),
            vocal_language=params_data.get("vocal_language") or "unknown",
            temperature=float(params_data.get("lm_temperature", 0.85)),
            top_k=int(params_data.get("lm_top_k") or 0),
            top_p=float(params_data.get("lm_top_p") or 0.9),
            repetition_penalty=float(params_data.get("repetition_penalty") or 1.0),
            use_constrained_decoding=bool(params_data.get("use_constrained_decoding", True)),
            constrained_decoding_debug=bool(params_data.get("constrained_decoding_debug")),
        )
        if not sample.success:
            raise RuntimeError(sample.error or sample.status_message or "create_sample failed")
        params_data["caption"] = sample.caption
        params_data["lyrics"] = sample.lyrics
        params_data["instrumental"] = bool(sample.instrumental)
        params_data["bpm"] = params_data.get("bpm") or sample.bpm
        params_data["keyscale"] = params_data.get("keyscale") or sample.keyscale
        params_data["timesignature"] = params_data.get("timesignature") or sample.timesignature
        params_data["duration"] = params_data.get("duration") if params_data.get("duration", -1) > 0 else sample.duration
        if params_data.get("vocal_language") in ("", "unknown", None):
            params_data["vocal_language"] = sample.language

    if params_data.pop("use_format", False):
        if llm_handler is None:
            raise RuntimeError("use_format requires an initialized ACE-Step LM")
        formatted = format_sample(
            llm_handler=llm_handler,
            caption=params_data.get("caption") or "",
            lyrics=params_data.get("lyrics") or "",
            user_metadata={
                key: value
                for key, value in {
                    "bpm": params_data.get("bpm"),
                    "keyscale": params_data.get("keyscale"),
                    "timesignature": params_data.get("timesignature"),
                    "duration": params_data.get("duration"),
                    "language": params_data.get("vocal_language"),
                }.items()
                if value not in (None, "", "unknown")
            },
            temperature=float(params_data.get("lm_temperature", 0.85)),
            top_k=int(params_data.get("lm_top_k") or 0),
            top_p=float(params_data.get("lm_top_p") or 0.9),
            repetition_penalty=float(params_data.get("repetition_penalty") or 1.0),
            use_constrained_decoding=bool(params_data.get("use_constrained_decoding", True)),
            constrained_decoding_debug=bool(params_data.get("constrained_decoding_debug")),
        )
        if not formatted.success:
            raise RuntimeError(formatted.error or formatted.status_message or "format_sample failed")
        params_data["caption"] = formatted.caption or params_data.get("caption", "")
        params_data["lyrics"] = formatted.lyrics or params_data.get("lyrics", "")
        params_data["bpm"] = formatted.bpm or params_data.get("bpm")
        params_data["keyscale"] = formatted.keyscale or params_data.get("keyscale", "")
        params_data["timesignature"] = formatted.timesignature or params_data.get("timesignature", "")
        params_data["duration"] = formatted.duration or params_data.get("duration")

    config = GenerationConfig(
        batch_size=int(config_data.get("batch_size") or 1),
        allow_lm_batch=bool(config_data.get("allow_lm_batch")),
        use_random_seed=bool(config_data.get("use_random_seed")),
        seeds=_parse_seeds(config_data.get("seeds")),
        lm_batch_chunk_size=int(config_data.get("lm_batch_chunk_size") or 8),
        constrained_decoding_debug=bool(config_data.get("constrained_decoding_debug")),
        audio_format=str(config_data.get("audio_format") or "flac"),
        mp3_bitrate=str(config_data.get("mp3_bitrate") or "128k"),
        mp3_sample_rate=int(config_data.get("mp3_sample_rate") or 48000),
    )
    params = GenerationParams(**_filter_generation_params(params_data, GenerationParams))
    result = generate_music(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        params=params,
        config=config,
        save_dir=str(save_dir),
    )
    result_data = result.to_dict()
    audios = []
    for audio in result_data.get("audios", []):
        audios.append(
            {
                "path": audio.get("path", ""),
                "key": audio.get("key", ""),
                "sample_rate": audio.get("sample_rate", 48000),
                "params": _jsonable(audio.get("params") or {}),
            }
        )
    response_path.write_text(
        json.dumps(
            {
                "success": bool(result_data.get("success")),
                "error": result_data.get("error"),
                "status_message": result_data.get("status_message", ""),
                "audios": audios,
                "time_costs": _jsonable((result_data.get("extra_outputs") or {}).get("time_costs", {})),
                "lm_metadata": _jsonable((result_data.get("extra_outputs") or {}).get("lm_metadata")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    try:
        _run(request_path, response_path)
        return 0
    except Exception as exc:
        response_path.write_text(
            json.dumps({"success": False, "error": str(exc), "traceback": traceback.format_exc()}, indent=2),
            encoding="utf-8",
        )
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
