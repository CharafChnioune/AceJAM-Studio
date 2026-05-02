from __future__ import annotations

import json
import os
import inspect
import platform
import sys
import traceback
from pathlib import Path
from typing import Any


ACE_STEP_CAPTION_CHAR_LIMIT = 512
ACE_STEP_LYRICS_CHAR_LIMIT = 4096


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


def _clip_caption_for_acestep(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) <= ACE_STEP_CAPTION_CHAR_LIMIT:
        return text
    clipped = text[:ACE_STEP_CAPTION_CHAR_LIMIT].rsplit(",", 1)[0].strip(" ,")
    return clipped or text[:ACE_STEP_CAPTION_CHAR_LIMIT].strip()


def _enforce_text_budgets(params_data: dict[str, Any], *, stage: str) -> None:
    caption = str(params_data.get("caption") or "")
    if len(caption) > ACE_STEP_CAPTION_CHAR_LIMIT:
        print(
            f"[official_runner] {stage} caption over {ACE_STEP_CAPTION_CHAR_LIMIT} chars; clipping "
            f"{len(caption)}->{ACE_STEP_CAPTION_CHAR_LIMIT}"
        )
        params_data["caption"] = _clip_caption_for_acestep(caption)
    lyrics = str(params_data.get("lyrics") or "")
    if lyrics.strip().lower() == "[instrumental]":
        return
    if len(lyrics) > ACE_STEP_LYRICS_CHAR_LIMIT:
        raise RuntimeError(
            f"{stage} lyrics exceed ACE-Step budget: {len(lyrics)}/{ACE_STEP_LYRICS_CHAR_LIMIT} chars. "
            "Shorten, split into parts, or run the app auto-fit step before rendering."
        )


def _resolve_backend(requested: str) -> str:
    value = (requested or "auto").strip().lower()
    apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    if value == "auto":
        return "mlx" if apple_silicon else "pt"
    if value == "mlx":
        return "mlx" if apple_silicon else "pt"
    if value in {"pt", "vllm"}:
        return value
    return "mlx" if apple_silicon else "pt"


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
    """Bypass the diffusion worker thread when MLX is the DiT backend.

    MLX Metal streams are thread-local. ACE-Step's default wrapper runs
    service_generate in a child thread so it can enforce a timeout; that trips
    MLX with "There is no Stream(gpu, 0) in current thread". For active MLX DiT
    runs, execute service_generate directly on the caller thread and block the
    vendor PyTorch fallback that would otherwise OOM on XL models.
    """
    try:
        from acestep.core.generation.handler.generate_music_execute import (
            GenerateMusicExecuteMixin,
        )
    except ImportError:
        return

    from loguru import logger as _logger

    current = GenerateMusicExecuteMixin._run_generate_music_service_with_progress
    if getattr(current, "_acejam_mlx_direct", False):
        return

    _orig = GenerateMusicExecuteMixin._run_generate_music_service_with_progress

    def _patched_run_with_progress(self, *args, **kwargs):
        use_mlx = getattr(self, "use_mlx_dit", False) and getattr(self, "mlx_decoder", None) is not None
        if not use_mlx:
            return _orig(self, *args, **kwargs)

        bound = inspect.signature(_orig).bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        progress = values["progress"]
        actual_batch_size = values["actual_batch_size"]
        audio_duration = values["audio_duration"]
        inference_steps = values["inference_steps"]
        timesteps = values["timesteps"]
        service_inputs = values["service_inputs"]
        refer_audios = values["refer_audios"]
        guidance_scale = values["guidance_scale"]
        actual_seed_list = values["actual_seed_list"]
        audio_cover_strength = values["audio_cover_strength"]
        cover_noise_strength = values["cover_noise_strength"]
        use_adg = values["use_adg"]
        cfg_interval_start = values["cfg_interval_start"]
        cfg_interval_end = values["cfg_interval_end"]
        shift = values["shift"]
        infer_method = values["infer_method"]
        sampler_mode = values["sampler_mode"]
        velocity_norm_threshold = values["velocity_norm_threshold"]
        velocity_ema_factor = values["velocity_ema_factor"]
        repaint_crossfade_frames = values["repaint_crossfade_frames"]
        repaint_injection_ratio = values["repaint_injection_ratio"]
        dcw_enabled = values["dcw_enabled"]
        dcw_mode = values["dcw_mode"]
        dcw_scaler = values["dcw_scaler"]
        dcw_high_scaler = values["dcw_high_scaler"]
        dcw_wavelet = values["dcw_wavelet"]
        task_type = values["task_type"]

        infer_steps_for_progress = len(timesteps) if timesteps else inference_steps
        progress_desc = f"Generating music (batch size: {actual_batch_size})..."
        if callable(progress):
            progress(0.52, desc=progress_desc)

        stop_event = None
        progress_thread = None
        original_generate_audio = getattr(getattr(self, "model", None), "generate_audio", None)

        def _blocked_pytorch_fallback(*_a: Any, **_kw: Any) -> Any:
            raise RuntimeError(
                "MLX diffusion failed and AceJAM blocked the PyTorch/MPS fallback to avoid "
                "XL model OOM. Check the previous '[service_generate] MLX diffusion failed' "
                "line for the native MLX error."
            )

        try:
            start_estimator = getattr(self, "_start_diffusion_progress_estimator", None)
            if callable(start_estimator):
                stop_event, progress_thread = start_estimator(
                    progress=progress,
                    start=0.52,
                    end=0.79,
                    infer_steps=infer_steps_for_progress,
                    batch_size=actual_batch_size,
                    duration_sec=audio_duration if audio_duration and audio_duration > 0 else None,
                    desc=progress_desc,
                )
            if original_generate_audio is not None:
                setattr(self.model, "generate_audio", _blocked_pytorch_fallback)

            _logger.info(
                "[generate_music] MLX active - running diffusion directly on the main thread "
                "(task_type={}, dcw_enabled={}).",
                task_type,
                dcw_enabled,
            )
            outputs = self.service_generate(
                captions=service_inputs["captions_batch"],
                global_captions=service_inputs.get("global_captions_batch"),
                lyrics=service_inputs["lyrics_batch"],
                metas=service_inputs["metas_batch"],
                vocal_languages=service_inputs["vocal_languages_batch"],
                refer_audios=refer_audios,
                target_wavs=service_inputs["target_wavs_tensor"],
                infer_steps=inference_steps,
                guidance_scale=guidance_scale,
                seed=actual_seed_list,
                repainting_start=service_inputs["repainting_start_batch"],
                repainting_end=service_inputs["repainting_end_batch"],
                instructions=service_inputs["instructions_batch"],
                audio_cover_strength=audio_cover_strength,
                cover_noise_strength=cover_noise_strength,
                use_adg=use_adg,
                cfg_interval_start=cfg_interval_start,
                cfg_interval_end=cfg_interval_end,
                shift=shift,
                infer_method=infer_method,
                sampler_mode=sampler_mode,
                velocity_norm_threshold=velocity_norm_threshold,
                velocity_ema_factor=velocity_ema_factor,
                dcw_enabled=dcw_enabled,
                dcw_mode=dcw_mode,
                dcw_scaler=dcw_scaler,
                dcw_high_scaler=dcw_high_scaler,
                dcw_wavelet=dcw_wavelet,
                audio_code_hints=service_inputs["audio_code_hints_batch"],
                return_intermediate=service_inputs["should_return_intermediate"],
                timesteps=timesteps,
                chunk_mask_modes=service_inputs.get("chunk_mask_modes_batch"),
                repaint_crossfade_frames=repaint_crossfade_frames,
                repaint_injection_ratio=repaint_injection_ratio,
                task_type=task_type,
            )
            return {"outputs": outputs, "infer_steps_for_progress": infer_steps_for_progress}
        finally:
            if original_generate_audio is not None:
                setattr(self.model, "generate_audio", original_generate_audio)
            if stop_event is not None:
                stop_event.set()
            if progress_thread is not None:
                progress_thread.join(timeout=1.0)

    _patched_run_with_progress._acejam_mlx_direct = True
    _patched_run_with_progress._acejam_original = _orig
    GenerateMusicExecuteMixin._run_generate_music_service_with_progress = _patched_run_with_progress


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


def _normalize_generation_params(params_data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params_data)
    if str(normalized.get("timesteps") or "").strip() == "":
        normalized["timesteps"] = None
    for key in ["repainting_start", "repainting_end"]:
        if str(normalized.get(key) or "").strip() == "":
            normalized[key] = None
    return normalized


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


def _lm_sampling(request: dict[str, Any], params_data: dict[str, Any]) -> dict[str, Any]:
    sampling = dict(request.get("lm_sampling") or {})
    return {
        "temperature": float(sampling.get("temperature", params_data.get("lm_temperature", 0.85))),
        "top_k": int(sampling.get("top_k", params_data.get("lm_top_k", 0)) or 0),
        "top_p": float(sampling.get("top_p", params_data.get("lm_top_p", 0.9))),
        "repetition_penalty": float(
            sampling.get(
                "repetition_penalty",
                params_data.get("repetition_penalty", params_data.get("lm_repetition_penalty", 1.0)),
            )
            or 1.0
        ),
        "use_constrained_decoding": bool(
            sampling.get("use_constrained_decoding", params_data.get("use_constrained_decoding", True))
        ),
        "constrained_decoding_debug": bool(
            sampling.get("constrained_decoding_debug", params_data.get("constrained_decoding_debug", False))
        ),
    }


def _apply_lora_request(dit_handler: Any, request: dict[str, Any]) -> dict[str, Any]:
    if not bool(request.get("use_lora")):
        return {"success": True, "active": False}
    adapter_path = str(request.get("lora_adapter_path") or "").strip()
    if not adapter_path:
        raise RuntimeError("Official runner received use_lora=true without lora_adapter_path")
    scale = float(request.get("lora_scale") or 1.0)
    status = dit_handler.load_lora(adapter_path)
    if str(status).startswith("❌"):
        raise RuntimeError(status)
    scale_status = dit_handler.set_lora_scale(scale)
    use_status = dit_handler.set_use_lora(True)
    if str(use_status).startswith("❌"):
        raise RuntimeError(use_status)
    print(f"[official_runner] LoRA active: {adapter_path} scale={scale}")
    return {
        "success": True,
        "active": True,
        "path": adapter_path,
        "scale": scale,
        "status": status,
        "scale_status": scale_status,
        "use_status": use_status,
    }


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
        sampling = _lm_sampling(request, params_data)
        sample = create_sample(
            llm_handler=llm_handler,
            query=request.get("query") or params_data.get("sample_query") or params_data.get("caption") or "NO USER INPUT",
            instrumental=bool(params_data.get("instrumental")),
            vocal_language=params_data.get("vocal_language") or "unknown",
            **sampling,
        )
        response_path.write_text(json.dumps(_jsonable(sample.to_dict()), indent=2), encoding="utf-8")
        return

    if action == "format_sample":
        llm_handler = _init_llm()
        params_data = dict(request.get("params") or {})
        sampling = _lm_sampling(request, params_data)
        formatted = format_sample(
            llm_handler=llm_handler,
            caption=params_data.get("caption") or "",
            lyrics=params_data.get("lyrics") or "",
            user_metadata=params_data.get("user_metadata") or {},
            **sampling,
        )
        response_path.write_text(json.dumps(_jsonable(formatted.to_dict()), indent=2), encoding="utf-8")
        return

    if action == "understand_music":
        llm_handler = _init_llm()
        params_data = dict(request.get("params") or {})
        sampling = _lm_sampling(request, params_data)
        understood = understand_music(
            llm_handler=llm_handler,
            audio_codes=params_data.get("audio_codes") or "NO USER INPUT",
            **sampling,
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

    lora_status = _apply_lora_request(dit_handler, request)

    params_data = _normalize_generation_params(dict(request["params"]))
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
        sampling = _lm_sampling(request, params_data)
        sample = create_sample(
            llm_handler=llm_handler,
            query=request["params"].get("sample_query") or params_data.get("caption") or "NO USER INPUT",
            instrumental=bool(params_data.get("instrumental")),
            vocal_language=params_data.get("vocal_language") or "unknown",
            **sampling,
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
        original_caption = str(params_data.get("caption") or "")
        original_lyrics = str(params_data.get("lyrics") or "")
        sampling = _lm_sampling(request, params_data)
        formatted = format_sample(
            llm_handler=llm_handler,
            caption=original_caption,
            lyrics=original_lyrics,
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
            **sampling,
        )
        if not formatted.success:
            raise RuntimeError(formatted.error or formatted.status_message or "format_sample failed")
        formatted_caption = str(formatted.caption or original_caption)
        formatted_lyrics = str(formatted.lyrics or original_lyrics)
        if len(formatted_lyrics) > ACE_STEP_LYRICS_CHAR_LIMIT and 0 < len(original_lyrics) <= ACE_STEP_LYRICS_CHAR_LIMIT:
            print(
                "[official_runner] format_sample lyrics exceeded ACE-Step budget; "
                f"preserving original lyrics ({len(formatted_lyrics)}->{len(original_lyrics)} chars)"
            )
            formatted_lyrics = original_lyrics
        params_data["caption"] = _clip_caption_for_acestep(formatted_caption)
        params_data["lyrics"] = formatted_lyrics
        params_data["bpm"] = formatted.bpm or params_data.get("bpm")
        params_data["keyscale"] = formatted.keyscale or params_data.get("keyscale", "")
        params_data["timesignature"] = formatted.timesignature or params_data.get("timesignature", "")
        params_data["duration"] = formatted.duration or params_data.get("duration")
        _enforce_text_budgets(params_data, stage="format_sample output")

    _enforce_text_budgets(params_data, stage="generation input")

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
                "lora_status": _jsonable(lora_status),
                "official_api_fields": _jsonable(request.get("official_api_fields") or {}),
                "guarded_api_fields": _jsonable(request.get("guarded_api_fields") or {}),
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
