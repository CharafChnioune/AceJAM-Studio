from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


OLLAMA_DEFAULT_HOST = "http://localhost:11434"
LMSTUDIO_DEFAULT_HOST = "http://localhost:1234"
PLANNER_LLM_DEFAULTS: dict[str, Any] = {
    "planner_creativity_preset": "balanced",
    "planner_temperature": 0.45,
    "planner_top_p": 0.92,
    "planner_top_k": 40,
    "planner_repeat_penalty": 1.1,
    "planner_seed": "",
    "planner_max_tokens": 2048,
    "planner_context_length": 8192,
    "planner_timeout": 600.0,
}
PLANNER_LLM_PRESETS: dict[str, dict[str, Any]] = {
    "stable": {
        "planner_temperature": 0.2,
        "planner_top_p": 0.85,
        "planner_top_k": 20,
        "planner_repeat_penalty": 1.15,
    },
    "balanced": {
        "planner_temperature": 0.45,
        "planner_top_p": 0.92,
        "planner_top_k": 40,
        "planner_repeat_penalty": 1.1,
    },
    "creative": {
        "planner_temperature": 0.8,
        "planner_top_p": 0.95,
        "planner_top_k": 70,
        "planner_repeat_penalty": 1.05,
    },
    "wild": {
        "planner_temperature": 1.1,
        "planner_top_p": 0.98,
        "planner_top_k": 100,
        "planner_repeat_penalty": 1.0,
    },
}
ACEJAM_PRINT_LLM_IO = os.environ.get(
    "ACEJAM_PRINT_LLM_IO",
    os.environ.get("ACEJAM_PRINT_AGENT_IO", "1"),
).strip().lower() in {"1", "true", "yes", "on"}
ACEJAM_PRINT_LLM_IO_MAX_CHARS = max(0, int(os.environ.get("ACEJAM_PRINT_LLM_IO_MAX_CHARS", "0") or 0))


class LocalLLMError(RuntimeError):
    pass


def _llm_debug_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _llm_debug_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_llm_debug_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _llm_debug_jsonable(value.model_dump())
        except Exception:
            return str(value)
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        try:
            return _llm_debug_jsonable(vars(value))
        except Exception:
            return str(value)
    return value


def _print_llm_io(label: str, payload: Any) -> None:
    if not ACEJAM_PRINT_LLM_IO:
        return
    text = payload if isinstance(payload, str) else json.dumps(_llm_debug_jsonable(payload), ensure_ascii=False, indent=2)
    original_len = len(text)
    if ACEJAM_PRINT_LLM_IO_MAX_CHARS and original_len > ACEJAM_PRINT_LLM_IO_MAX_CHARS:
        text = (
            text[:ACEJAM_PRINT_LLM_IO_MAX_CHARS].rstrip()
            + f"\n[truncated by ACEJAM_PRINT_LLM_IO_MAX_CHARS; original_chars={original_len}]"
        )
    print(f"[acejam_llm_io][BEGIN {label} chars={original_len}]", flush=True)
    print(text, flush=True)
    print(f"[acejam_llm_io][END {label}]", flush=True)


def normalize_provider(provider: Any) -> str:
    value = str(provider or "ollama").strip().lower().replace("_", "-")
    if value in {"lmstudio", "lm-studio", "lm studio"}:
        return "lmstudio"
    return "ollama"


def provider_label(provider: Any) -> str:
    return "LM Studio" if normalize_provider(provider) == "lmstudio" else "Ollama"


def _payload_first(payload: dict[str, Any] | None, *names: str) -> Any:
    source = payload if isinstance(payload, dict) else {}
    for name in names:
        if name in source and source.get(name) not in [None, ""]:
            return source.get(name)
    return None


def _clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    return max(float(minimum), min(float(maximum), number))


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = int(default)
    return max(int(minimum), min(int(maximum), number))


def planner_llm_settings_from_payload(
    payload: dict[str, Any] | None,
    *,
    default_max_tokens: int | None = None,
    default_timeout: float | None = None,
) -> dict[str, Any]:
    """Normalize Ollama/LM Studio planner controls, separate from ACE-Step LM controls."""
    source = payload if isinstance(payload, dict) else {}
    preset = str(
        _payload_first(source, "planner_creativity_preset", "local_llm_creativity_preset")
        or os.environ.get("ACEJAM_PLANNER_CREATIVITY_PRESET")
        or PLANNER_LLM_DEFAULTS["planner_creativity_preset"]
    ).strip().lower()
    if preset not in PLANNER_LLM_PRESETS:
        preset = "balanced"
    base = {**PLANNER_LLM_DEFAULTS, **PLANNER_LLM_PRESETS[preset]}
    if default_max_tokens is not None:
        base["planner_max_tokens"] = int(default_max_tokens)
    if default_timeout is not None:
        base["planner_timeout"] = float(default_timeout)

    temperature = _clamp_float(
        _payload_first(source, "planner_temperature", "local_llm_temperature")
        or os.environ.get("ACEJAM_PLANNER_TEMPERATURE")
        or base["planner_temperature"],
        float(base["planner_temperature"]),
        0.0,
        2.0,
    )
    top_p = _clamp_float(
        _payload_first(source, "planner_top_p", "local_llm_top_p")
        or os.environ.get("ACEJAM_PLANNER_TOP_P")
        or base["planner_top_p"],
        float(base["planner_top_p"]),
        0.0,
        1.0,
    )
    top_k = _clamp_int(
        _payload_first(source, "planner_top_k", "local_llm_top_k")
        or os.environ.get("ACEJAM_PLANNER_TOP_K")
        or base["planner_top_k"],
        int(base["planner_top_k"]),
        0,
        200,
    )
    repeat_penalty = _clamp_float(
        _payload_first(source, "planner_repeat_penalty", "local_llm_repeat_penalty")
        or os.environ.get("ACEJAM_PLANNER_REPEAT_PENALTY")
        or base["planner_repeat_penalty"],
        float(base["planner_repeat_penalty"]),
        0.8,
        2.0,
    )
    max_tokens = _clamp_int(
        _payload_first(source, "planner_max_tokens", "local_llm_max_tokens")
        or os.environ.get("ACEJAM_PLANNER_MAX_TOKENS")
        or base["planner_max_tokens"],
        int(base["planner_max_tokens"]),
        128,
        8192,
    )
    context_length = _clamp_int(
        _payload_first(source, "planner_context_length", "local_llm_context_length", "planner_num_ctx")
        or os.environ.get("ACEJAM_PLANNER_CONTEXT_LENGTH")
        or base["planner_context_length"],
        int(base["planner_context_length"]),
        2048,
        32768,
    )
    timeout = _clamp_float(
        _payload_first(source, "planner_timeout", "local_llm_timeout")
        or os.environ.get("ACEJAM_PLANNER_TIMEOUT")
        or base["planner_timeout"],
        float(base["planner_timeout"]),
        30.0,
        1800.0,
    )
    seed_raw = str(
        _payload_first(source, "planner_seed", "local_llm_seed")
        or os.environ.get("ACEJAM_PLANNER_SEED")
        or ""
    ).strip()
    planner_seed: str | int = ""
    if seed_raw and seed_raw.lower() != "random" and seed_raw != "-1":
        try:
            planner_seed = max(0, min(2**31 - 1, int(float(seed_raw))))
        except (TypeError, ValueError):
            planner_seed = ""
    return {
        "planner_creativity_preset": preset,
        "planner_temperature": temperature,
        "planner_top_p": top_p,
        "planner_top_k": top_k,
        "planner_repeat_penalty": repeat_penalty,
        "planner_seed": planner_seed,
        "planner_max_tokens": max_tokens,
        "planner_context_length": context_length,
        "planner_timeout": timeout,
    }


def planner_llm_options_for_provider(
    provider: Any,
    payload: dict[str, Any] | None,
    *,
    default_max_tokens: int | None = None,
    default_timeout: float | None = None,
) -> dict[str, Any]:
    settings = planner_llm_settings_from_payload(
        payload,
        default_max_tokens=default_max_tokens,
        default_timeout=default_timeout,
    )
    options: dict[str, Any] = {
        "temperature": settings["planner_temperature"],
        "top_p": settings["planner_top_p"],
        "top_k": settings["planner_top_k"],
        "repeat_penalty": settings["planner_repeat_penalty"],
        "timeout": settings["planner_timeout"],
    }
    if settings["planner_seed"] != "":
        options["seed"] = settings["planner_seed"]
    if normalize_provider(provider) == "ollama":
        options["num_ctx"] = settings["planner_context_length"]
        options["num_predict"] = settings["planner_max_tokens"]
    else:
        options["max_tokens"] = settings["planner_max_tokens"]
    return options


def ollama_host() -> str:
    return os.environ.get("OLLAMA_BASE_URL", OLLAMA_DEFAULT_HOST).strip() or OLLAMA_DEFAULT_HOST


def lmstudio_base_url() -> str:
    return os.environ.get("LMSTUDIO_BASE_URL", LMSTUDIO_DEFAULT_HOST).strip().rstrip("/") or LMSTUDIO_DEFAULT_HOST


def lmstudio_api_base_url() -> str:
    base = lmstudio_base_url()
    return base if base.rstrip("/").endswith("/v1") else f"{base}/v1"


def lmstudio_native_base_url() -> str:
    base = lmstudio_base_url()
    if base.rstrip("/").endswith("/v1"):
        base = base.rsplit("/v1", 1)[0]
    return f"{base}/api/v1"


def _lmstudio_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("LMSTUDIO_API_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=_lmstudio_headers(), method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise LocalLLMError(f"{method.upper()} {url} failed with HTTP {exc.code}: {body[:500]}") from exc
    except Exception as exc:
        raise LocalLLMError(f"{method.upper()} {url} failed: {exc}") from exc
    if not body.strip():
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


def _attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _is_embedding_name(name: str) -> bool:
    return bool(re.search(r"(embed|embedding|bge|e5|gte|nomic|jina|snowflake|mxbai|arctic)", name or "", re.I))


def _ollama_client():
    import ollama

    return ollama.Client(host=ollama_host())


def ollama_model_catalog() -> dict[str, Any]:
    host = ollama_host()
    try:
        client = _ollama_client()
        response = client.list()
        raw_models = list(_attr(response, "models", []) or [])
        details: list[dict[str, Any]] = []
        for item in raw_models:
            name = str(_attr(item, "model", _attr(item, "name", "")) or "").strip()
            if not name:
                continue
            model_details = _attr(item, "details", {}) or {}
            size = int(_attr(item, "size", 0) or 0)
            kind = "embedding" if _is_embedding_name(name) else "chat"
            details.append(
                {
                    "name": name,
                    "model": name,
                    "provider": "ollama",
                    "kind": kind,
                    "type": "embedding" if kind == "embedding" else "llm",
                    "size": size,
                    "size_gb": round(size / 1e9, 2) if size else 0,
                    "modified_at": str(_attr(item, "modified_at", "") or ""),
                    "digest": str(_attr(item, "digest", "") or ""),
                    "family": str(_attr(model_details, "family", "") or (model_details.get("family", "") if isinstance(model_details, dict) else "")),
                    "parameter_size": str(_attr(model_details, "parameter_size", "") or (model_details.get("parameter_size", "") if isinstance(model_details, dict) else "")),
                    "quantization_level": str(_attr(model_details, "quantization_level", "") or (model_details.get("quantization_level", "") if isinstance(model_details, dict) else "")),
                    "loaded": False,
                }
            )
        running_models: list[str] = []
        if hasattr(client, "ps"):
            try:
                running = list(_attr(client.ps(), "models", []) or [])
                running_models = [str(_attr(item, "model", _attr(item, "name", "")) or "").strip() for item in running]
            except Exception:
                running_models = []
        running_set = set(running_models)
        for item in details:
            item["loaded"] = item["name"] in running_set
        models = [item["name"] for item in details]
        embedding_models = [item["name"] for item in details if item["kind"] == "embedding"]
        chat_models = [item["name"] for item in details if item["kind"] != "embedding"]
        return {
            "success": True,
            "ready": True,
            "provider": "ollama",
            "provider_label": "Ollama",
            "host": host,
            "ollama_host": host,
            "models": models,
            "chat_models": chat_models,
            "embedding_models": embedding_models,
            "details": details,
            "loaded_models": running_models,
            "running_models": running_models,
            "error": "",
        }
    except Exception as exc:
        return {
            "success": False,
            "ready": False,
            "provider": "ollama",
            "provider_label": "Ollama",
            "host": host,
            "ollama_host": host,
            "models": [],
            "chat_models": [],
            "embedding_models": [],
            "details": [],
            "loaded_models": [],
            "running_models": [],
            "error": f"Ollama is not reachable at {host}: {exc}",
        }


def _lmstudio_model_name(item: dict[str, Any]) -> str:
    for key in ("key", "id", "model", "name", "path"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return ""


def _lmstudio_model_kind(item: dict[str, Any], name: str) -> str:
    value = str(item.get("type") or item.get("kind") or item.get("model_type") or "").strip().lower()
    if value in {"embedding", "embeddings", "embed"}:
        return "embedding"
    if _is_embedding_name(name):
        return "embedding"
    return "chat"


def _lmstudio_loaded_names(item: dict[str, Any]) -> list[str]:
    loaded = item.get("loaded_instances") or item.get("loadedInstances") or item.get("instances") or []
    names: list[str] = []
    if isinstance(loaded, list):
        for instance in loaded:
            if isinstance(instance, dict):
                names.append(str(instance.get("identifier") or instance.get("id") or instance.get("name") or "").strip())
            else:
                names.append(str(instance or "").strip())
    elif isinstance(loaded, dict):
        names.append(str(loaded.get("identifier") or loaded.get("id") or loaded.get("name") or "").strip())
    return [name for name in names if name]


def _lmstudio_loaded_context_length(item: dict[str, Any]) -> int:
    loaded = item.get("loaded_instances") or item.get("loadedInstances") or item.get("instances") or []
    contexts: list[int] = []
    if isinstance(loaded, list):
        for instance in loaded:
            if not isinstance(instance, dict):
                continue
            config = instance.get("config") if isinstance(instance.get("config"), dict) else {}
            value = instance.get("context_length") or instance.get("contextLength") or config.get("context_length") or config.get("contextLength")
            try:
                if value:
                    contexts.append(int(value))
            except (TypeError, ValueError):
                continue
    return max(contexts) if contexts else 0


def lmstudio_model_catalog() -> dict[str, Any]:
    native = lmstudio_native_base_url()
    try:
        response = _http_json("GET", f"{native}/models", timeout=15)
        raw_models = response.get("models", response) if isinstance(response, dict) else response
        if not isinstance(raw_models, list):
            raw_models = []
        details: list[dict[str, Any]] = []
        loaded_models: list[str] = []
        for raw in raw_models:
            if not isinstance(raw, dict):
                continue
            name = _lmstudio_model_name(raw)
            if not name:
                continue
            kind = _lmstudio_model_kind(raw, name)
            loaded_instances = _lmstudio_loaded_names(raw)
            loaded_context_length = _lmstudio_loaded_context_length(raw)
            loaded = bool(raw.get("loaded") or raw.get("is_loaded") or loaded_instances)
            if loaded and not loaded_instances:
                loaded_instances = [name]
            loaded_models.extend(loaded_instances)
            quant = raw.get("quantization") or raw.get("quantization_level") or raw.get("quantizationLevel") or ""
            fmt = raw.get("format") or raw.get("format_type") or raw.get("architecture") or ""
            details.append(
                {
                    "name": name,
                    "model": name,
                    "provider": "lmstudio",
                    "kind": kind,
                    "type": "embedding" if kind == "embedding" else "llm",
                    "path": str(raw.get("path") or ""),
                    "display_name": str(raw.get("display_name") or raw.get("displayName") or raw.get("name") or name),
                    "format": str(fmt or ""),
                    "quantization_level": str(quant or ""),
                    "architecture": str(raw.get("architecture") or raw.get("arch") or ""),
                    "context_length": raw.get("context_length") or raw.get("contextLength") or raw.get("max_context_length") or "",
                    "loaded_context_length": loaded_context_length,
                    "reasoning": bool(raw.get("reasoning") or raw.get("is_reasoning_model") or False),
                    "loaded": loaded,
                    "loaded_instances": loaded_instances,
                    "raw": raw,
                }
            )
        models = [item["name"] for item in details]
        return {
            "success": True,
            "ready": True,
            "provider": "lmstudio",
            "provider_label": "LM Studio",
            "host": lmstudio_base_url(),
            "lmstudio_host": lmstudio_base_url(),
            "api_base": lmstudio_api_base_url(),
            "native_base": native,
            "models": models,
            "chat_models": [item["name"] for item in details if item["kind"] != "embedding"],
            "embedding_models": [item["name"] for item in details if item["kind"] == "embedding"],
            "details": details,
            "loaded_models": sorted(set(loaded_models)),
            "running_models": sorted(set(loaded_models)),
            "error": "",
        }
    except Exception as exc:
        return {
            "success": False,
            "ready": False,
            "provider": "lmstudio",
            "provider_label": "LM Studio",
            "host": lmstudio_base_url(),
            "lmstudio_host": lmstudio_base_url(),
            "api_base": lmstudio_api_base_url(),
            "native_base": native,
            "models": [],
            "chat_models": [],
            "embedding_models": [],
            "details": [],
            "loaded_models": [],
            "running_models": [],
            "error": f"LM Studio is not reachable at {lmstudio_base_url()}: {exc}",
        }


def model_catalog(provider: Any) -> dict[str, Any]:
    return lmstudio_model_catalog() if normalize_provider(provider) == "lmstudio" else ollama_model_catalog()


def resolve_model(provider: Any, model_name: str, kind: str = "chat") -> str:
    provider_name = normalize_provider(provider)
    model = str(model_name or "").strip()
    catalog = model_catalog(provider_name)
    if not catalog.get("ready"):
        raise LocalLLMError(catalog.get("error") or f"{provider_label(provider_name)} is not reachable.")
    key = "embedding_models" if kind == "embedding" else "chat_models"
    models = [str(item) for item in catalog.get(key) or [] if str(item).strip()]
    if model:
        if model not in set(catalog.get("models") or []):
            raise LocalLLMError(f"{model} is not available in {provider_label(provider_name)}.")
        return model
    if not models:
        raise LocalLLMError(f"No local {provider_label(provider_name)} {kind} model is available.")
    return models[0]


def lmstudio_load_model(model_name: str, kind: str = "chat", context_length: int | None = None) -> dict[str, Any]:
    model = str(model_name or "").strip()
    if not model:
        raise LocalLLMError("LM Studio model name is required.")
    payload: dict[str, Any] = {"model": model}
    if context_length:
        payload["context_length"] = int(context_length)
        payload["echo_load_config"] = True
    data = _http_json("POST", f"{lmstudio_native_base_url()}/models/load", payload, timeout=120)
    return {"success": True, "provider": "lmstudio", "model": model, "kind": kind, "response": data}


def _lmstudio_openai_chat_options(options: dict[str, Any] | None) -> dict[str, Any]:
    allowed = {
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "metadata",
        "n",
        "presence_penalty",
        "response_format",
        "seed",
        "stop",
        "stream_options",
        "temperature",
        "tool_choice",
        "tools",
        "top_k",
        "top_logprobs",
        "top_p",
        "repeat_penalty",
        "user",
    }
    return {key: value for key, value in (options or {}).items() if key in allowed and value is not None}


def lmstudio_unload_model(model_name: str) -> dict[str, Any]:
    model = str(model_name or "").strip()
    if not model:
        raise LocalLLMError("LM Studio model name is required.")
    payload = {"model": model, "model_key": model}
    try:
        data = _http_json("POST", f"{lmstudio_native_base_url()}/models/unload", payload, timeout=60)
    except LocalLLMError:
        data = _http_json("POST", f"{lmstudio_native_base_url()}/models/{urllib.parse.quote(model, safe='')}/unload", payload, timeout=60)
    return {"success": True, "provider": "lmstudio", "model": model, "response": data}


def lmstudio_download_model(model_name: str, quantization: str = "", kind: str = "chat") -> dict[str, Any]:
    model = str(model_name or "").strip()
    if not model:
        raise LocalLLMError("LM Studio model name or Hugging Face URL is required.")
    payload: dict[str, Any] = {"model": model, "model_key": model, "source": model, "kind": kind}
    if quantization:
        payload["quantization"] = quantization
    try:
        data = _http_json("POST", f"{lmstudio_native_base_url()}/models/download", payload, timeout=60)
    except LocalLLMError:
        data = _http_json("POST", f"{lmstudio_native_base_url()}/download", payload, timeout=60)
    job_id = ""
    if isinstance(data, dict):
        job_id = str(data.get("job_id") or data.get("jobId") or data.get("id") or data.get("download_id") or "")
    return {"success": True, "provider": "lmstudio", "model": model, "kind": kind, "job_id": job_id, "job": data}


def lmstudio_download_status(job_id: str) -> dict[str, Any]:
    job = str(job_id or "").strip()
    if not job:
        raise LocalLLMError("LM Studio download job id is required.")
    quoted = urllib.parse.quote(job, safe="")
    for path in (f"/models/download/{quoted}", f"/download/{quoted}"):
        try:
            data = _http_json("GET", f"{lmstudio_native_base_url()}{path}", timeout=20)
            return {"success": True, "provider": "lmstudio", "job_id": job, "job": data}
        except LocalLLMError:
            continue
    raise LocalLLMError(f"LM Studio download job {job} was not found.")


def ollama_chat(model_name: str, messages: list[dict[str, str]], *, options: dict[str, Any] | None = None, json_format: bool = False) -> str:
    request_timeout = float(os.environ.get("ACEJAM_OLLAMA_CHAT_TIMEOUT", "600"))
    kwargs: dict[str, Any] = {"model": model_name, "messages": messages, "think": False, "stream": False}
    if options:
        ollama_options = dict(options)
        request_timeout = float(ollama_options.pop("timeout", request_timeout) or request_timeout)
        if "max_tokens" in ollama_options and "num_predict" not in ollama_options:
            ollama_options["num_predict"] = ollama_options.pop("max_tokens")
        kwargs["options"] = ollama_options
    if json_format:
        kwargs["format"] = "json"
    _print_llm_io("ollama_chat_request_json", {"url": f"{ollama_host().rstrip('/')}/api/chat", "payload": kwargs})
    data = _http_json("POST", f"{ollama_host().rstrip('/')}/api/chat", kwargs, timeout=request_timeout)
    _print_llm_io("ollama_chat_response_json", data)
    content = str(_attr(_attr(data, "message", {}), "content", "") or "")
    _print_llm_io("ollama_chat_content", content)
    return content


def lmstudio_chat(model_name: str, messages: list[dict[str, str]], *, options: dict[str, Any] | None = None, json_format: bool = False) -> str:
    request_timeout = float((options or {}).get("timeout") or 600)
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": False,
    }
    payload.update(_lmstudio_openai_chat_options(options))
    if json_format:
        payload.setdefault("response_format", {"type": "json_object"})
    _print_llm_io("lmstudio_chat_request_json", {"url": f"{lmstudio_api_base_url()}/chat/completions", "payload": payload})
    data = _http_json("POST", f"{lmstudio_api_base_url()}/chat/completions", payload, timeout=request_timeout)
    _print_llm_io("lmstudio_chat_response_json", data)
    choices = data.get("choices") if isinstance(data, dict) else None
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = str(message.get("content") or "")
            _print_llm_io("lmstudio_chat_content", content)
            return content
        content = str(choices[0].get("text") or "")
        _print_llm_io("lmstudio_chat_content", content)
        return content
    content = str(data.get("raw") if isinstance(data, dict) else data or "")
    _print_llm_io("lmstudio_chat_content", content)
    return content


def chat_completion(provider: Any, model_name: str, messages: list[dict[str, str]], *, options: dict[str, Any] | None = None, json_format: bool = False) -> str:
    provider_name = normalize_provider(provider)
    model = resolve_model(provider_name, model_name, "chat")
    if provider_name == "lmstudio":
        return lmstudio_chat(model, messages, options=options, json_format=json_format)
    return ollama_chat(model, messages, options=options, json_format=json_format)


def embed(provider: Any, model_name: str, text: str) -> list[float]:
    provider_name = normalize_provider(provider)
    model = resolve_model(provider_name, model_name, "embedding")
    if provider_name == "lmstudio":
        payload = {"model": model, "input": text or "AceJAM"}
        _print_llm_io("lmstudio_embedding_request_json", {"url": f"{lmstudio_api_base_url()}/embeddings", "payload": payload})
        data = _http_json(
            "POST",
            f"{lmstudio_api_base_url()}/embeddings",
            payload,
            timeout=120,
        )
        _print_llm_io("lmstudio_embedding_response_json", data)
        rows = data.get("data") if isinstance(data, dict) else []
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            vector = rows[0].get("embedding") or []
            return [float(x) for x in vector]
        return []
    _print_llm_io("ollama_embedding_request_json", {"host": ollama_host(), "model": model, "input": text or "AceJAM"})
    response = _ollama_client().embed(model=model, input=text or "AceJAM")
    _print_llm_io("ollama_embedding_response_json", response)
    vectors = _attr(response, "embeddings", [])
    if isinstance(vectors, list) and vectors:
        return [float(x) for x in vectors[0]]
    return []


def test_model(provider: Any, model_name: str, kind: str = "chat", options: dict[str, Any] | None = None) -> dict[str, Any]:
    provider_name = normalize_provider(provider)
    model = resolve_model(provider_name, model_name, "embedding" if kind == "embedding" else "chat")
    started = time.perf_counter()
    if kind == "embedding":
        vector = embed(provider_name, model, "AceJAM embedding test")
        return {
            "success": True,
            "provider": provider_name,
            "model": model,
            "kind": "embedding",
            "dimensions": len(vector),
            "elapsed": round(time.perf_counter() - started, 3),
        }
    chat_options = planner_llm_options_for_provider(provider_name, options or {}, default_max_tokens=16)
    chat_options["temperature"] = (options or {}).get("temperature", chat_options.get("temperature", 0.0))
    text = chat_completion(
        provider_name,
        model,
        [{"role": "user", "content": "Reply with just: OK"}],
        options=chat_options,
    )
    return {
        "success": True,
        "provider": provider_name,
        "model": model,
        "kind": "chat",
        "response": text[:200],
        "elapsed": round(time.perf_counter() - started, 3),
    }
