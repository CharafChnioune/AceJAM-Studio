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
    "planner_max_tokens": 8192,
    "planner_context_length": 32768,
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
    if value in {"ace-step-lm", "ace-lm", "acestep-lm", "acestep", "ace", "5hz-lm", "ace-step-5hz-lm"}:
        return "ace_step_lm"
    if value in {"lmstudio", "lm-studio", "lm studio"}:
        return "lmstudio"
    return "ollama"


def provider_label(provider: Any) -> str:
    provider_name = normalize_provider(provider)
    if provider_name == "ace_step_lm":
        return "ACE-Step 5Hz LM"
    return "LM Studio" if provider_name == "lmstudio" else "Ollama"


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


def _is_image_generation_name(name: str) -> bool:
    return bool(re.search(r"(^x/(?:z-image|flux)|\b(?:z-image|flux|imagegen|image-gen|text-to-image|txt2img|sdxl|stable-diffusion|qwen-image)\b)", name or "", re.I))


def _is_vision_name(name: str) -> bool:
    return bool(re.search(r"(vision|vl\b|llava|bakllava|moondream|minicpm-v|gemma3)", name or "", re.I))


def _ollama_client():
    import ollama

    return ollama.Client(host=ollama_host())


def _ollama_show_details(client: Any, name: str) -> dict[str, Any]:
    try:
        shown = client.show(name)
        return _llm_debug_jsonable(shown) if isinstance(_llm_debug_jsonable(shown), dict) else {}
    except Exception:
        return {}


def _model_capability_list(name: str, raw: dict[str, Any] | None = None) -> list[str]:
    raw = raw if isinstance(raw, dict) else {}
    caps: set[str] = set()
    raw_caps = raw.get("capabilities")
    if isinstance(raw_caps, list):
        caps.update(str(item).strip().lower() for item in raw_caps if str(item).strip())
    elif isinstance(raw_caps, dict):
        caps.update(str(key).strip().lower() for key, value in raw_caps.items() if value)
    details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
    families = details.get("families") if isinstance(details.get("families"), list) else []
    haystack = " ".join([name, str(details.get("family") or ""), " ".join(str(item) for item in families)]).lower()
    if _is_embedding_name(name):
        caps.add("embedding")
    if _is_image_generation_name(name):
        caps.add("image_generation")
    if _is_vision_name(name) or "vision" in haystack or "vl" in haystack:
        caps.add("vision")
    if "embedding" not in caps and "image_generation" not in caps:
        caps.add("chat")
    return sorted(caps)


def _model_kind_from_capabilities(name: str, capabilities: list[str]) -> str:
    caps = set(capabilities or [])
    if "embedding" in caps or _is_embedding_name(name):
        return "embedding"
    if "image_generation" in caps or _is_image_generation_name(name):
        return "image_generation"
    return "chat"


def ollama_model_catalog(*, enrich: bool = False) -> dict[str, Any]:
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
            shown = _ollama_show_details(client, name) if enrich else {}
            raw_for_caps = shown if shown else _llm_debug_jsonable(item)
            capabilities = _model_capability_list(name, raw_for_caps if isinstance(raw_for_caps, dict) else {})
            kind = _model_kind_from_capabilities(name, capabilities)
            details.append(
                {
                    "name": name,
                    "model": name,
                    "provider": "ollama",
                    "kind": kind,
                    "type": "embedding" if kind == "embedding" else ("image_generation" if kind == "image_generation" else "llm"),
                    "size": size,
                    "size_gb": round(size / 1e9, 2) if size else 0,
                    "modified_at": str(_attr(item, "modified_at", "") or ""),
                    "digest": str(_attr(item, "digest", "") or ""),
                    "family": str(_attr(model_details, "family", "") or (model_details.get("family", "") if isinstance(model_details, dict) else "")),
                    "parameter_size": str(_attr(model_details, "parameter_size", "") or (model_details.get("parameter_size", "") if isinstance(model_details, dict) else "")),
                    "quantization_level": str(_attr(model_details, "quantization_level", "") or (model_details.get("quantization_level", "") if isinstance(model_details, dict) else "")),
                    "format": str(_attr(model_details, "format", "") or (model_details.get("format", "") if isinstance(model_details, dict) else "")),
                    "capabilities": capabilities,
                    "vision": "vision" in capabilities,
                    "image_generation": "image_generation" in capabilities,
                    "loaded": False,
                    "raw_show": shown if enrich else {},
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
        image_models = [item["name"] for item in details if item["kind"] == "image_generation"]
        chat_models = [item["name"] for item in details if item["kind"] == "chat"]
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
            "image_models": image_models,
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
            "image_models": [],
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


def _lmstudio_quantization_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("level") or value.get("quantization") or "")
    return str(value or "")


def _lmstudio_capabilities(raw: dict[str, Any], name: str) -> dict[str, Any]:
    caps = raw.get("capabilities") if isinstance(raw.get("capabilities"), dict) else {}
    reasoning = caps.get("reasoning") if isinstance(caps.get("reasoning"), dict) else raw.get("reasoning")
    return {
        "vision": bool(caps.get("vision") or raw.get("vision") or _is_vision_name(name)),
        "trained_for_tool_use": bool(caps.get("trained_for_tool_use") or caps.get("tool_use") or raw.get("trained_for_tool_use") or raw.get("tool_use")),
        "reasoning": reasoning if isinstance(reasoning, dict) else ({"default": "on"} if reasoning else {}),
    }


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
            fmt = raw.get("format") or raw.get("format_type") or ""
            capabilities = _lmstudio_capabilities(raw, name)
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
                    "quantization_level": _lmstudio_quantization_name(quant),
                    "quantization": quant if isinstance(quant, dict) else {"name": str(quant or "")},
                    "size": int(raw.get("size_bytes") or raw.get("size") or 0),
                    "size_gb": round(int(raw.get("size_bytes") or raw.get("size") or 0) / 1e9, 2) if int(raw.get("size_bytes") or raw.get("size") or 0) else 0,
                    "params_string": str(raw.get("params_string") or ""),
                    "architecture": str(raw.get("architecture") or raw.get("arch") or ""),
                    "context_length": raw.get("context_length") or raw.get("contextLength") or raw.get("max_context_length") or "",
                    "max_context_length": raw.get("max_context_length") or raw.get("maxContextLength") or raw.get("context_length") or raw.get("contextLength") or "",
                    "loaded_context_length": loaded_context_length,
                    "capabilities": capabilities,
                    "vision": bool(capabilities.get("vision")),
                    "reasoning": bool(capabilities.get("reasoning") or raw.get("is_reasoning_model") or False),
                    "mlx_preferred": str(fmt or "").lower() == "mlx",
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
            "image_models": [],
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
            "image_models": [],
            "details": [],
            "loaded_models": [],
            "running_models": [],
            "error": f"LM Studio is not reachable at {lmstudio_base_url()}: {exc}",
        }


def model_catalog(provider: Any) -> dict[str, Any]:
    provider_name = normalize_provider(provider)
    if provider_name == "ace_step_lm":
        return {
            "ready": False,
            "provider": "ace_step_lm",
            "provider_label": provider_label(provider_name),
            "models": [],
            "chat_models": [],
            "embedding_models": [],
            "image_models": [],
            "details": [],
            "error": "ACE-Step 5Hz LM is exposed through AceJAM writer helper routes, not the generic local chat/embedding API.",
        }
    return lmstudio_model_catalog() if provider_name == "lmstudio" else ollama_model_catalog()


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
    payload = {"instance_id": model}
    data = _http_json("POST", f"{lmstudio_native_base_url()}/models/unload", payload, timeout=60)
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
    for path in (f"/models/download/status/{quoted}", f"/models/download/{quoted}", f"/download/{quoted}"):
        try:
            data = _http_json("GET", f"{lmstudio_native_base_url()}{path}", timeout=20)
            return {"success": True, "provider": "lmstudio", "job_id": job, "job": data}
        except LocalLLMError:
            continue
    raise LocalLLMError(f"LM Studio download job {job} was not found.")


def _ollama_response_content(data: Any) -> str:
    return str(_attr(_attr(data, "message", {}), "content", "") or _attr(data, "response", "") or "")


def _lmstudio_response_content(data: Any) -> str:
    choices = data.get("choices") if isinstance(data, dict) else None
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return str(choices[0].get("text") or "")
    return str(data.get("raw") if isinstance(data, dict) else data or "")


def ollama_chat_response(
    model_name: str,
    messages: list[dict[str, str]],
    *,
    options: dict[str, Any] | None = None,
    json_format: bool = False,
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_timeout = float(os.environ.get("ACEJAM_OLLAMA_CHAT_TIMEOUT", "600"))
    kwargs: dict[str, Any] = {"model": model_name, "messages": messages, "think": False, "stream": False}
    if options:
        ollama_options = dict(options)
        request_timeout = float(ollama_options.pop("timeout", request_timeout) or request_timeout)
        if "max_tokens" in ollama_options and "num_predict" not in ollama_options:
            ollama_options["num_predict"] = ollama_options.pop("max_tokens")
        kwargs["options"] = ollama_options
    if json_schema:
        kwargs["format"] = json_schema
    elif json_format:
        kwargs["format"] = "json"
    _print_llm_io("ollama_chat_request_json", {"url": f"{ollama_host().rstrip('/')}/api/chat", "payload": kwargs})
    data = _http_json("POST", f"{ollama_host().rstrip('/')}/api/chat", kwargs, timeout=request_timeout)
    _print_llm_io("ollama_chat_response_json", data)
    content = _ollama_response_content(data)
    _print_llm_io("ollama_chat_content", content)
    return {
        "content": content,
        "provider": "ollama",
        "model": model_name,
        "raw": data,
        "done_reason": str(data.get("done_reason") or "") if isinstance(data, dict) else "",
        "truncated": str(data.get("done_reason") or "").lower() == "length" if isinstance(data, dict) else False,
    }


def ollama_chat(
    model_name: str,
    messages: list[dict[str, str]],
    *,
    options: dict[str, Any] | None = None,
    json_format: bool = False,
    json_schema: dict[str, Any] | None = None,
) -> str:
    content = ollama_chat_response(
        model_name,
        messages,
        options=options,
        json_format=json_format,
        json_schema=json_schema,
    )["content"]
    return content


def _lmstudio_json_schema_response_format(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "acejam_structured_result",
            "strict": True,
            "schema": schema,
        },
    }


def lmstudio_chat_response(
    model_name: str,
    messages: list[dict[str, str]],
    *,
    options: dict[str, Any] | None = None,
    json_format: bool = False,
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_timeout = float((options or {}).get("timeout") or 600)
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": False,
    }
    payload.update(_lmstudio_openai_chat_options(options))
    if json_schema:
        payload["response_format"] = _lmstudio_json_schema_response_format(json_schema)
    elif json_format:
        payload.setdefault("response_format", {"type": "json_object"})
    _print_llm_io("lmstudio_chat_request_json", {"url": f"{lmstudio_api_base_url()}/chat/completions", "payload": payload})
    data = _http_json("POST", f"{lmstudio_api_base_url()}/chat/completions", payload, timeout=request_timeout)
    _print_llm_io("lmstudio_chat_response_json", data)
    content = _lmstudio_response_content(data)
    _print_llm_io("lmstudio_chat_content", content)
    finish_reason = ""
    choices = data.get("choices") if isinstance(data, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish_reason = str(choices[0].get("finish_reason") or choices[0].get("native_finish_reason") or "")
    return {
        "content": content,
        "provider": "lmstudio",
        "model": model_name,
        "raw": data,
        "done_reason": finish_reason,
        "truncated": finish_reason.lower() in {"length", "max_tokens", "content_filter_length"},
    }


def lmstudio_chat(
    model_name: str,
    messages: list[dict[str, str]],
    *,
    options: dict[str, Any] | None = None,
    json_format: bool = False,
    json_schema: dict[str, Any] | None = None,
) -> str:
    content = lmstudio_chat_response(
        model_name,
        messages,
        options=options,
        json_format=json_format,
        json_schema=json_schema,
    )["content"]
    return content


def chat_completion_response(
    provider: Any,
    model_name: str,
    messages: list[dict[str, str]],
    *,
    options: dict[str, Any] | None = None,
    json_format: bool = False,
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider_name = normalize_provider(provider)
    if provider_name == "ace_step_lm":
        raise LocalLLMError("ACE-Step 5Hz LM is not a generic chat backend; use the official ACE-Step writer helper route.")
    model = resolve_model(provider_name, model_name, "chat")
    if provider_name == "lmstudio":
        return lmstudio_chat_response(model, messages, options=options, json_format=json_format, json_schema=json_schema)
    return ollama_chat_response(model, messages, options=options, json_format=json_format, json_schema=json_schema)


def chat_completion(
    provider: Any,
    model_name: str,
    messages: list[dict[str, str]],
    *,
    options: dict[str, Any] | None = None,
    json_format: bool = False,
    json_schema: dict[str, Any] | None = None,
) -> str:
    return chat_completion_response(
        provider,
        model_name,
        messages,
        options=options,
        json_format=json_format,
        json_schema=json_schema,
    )["content"]


def ollama_generate_image(
    model_name: str,
    prompt: str,
    *,
    width: int = 1024,
    height: int = 1024,
    steps: int | None = None,
    seed: int | None = None,
    negative_prompt: str = "",
    timeout: float = 1800.0,
) -> dict[str, Any]:
    model = str(model_name or "").strip()
    if not model:
        raise LocalLLMError("Ollama image model is required.")
    full_prompt = str(prompt or "").strip()
    if negative_prompt:
        full_prompt = f"{full_prompt}\n\nNegative prompt: {negative_prompt.strip()}"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "width": int(width),
        "height": int(height),
    }
    if steps:
        payload["steps"] = int(steps)
    if seed is not None:
        payload["options"] = {"seed": int(seed)}
    _print_llm_io("ollama_image_request_json", {"url": f"{ollama_host().rstrip('/')}/api/generate", "payload": payload})
    data = _http_json("POST", f"{ollama_host().rstrip('/')}/api/generate", payload, timeout=timeout)
    _print_llm_io("ollama_image_response_json", {k: ("[base64 image]" if k == "image" else v) for k, v in (data.items() if isinstance(data, dict) else [])})
    image_value = ""
    if isinstance(data, dict):
        image_value = str(data.get("image") or "")
        if not image_value and isinstance(data.get("images"), list) and data["images"]:
            image_value = str(data["images"][0] or "")
        if not image_value and str(data.get("response") or "").startswith("data:image/"):
            image_value = str(data.get("response") or "")
            data["image"] = image_value
    if not isinstance(data, dict) or not image_value:
        raise LocalLLMError("Ollama image generation did not return an image.")
    data.setdefault("image", image_value)
    return data


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
