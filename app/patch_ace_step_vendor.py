from __future__ import annotations

from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
VENDOR_DIR = APP_DIR / "vendor" / "ACE-Step-1.5"


def patch_preprocess_audio() -> bool:
    path = VENDOR_DIR / "acestep" / "training" / "dataset_builder_modules" / "preprocess_audio.py"
    text = path.read_text(encoding="utf-8")
    patched = (
        '    # Try soundfile backend first (no FFmpeg/torchcodec dependency)\n'
        "    try:\n"
        '        audio, sr = torchaudio.load(audio_path, backend="soundfile")\n'
        "    except Exception:\n"
        "        audio, sr = torchaudio.load(audio_path)\n"
    )
    if patched in text:
        return False
    original = "    audio, sr = torchaudio.load(audio_path)\n"
    if original not in text:
        raise RuntimeError(f"Could not find torchaudio.load call to patch in {path}")
    path.write_text(text.replace(original, patched, 1), encoding="utf-8")
    return True


def patch_variant_dir_map() -> bool:
    path = VENDOR_DIR / "acestep" / "training_v2" / "cli" / "args.py"
    text = path.read_text(encoding="utf-8")
    changed = False
    inserts = {
        '"turbo_shift3"': '    "turbo_shift3": "acestep-v15-turbo-shift3",\n',
        '"xl_turbo"': '    "xl_turbo": "acestep-v15-xl-turbo",\n',
        '"xl_base"': '    "xl_base": "acestep-v15-xl-base",\n',
        '"xl_sft"': '    "xl_sft": "acestep-v15-xl-sft",\n',
    }
    if inserts['"turbo_shift3"'].strip() not in text:
        anchor = '    "turbo": "acestep-v15-turbo",\n'
        if anchor not in text:
            raise RuntimeError(f"Could not find turbo variant anchor in {path}")
        text = text.replace(anchor, anchor + inserts['"turbo_shift3"'], 1)
        changed = True
    xl_lines = [line for key, line in inserts.items() if key != '"turbo_shift3"' and line.strip() not in text]
    if xl_lines:
        anchor = '    "sft": "acestep-v15-sft",\n'
        if anchor not in text:
            raise RuntimeError(f"Could not find sft variant anchor in {path}")
        text = text.replace(anchor, anchor + "".join(xl_lines), 1)
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def patch_chunked_training_scheduler_epochs() -> bool:
    changed = False

    args_path = VENDOR_DIR / "acestep" / "training_v2" / "cli" / "args.py"
    args_text = args_path.read_text(encoding="utf-8")
    scheduler_arg = (
        '    g_ckpt.add_argument("--scheduler-epochs", type=int, default=None, help=argparse.SUPPRESS)\n'
    )
    if scheduler_arg.strip() not in args_text:
        anchor = '    g_ckpt.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs (default: 10)")\n'
        if anchor not in args_text:
            raise RuntimeError(f"Could not find save-every anchor in {args_path}")
        args_text = args_text.replace(anchor, anchor + scheduler_arg, 1)
        args_path.write_text(args_text, encoding="utf-8")
        changed = True

    configs_path = VENDOR_DIR / "acestep" / "training_v2" / "configs.py"
    configs_text = configs_path.read_text(encoding="utf-8")
    configs_changed = False
    scheduler_field = (
        "    scheduler_epochs: Optional[int] = None\n"
        '    """Total epoch count used for LR scheduler when chunking training."""\n\n'
    )
    if scheduler_field.strip() not in configs_text:
        anchor = (
            "    sample_every_n_epochs: int = 0\n"
            '    """Generate an audio sample every N epochs (0 = disabled)."""\n\n'
        )
        if anchor not in configs_text:
            raise RuntimeError(f"Could not find sample_every_n_epochs field anchor in {configs_path}")
        configs_text = configs_text.replace(anchor, anchor + scheduler_field, 1)
        configs_changed = True
    dict_line = '                "scheduler_epochs": self.scheduler_epochs,\n'
    if dict_line.strip() not in configs_text:
        dict_anchor = '                "sample_every_n_epochs": self.sample_every_n_epochs,\n'
        if dict_anchor not in configs_text:
            raise RuntimeError(f"Could not find sample_every_n_epochs dict anchor in {configs_path}")
        configs_text = configs_text.replace(dict_anchor, dict_anchor + dict_line, 1)
        configs_changed = True
    if configs_changed:
        configs_path.write_text(configs_text, encoding="utf-8")
        changed = True

    builder_path = VENDOR_DIR / "acestep" / "training_v2" / "cli" / "config_builder.py"
    builder_text = builder_path.read_text(encoding="utf-8")
    builder_line = '        scheduler_epochs=getattr(args, "scheduler_epochs", None),\n'
    if builder_line.strip() not in builder_text:
        anchor = "        sample_every_n_epochs=args.sample_every_n_epochs,\n"
        if anchor not in builder_text:
            raise RuntimeError(f"Could not find sample_every_n_epochs builder anchor in {builder_path}")
        builder_text = builder_text.replace(anchor, anchor + builder_line, 1)
        builder_path.write_text(builder_text, encoding="utf-8")
        changed = True

    scheduler_block = (
        "        scheduler_epochs = int(getattr(cfg, \"scheduler_epochs\", 0) or cfg.max_epochs)\n"
        "        scheduler_epochs = max(cfg.max_epochs, scheduler_epochs)\n"
        "        total_steps = steps_per_epoch * scheduler_epochs\n"
    )
    for trainer_name, indent in [("trainer_fixed.py", "        "), ("trainer_basic_loop.py", "    ")]:
        trainer_path = VENDOR_DIR / "acestep" / "training_v2" / trainer_name
        trainer_text = trainer_path.read_text(encoding="utf-8")
        block = scheduler_block if indent == "        " else scheduler_block.replace("        ", "    ")
        if "scheduler_epochs = int(getattr(cfg, \"scheduler_epochs\", 0) or cfg.max_epochs)" in trainer_text:
            continue
        original = f"{indent}total_steps = steps_per_epoch * cfg.max_epochs\n"
        if original not in trainer_text:
            raise RuntimeError(f"Could not find total_steps scheduler anchor in {trainer_path}")
        trainer_text = trainer_text.replace(original, block, 1)
        trainer_path.write_text(trainer_text, encoding="utf-8")
        changed = True

    return changed


def patch_lora_adapter_name_sanitizer() -> bool:
    path = VENDOR_DIR / "acestep" / "core" / "generation" / "handler" / "lora" / "lifecycle.py"
    text = path.read_text(encoding="utf-8")
    original = (
        'def _default_adapter_name_from_path(lora_path: str) -> str:\n'
        '    """Derive a default adapter name from path (e.g. \'final\' from \'./lora/final\')."""\n'
        '    name = os.path.basename(lora_path.rstrip(os.sep))\n'
        '    return name if name else "default"\n'
    )
    patched = (
        'def _default_adapter_name_from_path(lora_path: str) -> str:\n'
        '    """Derive a PEFT-safe default adapter name from a path basename."""\n'
        '    import re\n\n'
        '    name = os.path.basename(lora_path.rstrip(os.sep)) or "default"\n'
        '    safe = re.sub(r"[^0-9A-Za-z_]+", "_", name)\n'
        '    safe = re.sub(r"_+", "_", safe).strip("_") or "default"\n'
        '    if safe[0].isdigit():\n'
        '        safe = f"adapter_{safe}"\n'
        '    return safe[:90]\n'
    )
    if patched in text:
        return False
    if original not in text:
        raise RuntimeError(f"Could not find LoRA adapter name helper in {path}")
    path.write_text(text.replace(original, patched, 1), encoding="utf-8")
    return True


def patch_mps_training_auto_precision() -> bool:
    changed = False

    path = VENDOR_DIR / "acestep" / "training_v2" / "gpu_utils.py"
    text = path.read_text(encoding="utf-8")
    original = (
        '        elif device_type == "mps":\n'
        '            precision = "fp16"\n'
    )
    patched = (
        '        elif device_type == "mps":\n'
        '            precision = "fp32"\n'
    )
    if patched not in text:
        if original not in text:
            raise RuntimeError(f"Could not find MPS auto precision block in {path}")
        path.write_text(text.replace(original, patched, 1), encoding="utf-8")
        changed = True

    module_path = VENDOR_DIR / "acestep" / "training_v2" / "fixed_lora_module.py"
    module_text = module_path.read_text(encoding="utf-8")
    dtype_original = (
        '    if device_type == "mps":\n'
        '        return torch.float16\n'
    )
    dtype_patched = (
        '    if device_type == "mps":\n'
        '        return torch.float32\n'
    )
    if dtype_patched not in module_text:
        if dtype_original not in module_text:
            raise RuntimeError(f"Could not find MPS compute dtype block in {module_path}")
        module_text = module_text.replace(dtype_original, dtype_patched, 1)
        changed = True

    fabric_original = (
        '    if device_type == "mps":\n'
        '        return "16-mixed"\n'
    )
    fabric_patched = (
        '    if device_type == "mps":\n'
        '        return "32-true"\n'
    )
    if fabric_patched not in module_text:
        if fabric_original not in module_text:
            raise RuntimeError(f"Could not find MPS fabric precision block in {module_path}")
        module_text = module_text.replace(fabric_original, fabric_patched, 1)
        changed = True

    if changed:
        module_path.write_text(module_text, encoding="utf-8")
    return changed


def patch_bitsandbytes_non_cuda_warning() -> bool:
    path = VENDOR_DIR / "acestep" / "training" / "trainer.py"
    text = path.read_text(encoding="utf-8")
    patched = (
        "# OPTIMIZATION: Use 8-bit Adam to save CUDA VRAM.\n"
        "#\n"
        "# bitsandbytes is a CUDA-oriented optional optimizer backend. Importing it on\n"
        "# Apple/MPS only produces noise and cannot be used below anyway, so keep MPS\n"
        "# training on the documented AdamW path without warning spam.\n"
        "if torch.cuda.is_available():\n"
        "    try:\n"
        "        import bitsandbytes as bnb\n\n"
        "        HAS_BNB = True\n"
        "    except ImportError:\n"
        "        bnb = None\n"
        "        HAS_BNB = False\n"
        '        logger.info("bitsandbytes not installed. Using standard AdamW.")\n'
        "else:\n"
        "    bnb = None\n"
        "    HAS_BNB = False\n"
    )
    if patched in text:
        return False
    original = (
        "# OPTIMIZATION: Use 8-bit Adam to save some VRAM\n"
        "try:\n"
        "    import bitsandbytes as bnb\n\n"
        "    HAS_BNB = True\n"
        "except ImportError:\n"
        "    HAS_BNB = False\n"
        '    logger.warning("bitsandbytes not installed. Using standard AdamW.")\n'
    )
    if original not in text:
        raise RuntimeError(f"Could not find bitsandbytes import block in {path}")
    path.write_text(text.replace(original, patched, 1), encoding="utf-8")
    return True


def patch_mlx_single_seed_propagation() -> bool:
    path = VENDOR_DIR / "acestep" / "llm_inference.py"
    text = path.read_text(encoding="utf-8")
    changed = False

    phase2_original = (
        '                    "generation_phase": "codes",\n'
        "                    # Pass context for building unconditional prompt in codes phase\n"
    )
    phase2_patched = (
        '                    "generation_phase": "codes",\n'
        '                    "seeds": seeds,\n'
        "                    # Pass context for building unconditional prompt in codes phase\n"
    )
    if phase2_patched not in text:
        if phase2_original not in text:
            raise RuntimeError(f"Could not find MLX phase-2 seed cfg anchor in {path}")
        text = text.replace(phase2_original, phase2_patched, 1)
        changed = True

    cfg_original = (
        '        generation_phase = cfg.get("generation_phase", "cot")  # "cot" or "codes"\n'
        "        # Additional context for codes phase unconditional prompt building\n"
    )
    cfg_patched = (
        '        generation_phase = cfg.get("generation_phase", "cot")  # "cot" or "codes"\n'
        '        seeds = cfg.get("seeds")\n'
        "        # Additional context for codes phase unconditional prompt building\n"
    )
    if cfg_patched not in text:
        if cfg_original not in text:
            raise RuntimeError(f"Could not find MLX cfg seed anchor in {path}")
        text = text.replace(cfg_original, cfg_patched, 1)
        changed = True

    mlx_call_original = (
        '            elif self.llm_backend == "mlx":\n'
        "                # MLX backend (Apple Silicon native)\n"
        "                output_text = self._run_mlx(\n"
        "                    formatted_prompts=formatted_prompt,\n"
        "                    temperature=temperature,\n"
        "                    cfg_scale=cfg_scale,\n"
        "                    negative_prompt=negative_prompt,\n"
        "                    top_k=top_k,\n"
        "                    top_p=top_p,\n"
        "                    repetition_penalty=repetition_penalty,\n"
        "                    use_constrained_decoding=use_constrained_decoding,\n"
        "                    constrained_decoding_debug=constrained_decoding_debug,\n"
        "                    target_duration=target_duration,\n"
        "                    user_metadata=user_metadata,\n"
        "                    stop_at_reasoning=stop_at_reasoning,\n"
        "                    skip_genres=skip_genres,\n"
        "                    skip_caption=skip_caption,\n"
        "                    skip_language=skip_language,\n"
        "                    generation_phase=generation_phase,\n"
        "                    caption=caption,\n"
    )
    mlx_call_patched = (
        '            elif self.llm_backend == "mlx":\n'
        "                # MLX backend (Apple Silicon native)\n"
        "                output_text = self._run_mlx(\n"
        "                    formatted_prompts=formatted_prompt,\n"
        "                    temperature=temperature,\n"
        "                    cfg_scale=cfg_scale,\n"
        "                    negative_prompt=negative_prompt,\n"
        "                    top_k=top_k,\n"
        "                    top_p=top_p,\n"
        "                    repetition_penalty=repetition_penalty,\n"
        "                    use_constrained_decoding=use_constrained_decoding,\n"
        "                    constrained_decoding_debug=constrained_decoding_debug,\n"
        "                    target_duration=target_duration,\n"
        "                    user_metadata=user_metadata,\n"
        "                    stop_at_reasoning=stop_at_reasoning,\n"
        "                    skip_genres=skip_genres,\n"
        "                    skip_caption=skip_caption,\n"
        "                    skip_language=skip_language,\n"
        "                    generation_phase=generation_phase,\n"
        "                    seeds=seeds,\n"
        "                    caption=caption,\n"
    )
    if mlx_call_patched not in text:
        if mlx_call_original not in text:
            raise RuntimeError(f"Could not find generate_from_formatted_prompt MLX call anchor in {path}")
        text = text.replace(mlx_call_original, mlx_call_patched, 1)
        changed = True

    signature_original = (
        "        caption: str,\n"
        "        lyrics: str,\n"
        "        cot_text: str,\n"
        "    ) -> str:\n"
    )
    signature_patched = (
        "        caption: str,\n"
        "        lyrics: str,\n"
        "        cot_text: str,\n"
        "        seed: Optional[int] = None,\n"
        "    ) -> str:\n"
    )
    if text.count(signature_patched) < 2:
        if text.count(signature_original) < 2:
            raise RuntimeError(f"Could not find both MLX single signature anchors in {path}")
        text = text.replace(signature_original, signature_patched, 2)
        changed = True

    native_call_original = (
        "                caption=caption,\n"
        "                lyrics=lyrics,\n"
        "                cot_text=cot_text,\n"
        "            )\n"
    )
    native_call_patched = (
        "                caption=caption,\n"
        "                lyrics=lyrics,\n"
        "                cot_text=cot_text,\n"
        "                seed=seed,\n"
        "            )\n"
    )
    if native_call_patched not in text:
        if native_call_original not in text:
            raise RuntimeError(f"Could not find native MLX single call anchor in {path}")
        text = text.replace(native_call_original, native_call_patched, 1)
        changed = True

    decode_original = (
        "        decode_start = time.time()\n\n"
        '        pbar = tqdm(total=max_new_tokens, desc=tqdm_desc, unit="tok")\n'
    )
    decode_patched = (
        "        decode_start = time.time()\n"
        "        try:\n"
        "            seed_base = int(seed) if seed is not None and int(seed) >= 0 else None\n"
        "        except (TypeError, ValueError):\n"
        "            seed_base = None\n\n"
        '        pbar = tqdm(total=max_new_tokens, desc=tqdm_desc, unit="tok")\n'
    )
    if text.count(decode_patched) < 2:
        if text.count(decode_original) < 2:
            raise RuntimeError(f"Could not find both MLX decode seed anchors in {path}")
        text = text.replace(decode_original, decode_patched, 2)
        changed = True

    native_loop_original = (
        "        for step in range(max_new_tokens):\n"
        "            # ---- Combine logits (CFG formula in MLX, lazy) ----\n"
    )
    native_loop_patched = (
        "        for step in range(max_new_tokens):\n"
        "            if seed_base is not None:\n"
        "                mx.random.seed(seed_base + step * 1000003)\n\n"
        "            # ---- Combine logits (CFG formula in MLX, lazy) ----\n"
    )
    if native_loop_patched not in text:
        if native_loop_original not in text:
            raise RuntimeError(f"Could not find native MLX loop seed anchor in {path}")
        text = text.replace(native_loop_original, native_loop_patched, 1)
        changed = True

    hybrid_loop_original = (
        "        for step in range(max_new_tokens):\n"
        "            # Apply CFG formula in MLX\n"
    )
    hybrid_loop_patched = (
        "        for step in range(max_new_tokens):\n"
        "            if seed_base is not None:\n"
        "                mx.random.seed(seed_base + step * 1000003)\n"
        "                torch.manual_seed(seed_base + step * 1000003)\n\n"
        "            # Apply CFG formula in MLX\n"
    )
    if hybrid_loop_patched not in text:
        if hybrid_loop_original not in text:
            raise RuntimeError(f"Could not find hybrid MLX loop seed anchor in {path}")
        text = text.replace(hybrid_loop_original, hybrid_loop_patched, 1)
        changed = True

    sequential_call_original = (
        "                    caption=caption,\n"
        "                    lyrics=lyrics,\n"
        "                    cot_text=cot_text,\n"
        "                )\n"
    )
    sequential_call_patched = (
        "                    caption=caption,\n"
        "                    lyrics=lyrics,\n"
        "                    cot_text=cot_text,\n"
        "                    seed=seeds[i] if seeds and i < len(seeds) else None,\n"
        "                )\n"
    )
    if sequential_call_patched not in text:
        if sequential_call_original not in text:
            raise RuntimeError(f"Could not find sequential MLX single seed anchor in {path}")
        text = text.replace(sequential_call_original, sequential_call_patched, 1)
        changed = True

    single_call_original = (
        "            caption=caption,\n"
        "            lyrics=lyrics,\n"
        "            cot_text=cot_text,\n"
        "        )\n"
    )
    single_call_patched = (
        "            caption=caption,\n"
        "            lyrics=lyrics,\n"
        "            cot_text=cot_text,\n"
        "            seed=seeds[0] if seeds else None,\n"
        "        )\n"
    )
    if single_call_patched not in text:
        if single_call_original not in text:
            raise RuntimeError(f"Could not find single MLX seed anchor in {path}")
        text = text.replace(single_call_original, single_call_patched, 1)
        changed = True

    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def patch_mlx_effective_lora_sync() -> bool:
    """Teach native MLX DiT conversion to mirror active PyTorch/PEFT LoRA state."""
    changed = False

    convert_path = VENDOR_DIR / "acestep" / "models" / "mlx" / "dit_convert.py"
    convert_text = convert_path.read_text(encoding="utf-8")
    if "from typing import Any, Dict, List, Tuple" not in convert_text:
        original = "from typing import Dict, List, Tuple\n"
        if original not in convert_text:
            raise RuntimeError(f"Could not find typing import anchor in {convert_path}")
        convert_text = convert_text.replace(original, "from typing import Any, Dict, List, Tuple\n", 1)
        changed = True
    if "def _effective_decoder_state_dict(decoder: Any)" not in convert_text:
        anchor = "logger = logging.getLogger(__name__)\n\n\n"
        if anchor not in convert_text:
            raise RuntimeError(f"Could not find MLX converter logger anchor in {convert_path}")
        helper_block = '''logger = logging.getLogger(__name__)


def _active_lora_adapters(module: Any) -> list[str]:
    """Return the active PEFT LoRA adapters for a wrapped module."""
    if bool(getattr(module, "disable_adapters", False)):
        return []
    active = getattr(module, "active_adapters", None)
    if active is None:
        active = getattr(module, "active_adapter", None)
    if isinstance(active, str):
        return [active]
    if isinstance(active, (list, tuple, set)):
        return [name for name in active if isinstance(name, str)]
    return []


def _effective_lora_layer_state(module: Any) -> dict[str, "torch.Tensor"] | None:
    """Return base-layer state with active PEFT LoRA deltas merged in-memory.

    This is intentionally non-destructive: it mirrors PEFT's forward/merge math
    without calling ``merge_and_unload`` or mutating the PyTorch decoder. Native
    MLX DiT consumes static weights, so any active adapter delta has to be baked
    into the weights before ``load_weights``.
    """
    get_base_layer = getattr(module, "get_base_layer", None)
    get_delta_weight = getattr(module, "get_delta_weight", None)
    if not callable(get_base_layer) or not callable(get_delta_weight):
        return None
    if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
        return None

    base_layer = get_base_layer()
    if base_layer is None or not hasattr(base_layer, "state_dict"):
        return None

    state = {key: value.detach().clone() for key, value in base_layer.state_dict().items()}
    active_adapters = _active_lora_adapters(module)
    lora_A = getattr(module, "lora_A", {})
    if "weight" not in state or not active_adapters:
        return state

    weight = state["weight"]
    for adapter in active_adapters:
        if adapter not in lora_A:
            continue
        delta = get_delta_weight(adapter).detach().to(device=weight.device, dtype=weight.dtype)
        weight = weight + delta

        lora_bias = getattr(module, "lora_bias", {})
        lora_B = getattr(module, "lora_B", {})
        if bool(lora_bias.get(adapter)) and "bias" in state and adapter in lora_B:
            bias_module = lora_B[adapter]
            bias = getattr(bias_module, "bias", None)
            scaling = getattr(module, "scaling", {}).get(adapter, 1.0)
            if bias is not None:
                state["bias"] = state["bias"] + bias.detach().to(
                    device=state["bias"].device,
                    dtype=state["bias"].dtype,
                ) * scaling

    state["weight"] = weight
    return state


def _effective_decoder_state_dict(decoder: Any) -> dict[str, "torch.Tensor"]:
    """Return a decoder state dict with active PEFT LoRA deltas baked in."""
    get_base_model = getattr(decoder, "get_base_model", None)
    base_decoder = get_base_model() if callable(get_base_model) else decoder
    raw_state = base_decoder.state_dict()
    effective_state: dict[str, "torch.Tensor"] = {}
    lora_wrapped_prefixes: set[str] = set()

    for module_name, module in base_decoder.named_modules():
        layer_state = _effective_lora_layer_state(module)
        if layer_state is None:
            continue
        lora_wrapped_prefixes.add(module_name)
        prefix = f"{module_name}." if module_name else ""
        for key, value in layer_state.items():
            effective_state[f"{prefix}{key}"] = value

    for key, value in raw_state.items():
        if ".lora_" in key or key.startswith("lora_"):
            continue
        if ".base_layer." in key or key.startswith("base_layer."):
            prefix = key.split(".base_layer.", 1)[0] if ".base_layer." in key else ""
            if prefix in lora_wrapped_prefixes:
                continue
            key = key.replace(".base_layer.", ".").replace("base_layer.", "")
        effective_state.setdefault(key, value)

    return effective_state


'''
        convert_text = convert_text.replace(anchor, helper_block, 1)
        changed = True
    original_state = "    state_dict = decoder.state_dict()\n"
    patched_state = "    state_dict = _effective_decoder_state_dict(decoder)\n"
    if patched_state not in convert_text:
        if original_state not in convert_text:
            raise RuntimeError(f"Could not find decoder state_dict anchor in {convert_path}")
        convert_text = convert_text.replace(original_state, patched_state, 1)
        changed = True
    if changed:
        convert_path.write_text(convert_text, encoding="utf-8")

    init_path = VENDOR_DIR / "acestep" / "core" / "generation" / "handler" / "mlx_dit_init.py"
    init_text = init_path.read_text(encoding="utf-8")
    if "def _sync_mlx_dit_weights_from_torch" not in init_text:
        anchor = 'class MlxDitInitMixin:\n    """Initialize native MLX DiT decoder state used by generation runtime."""\n'
        if anchor not in init_text:
            raise RuntimeError(f"Could not find MlxDitInitMixin class anchor in {init_path}")
        helper_block = anchor + '''
    def _mlx_dit_weight_sync_key(self) -> tuple:
        """Return a compact key for the PyTorch decoder state mirrored to MLX."""
        active_loras = getattr(self, "_active_loras", {}) or {}
        return (
            id(getattr(getattr(self, "model", None), "decoder", None)),
            bool(getattr(self, "lora_loaded", False)),
            bool(getattr(self, "use_lora", False)),
            getattr(self, "_adapter_type", None),
            getattr(self, "_lora_active_adapter", None),
            tuple(sorted((str(name), float(scale)) for name, scale in active_loras.items())),
        )

    def _sync_mlx_dit_weights_from_torch(self, reason: str = "manual", force: bool = False) -> dict:
        """Refresh native MLX DiT weights from the effective PyTorch decoder.

        PEFT LoRA adapters are applied dynamically in PyTorch modules, while the
        native MLX decoder is a static converted copy. Re-syncing here bakes any
        active adapter delta and scale into the MLX weights without mutating the
        PyTorch decoder.
        """
        if not bool(getattr(self, "use_mlx_dit", False)) or getattr(self, "mlx_decoder", None) is None:
            return {"synced": False, "reason": "mlx_dit_inactive"}
        if getattr(self, "model", None) is None or getattr(self.model, "decoder", None) is None:
            return {"synced": False, "reason": "torch_decoder_unavailable"}

        sync_key = self._mlx_dit_weight_sync_key()
        if not force and getattr(self, "_mlx_dit_weight_sync_key_cache", None) == sync_key:
            return {"synced": False, "reason": "unchanged", "sync_reason": reason}

        try:
            from acestep.models.mlx.dit_convert import convert_and_load

            convert_and_load(self.model, self.mlx_decoder)
            self.mlx_decoder.materialize_static_buffers()
            self._mlx_dit_weight_sync_key_cache = sync_key
            logger.info(f"[MLX-DiT] Synced effective PyTorch decoder weights to MLX (reason={reason}).")
            return {"synced": True, "reason": reason}
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[MLX-DiT] Failed to sync effective PyTorch decoder weights to MLX: {exc}")
            return {"synced": False, "reason": "sync_failed", "error": str(exc), "sync_reason": reason}
'''
        init_text = init_text.replace(anchor, helper_block, 1)
        changed = True
    cache_line = "            self._mlx_dit_weight_sync_key_cache = self._mlx_dit_weight_sync_key()\n"
    cache_anchor = "            self.mlx_dit_compiled = compile_model\n"
    if cache_line not in init_text:
        if cache_anchor not in init_text:
            raise RuntimeError(f"Could not find MLX init sync key anchor in {init_path}")
        init_text = init_text.replace(cache_anchor, cache_anchor + cache_line, 1)
        changed = True
    if changed:
        init_path.write_text(init_text, encoding="utf-8")

    lifecycle_path = VENDOR_DIR / "acestep" / "core" / "generation" / "handler" / "lora" / "lifecycle.py"
    lifecycle_text = lifecycle_path.read_text(encoding="utf-8")
    sync_loaded = (
        '        sync_mlx = getattr(self, "_sync_mlx_dit_weights_from_torch", None)\n'
        "        if callable(sync_mlx):\n"
        '            sync_mlx(reason="lora_loaded", force=True)\n'
    )
    if sync_loaded not in lifecycle_text:
        anchor = '            prefix="lora",\n        )\n        return f"✅ LoRA \'{effective_name}\' loaded from {lora_path}"\n'
        if anchor not in lifecycle_text:
            raise RuntimeError(f"Could not find LoRA loaded return anchor in {lifecycle_path}")
        lifecycle_text = lifecycle_text.replace(anchor, anchor.replace('        return', sync_loaded + '        return'), 1)
        changed = True
    lifecycle_sync_specs = [
        (
            "lora_removed_last_no_backup",
            '                self._lora_scale_state = {}\n                return "✅ Last adapter removed; base decoder still wrapped (no backup). Restart or load a new LoRA."\n',
            '                sync_mlx = getattr(self, "_sync_mlx_dit_weights_from_torch", None)\n'
            "                if callable(sync_mlx):\n"
            '                    sync_mlx(reason="lora_removed_last_no_backup", force=True)\n',
        ),
        (
            "lora_removed_last",
            '            logger.info("LoRA unloaded, base decoder restored")\n            return "✅ LoRA unloaded, using base model"\n',
            '            sync_mlx = getattr(self, "_sync_mlx_dit_weights_from_torch", None)\n'
            "            if callable(sync_mlx):\n"
            '                sync_mlx(reason="lora_removed_last", force=True)\n',
        ),
        (
            "lora_removed",
            '        logger.info(f"Adapter \'{adapter_name}\' removed. Active: {next_active}")\n        return f"✅ Adapter \'{adapter_name}\' removed. Active: {next_active}"\n',
            '        sync_mlx = getattr(self, "_sync_mlx_dit_weights_from_torch", None)\n'
            "        if callable(sync_mlx):\n"
            '            sync_mlx(reason="lora_removed", force=True)\n',
        ),
        (
            "lora_unloaded",
            '        logger.info("LoRA unloaded, base decoder restored")\n        return "✅ LoRA unloaded, using base model"\n',
            '        sync_mlx = getattr(self, "_sync_mlx_dit_weights_from_torch", None)\n'
            "        if callable(sync_mlx):\n"
            '            sync_mlx(reason="lora_unloaded", force=True)\n',
        ),
    ]
    for marker, anchor, snippet in lifecycle_sync_specs:
        if f'sync_mlx(reason="{marker}", force=True)' in lifecycle_text:
            continue
        if anchor not in lifecycle_text:
            raise RuntimeError(f"Could not find {marker} return anchor in {lifecycle_path}")
        lifecycle_text = lifecycle_text.replace(anchor, anchor.replace("return", snippet + "return"), 1)
        changed = True
    if changed:
        lifecycle_path.write_text(lifecycle_text, encoding="utf-8")

    controls_path = VENDOR_DIR / "acestep" / "core" / "generation" / "handler" / "lora" / "controls.py"
    controls_text = controls_path.read_text(encoding="utf-8")
    sync_toggle = (
        '    sync_mlx = getattr(self, "_sync_mlx_dit_weights_from_torch", None)\n'
        "    if callable(sync_mlx):\n"
        '        sync_mlx(reason="lora_toggled", force=True)\n\n'
    )
    if sync_toggle not in controls_text:
        anchor = '    adapter_label = "LoKr" if getattr(self, "_adapter_type", None) == "lokr" else "LoRA"\n'
        if anchor not in controls_text:
            raise RuntimeError(f"Could not find LoRA toggle status anchor in {controls_path}")
        controls_text = controls_text.replace(anchor, sync_toggle + anchor, 1)
        changed = True
    sync_scale = (
        '            sync_mlx = getattr(self, "_sync_mlx_dit_weights_from_torch", None)\n'
        "            if callable(sync_mlx):\n"
        '                sync_mlx(reason="lora_scale_changed", force=True)\n'
    )
    if 'sync_mlx(reason="lora_scale_changed", force=True)' not in controls_text:
        anchor = '            return (\n                f"✅ LoRA scale ({effective_name}): {scale_value:.2f}"\n'
        if anchor not in controls_text:
            raise RuntimeError(f"Could not find LoRA scale return anchor in {controls_path}")
        controls_text = controls_text.replace(anchor, sync_scale + anchor, 1)
        changed = True
    sync_active = (
        '    sync_mlx = getattr(self, "_sync_mlx_dit_weights_from_torch", None)\n'
        "    if callable(sync_mlx):\n"
        '        sync_mlx(reason="lora_active_adapter_changed", force=True)\n'
    )
    if sync_active not in controls_text:
        anchor = '    return f"✅ Active LoRA adapter: {adapter_name}"\n'
        if anchor not in controls_text:
            raise RuntimeError(f"Could not find active LoRA adapter return anchor in {controls_path}")
        controls_text = controls_text.replace(anchor, sync_active + anchor, 1)
        changed = True
    if changed:
        controls_path.write_text(controls_text, encoding="utf-8")

    return changed


def main() -> None:
    if not (VENDOR_DIR / "train.py").is_file():
        raise SystemExit(f"ACE-Step vendor checkout is missing: {VENDOR_DIR}")
    changed = [
        "preprocess_audio" if patch_preprocess_audio() else "",
        "variant_dir_map" if patch_variant_dir_map() else "",
        "chunked_training_scheduler_epochs" if patch_chunked_training_scheduler_epochs() else "",
        "lora_adapter_name_sanitizer" if patch_lora_adapter_name_sanitizer() else "",
        "mps_training_auto_precision" if patch_mps_training_auto_precision() else "",
        "bitsandbytes_non_cuda_warning" if patch_bitsandbytes_non_cuda_warning() else "",
        "mlx_single_seed_propagation" if patch_mlx_single_seed_propagation() else "",
        "mlx_effective_lora_sync" if patch_mlx_effective_lora_sync() else "",
    ]
    applied = [item for item in changed if item]
    if applied:
        print("Applied ACE-Step vendor patches: " + ", ".join(applied))
    else:
        print("ACE-Step vendor patches already applied")


if __name__ == "__main__":
    main()
