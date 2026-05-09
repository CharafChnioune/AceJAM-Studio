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


def main() -> None:
    if not (VENDOR_DIR / "train.py").is_file():
        raise SystemExit(f"ACE-Step vendor checkout is missing: {VENDOR_DIR}")
    changed = [
        "preprocess_audio" if patch_preprocess_audio() else "",
        "variant_dir_map" if patch_variant_dir_map() else "",
        "chunked_training_scheduler_epochs" if patch_chunked_training_scheduler_epochs() else "",
        "lora_adapter_name_sanitizer" if patch_lora_adapter_name_sanitizer() else "",
        "mps_training_auto_precision" if patch_mps_training_auto_precision() else "",
        "mlx_single_seed_propagation" if patch_mlx_single_seed_propagation() else "",
    ]
    applied = [item for item in changed if item]
    if applied:
        print("Applied ACE-Step vendor patches: " + ", ".join(applied))
    else:
        print("ACE-Step vendor patches already applied")


if __name__ == "__main__":
    main()
