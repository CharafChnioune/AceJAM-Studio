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


def main() -> None:
    if not (VENDOR_DIR / "train.py").is_file():
        raise SystemExit(f"ACE-Step vendor checkout is missing: {VENDOR_DIR}")
    changed = [
        "preprocess_audio" if patch_preprocess_audio() else "",
        "variant_dir_map" if patch_variant_dir_map() else "",
        "chunked_training_scheduler_epochs" if patch_chunked_training_scheduler_epochs() else "",
        "lora_adapter_name_sanitizer" if patch_lora_adapter_name_sanitizer() else "",
    ]
    applied = [item for item in changed if item]
    if applied:
        print("Applied ACE-Step vendor patches: " + ", ".join(applied))
    else:
        print("ACE-Step vendor patches already applied")


if __name__ == "__main__":
    main()
