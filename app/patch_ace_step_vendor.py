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


def main() -> None:
    if not (VENDOR_DIR / "train.py").is_file():
        raise SystemExit(f"ACE-Step vendor checkout is missing: {VENDOR_DIR}")
    changed = [
        "preprocess_audio" if patch_preprocess_audio() else "",
        "variant_dir_map" if patch_variant_dir_map() else "",
    ]
    applied = [item for item in changed if item]
    if applied:
        print("Applied ACE-Step vendor patches: " + ", ".join(applied))
    else:
        print("ACE-Step vendor patches already applied")


if __name__ == "__main__":
    main()
