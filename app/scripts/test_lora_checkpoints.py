#!/usr/bin/env python3
"""Render a small comparison set for trained AceJAM LoRA checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:42003"
DEFAULT_LYRICS = """[Verse]
Street lights blink, I move at a steady pace.
Fresh ink clean, every word has space.
Bass hits low while the snare stays tight.
I keep my head high through the long night.
No copied lines, just a new clear test.
Voice up front with the beat in the chest.

[Chorus]
Keep your head high when the night gets cold.
Step by step on the open road.
If the world gets loud, let the words stay clear.
I am still right here, still right here."""


def _request_json(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 1800) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(
        base_url.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as res:
            raw = res.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed with HTTP {exc.code}: {raw[:1200]}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {path} failed: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{method} {path} returned non-JSON: {raw[:1200]}") from exc


def _epoch_loss(name: str) -> tuple[int | None, float | None]:
    match = re.search(r"epoch[_-](\d+).*?loss[_-]([0-9]+)[_-]([0-9]+)", name, re.IGNORECASE)
    if not match:
        return None, None
    epoch = int(match.group(1))
    loss = float(f"{match.group(2)}.{match.group(3)}")
    return epoch, loss


def _load_adapters(base_url: str, trigger: str) -> list[dict[str, Any]]:
    data = _request_json(base_url, "GET", "/api/lora/adapters", timeout=30)
    adapters = data.get("adapters") or []
    selected: list[dict[str, Any]] = []
    trigger_lower = trigger.lower()
    for adapter in adapters:
        name = str(adapter.get("name") or adapter.get("display_name") or "")
        label = str(adapter.get("label") or "")
        path = str(adapter.get("path") or "")
        searchable = " ".join([name, label, path, str(adapter.get("trigger_tag") or "")]).lower()
        if trigger_lower not in searchable:
            continue
        if str(adapter.get("adapter_type") or "").lower() != "lora":
            continue
        if not (adapter.get("generation_loadable") or adapter.get("is_loadable")):
            continue
        epoch, loss = _epoch_loss(name)
        item = dict(adapter)
        item["epoch"] = epoch
        item["loss"] = loss
        selected.append(item)
    selected.sort(key=lambda a: (a.get("epoch") is None, a.get("epoch") or 10**9, a.get("loss") or 10**9))
    return selected


def _smart_selection(adapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(adapters) <= 4:
        return adapters
    chosen: list[dict[str, Any]] = []

    def add(item: dict[str, Any] | None) -> None:
        if item and all(str(item.get("path")) != str(existing.get("path")) for existing in chosen):
            chosen.append(item)

    add(adapters[0])
    add(next((item for item in adapters if item.get("loss") is not None and float(item["loss"]) < 0.90), None))
    add(min((item for item in adapters if item.get("loss") is not None), key=lambda item: float(item["loss"]), default=None))
    add(max((item for item in adapters if item.get("epoch") is not None), key=lambda item: int(item["epoch"]), default=adapters[-1]))
    return sorted(chosen, key=lambda a: (a.get("epoch") is None, a.get("epoch") or 10**9))


def _filter_epochs(adapters: list[dict[str, Any]], epochs: str) -> list[dict[str, Any]]:
    wanted = {int(part) for part in re.split(r"[, ]+", epochs.strip()) if part}
    return [adapter for adapter in adapters if adapter.get("epoch") in wanted]


def _payload(adapter: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    is_baseline = bool(adapter.get("baseline"))
    name = str(adapter.get("display_name") or adapter.get("label") or adapter.get("name") or Path(str(adapter.get("path") or "baseline-no-lora")).name)
    epoch = adapter.get("epoch")
    title = "2pac baseline no lora" if is_baseline else (f"2pac checkpoint {epoch}" if epoch is not None else f"2pac checkpoint {name}")
    caption_trigger = args.caption_trigger
    if caption_trigger is None:
        caption_trigger = args.trigger if (not is_baseline or args.baseline_uses_trigger) else ""
    caption_trigger = str(caption_trigger or "").strip()
    trigger_prefix = f"{caption_trigger}, " if caption_trigger else ""
    return {
        "task_type": "text2music",
        "title": title,
        "artist_name": "AceJAM LoRA Test",
        "caption": f"{trigger_prefix}original English west coast rap, clear male rap vocal, punchy drums, deep bass, polished mix",
        "tags": f"{trigger_prefix}west coast rap, hip hop, clear vocal",
        "lyrics": args.lyrics,
        "instrumental": False,
        "duration": args.duration,
        "audio_duration": args.duration,
        "bpm": args.bpm,
        "key_scale": args.key,
        "time_signature": "4",
        "vocal_language": "en",
        "song_model": "acestep-v15-xl-sft",
        "quality_profile": "chart_master",
        "seed": str(args.seed),
        "inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "shift": args.shift,
        "batch_size": 1,
        "audio_format": "wav",
        "ace_lm_model": "none",
        "lm_model": "none",
        "thinking": False,
        "use_format": False,
        "use_cot_metas": False,
        "use_cot_caption": False,
        "use_cot_lyrics": False,
        "use_lora": not is_baseline,
        "lora_adapter_path": "" if is_baseline else str(adapter["path"]),
        "lora_adapter_name": "" if is_baseline else name,
        "lora_scale": 0.0 if is_baseline else args.scale,
        "adapter_model_variant": "" if is_baseline else str(adapter.get("model_variant") or "xl_sft"),
        "vocal_intelligibility_gate": args.vocal_gate,
        "vocal_intelligibility_attempts": 1,
        "vocal_intelligibility_model_rescue": False,
        "save_to_library": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render checkpoint comparison tests for AceJAM LoRAs.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--trigger", default="2pac")
    parser.add_argument("--all", action="store_true", help="Render every matching checkpoint instead of the smart subset.")
    parser.add_argument("--dry-run", action="store_true", help="Only list checkpoints and write payloads.")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--scale", type=float, default=0.45)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=2244)
    parser.add_argument("--bpm", type=int, default=94)
    parser.add_argument("--key", default="A minor")
    parser.add_argument("--lyrics", default=DEFAULT_LYRICS)
    parser.add_argument("--caption-trigger", default=None, help="Trigger/tag text to prepend to caption and tags. Empty string disables it.")
    parser.add_argument("--baseline-uses-trigger", action="store_true", help="Also prepend the trigger/tag to the no-LoRA baseline.")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--epochs", default="", help="Comma-separated checkpoint epochs to render, e.g. 1,11,12.")
    parser.add_argument("--no-vocal-gate", dest="vocal_gate", action="store_false")
    parser.set_defaults(vocal_gate=True)
    args = parser.parse_args()

    adapters = _load_adapters(args.base_url, args.trigger)
    if not adapters:
        raise SystemExit(f"No generation-loadable LoRA adapters found for trigger/search '{args.trigger}'.")
    baseline = {"baseline": True, "name": "baseline-no-lora", "epoch": "baseline", "loss": None, "path": ""}
    if args.baseline_only:
        selected = [baseline]
    else:
        selected = _filter_epochs(adapters, args.epochs) if args.epochs.strip() else (adapters if args.all else _smart_selection(adapters))
        if args.epochs.strip() and not selected:
            raise SystemExit(f"No selected adapters match --epochs {args.epochs!r}.")
        if not args.skip_baseline:
            selected = [baseline] + selected
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = BASE_DIR / "data" / "lora_checkpoint_tests" / f"{args.trigger}-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "trigger": args.trigger,
        "available": [
            {
                "name": item.get("name"),
                "epoch": item.get("epoch"),
                "loss": item.get("loss"),
                "song_model": item.get("song_model"),
                "path": item.get("path"),
            }
            for item in adapters
        ],
        "selected": [],
    }
    print("Available checkpoints:")
    for item in adapters:
        print(f"  epoch={item.get('epoch')} loss={item.get('loss')} name={item.get('name')}")
    print("\nSelected for this run:")
    for item in selected:
        print(f"  epoch={item.get('epoch')} loss={item.get('loss')} name={item.get('name')}")

    for index, adapter in enumerate(selected, start=1):
        payload = _payload(adapter, args)
        stem = f"{index:02d}-epoch-{adapter.get('epoch') or 'unknown'}"
        (out_dir / f"{stem}-payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        record: dict[str, Any] = {
            "adapter_name": adapter.get("name"),
            "epoch": adapter.get("epoch"),
            "loss": adapter.get("loss"),
            "path": adapter.get("path"),
            "baseline": bool(adapter.get("baseline")),
            "payload_path": str(out_dir / f"{stem}-payload.json"),
        }
        if args.dry_run:
            record["status"] = "dry_run"
            summary["selected"].append(record)
            continue
        print(f"\nRendering {adapter.get('name')} ...", flush=True)
        started = time.time()
        try:
            response = _request_json(args.base_url, "POST", "/api/generate_advanced", payload, timeout=3600)
            record["elapsed_sec"] = round(time.time() - started, 1)
            record["success"] = bool(response.get("success"))
            record["result_id"] = response.get("result_id")
            record["audio_urls"] = [audio.get("audio_url") for audio in response.get("audios", []) if audio.get("audio_url")]
            record["vocal_gate"] = response.get("vocal_intelligibility_gate")
            record["needs_review"] = bool(response.get("needs_review"))
            record["error"] = response.get("error") or ""
            (out_dir / f"{stem}-response.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
            print(f"  -> success={record['success']} result={record.get('result_id')} needs_review={record['needs_review']}")
            for url in record["audio_urls"]:
                print(f"     {url}")
        except Exception as exc:
            record["elapsed_sec"] = round(time.time() - started, 1)
            record["success"] = False
            record["error"] = str(exc)
            match = re.search(r"result ([0-9a-f]{8,16})", str(exc), re.IGNORECASE)
            if match:
                result_id = match.group(1)
                record["result_id"] = result_id
                result_dir = BASE_DIR / "data" / "results" / result_id
                wavs = sorted(p for p in result_dir.glob("*.wav") if p.is_file())
                record["audio_files"] = [str(p) for p in wavs]
                record["audio_urls"] = [f"/media/results/{result_id}/{p.name}" for p in wavs]
            print(f"  -> failed: {exc}", file=sys.stderr)
        summary["selected"].append(record)
        (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nManifest: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
