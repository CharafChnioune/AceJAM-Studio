import json
import tempfile
import unittest
from pathlib import Path

from album_quality_gate import (
    ALBUM_PAYLOAD_GATE_VERSION,
    AlbumRunDebugLogger,
    build_album_global_sonic_caption,
    evaluate_album_payload_quality,
    tag_dimension_coverage,
)


def _lyrics_for_sections(sections, lines_per_section=4):
    lines = []
    for section in sections:
        lines.append(f"[{section}]")
        for idx in range(lines_per_section):
            lines.append(f"{section} melody carries the promise home {idx}")
    return "\n".join(lines)


class AlbumQualityGateTest(unittest.TestCase):
    def test_caption_with_prompt_and_lyric_leakage_is_repaired(self):
        payload = {
            "caption": 'Track 1: "Bad Caption" BPM: 95\n[Verse]\nThis should never be in caption',
            "tag_list": ["pop", "steady groove", "piano", "clear lead vocal", "uplifting", "dynamic hook", "polished studio mix"],
            "lyrics": "[Instrumental]",
            "instrumental": True,
            "duration": 60,
        }

        report = evaluate_album_payload_quality(payload, repair=True)

        self.assertEqual(report["status"], "auto_repair")
        self.assertTrue(report["gate_passed"])
        repaired = report["repaired_payload"]
        self.assertLessEqual(len(repaired["caption"]), 512)
        self.assertNotIn("[Verse]", repaired["caption"])
        self.assertNotIn("Track 1:", repaired["caption"])
        self.assertIn("caption_rebuilt_from_tag_dimensions", report["repair_actions"])

    def test_caption_dict_fragment_in_tags_is_repaired(self):
        payload = {
            "caption": "pop, funk, groovy, {'bpm': '95'}, clean lead vocal",
            "tag_list": ["pop", "groovy", "{'bpm': '95'}", "piano", "clear lead vocal", "bright", "dynamic hook", "polished studio mix"],
            "lyrics": "[Instrumental]",
            "instrumental": True,
            "duration": 60,
        }

        report = evaluate_album_payload_quality(payload, repair=True)
        repaired = report["repaired_payload"]

        self.assertEqual(report["status"], "auto_repair")
        self.assertTrue(report["gate_passed"])
        self.assertNotIn("{", repaired["caption"])
        self.assertNotIn("{'bpm'", repaired["tag_list"])

    def test_global_caption_is_compact_album_sonic_dna_not_track_list(self):
        tracks = [{
            "tags": "hip-hop, boom-bap, dusty piano, male rap vocal, gritty mood, anthemic hook, punchy studio mix",
            "style": "warm boom-bap",
            "vibe": "dusty piano and brass",
        }]
        global_caption = build_album_global_sonic_caption(
            "Track 1: Foo\nTrack 2: Bar\nLyrics: secret prompt dump",
            tracks,
            existing="Track 1: Foo\nLyrics: leaked lines",
        )

        self.assertLessEqual(len(global_caption), 512)
        self.assertNotIn("Track 1:", global_caption)
        self.assertNotIn("Lyrics:", global_caption)
        self.assertIn("hip-hop", global_caption.lower())

    def test_tag_dimension_coverage_requires_all_album_payload_dimensions(self):
        caption = (
            "pop, steady groove, piano, clear lead vocal, uplifting mood, "
            "dynamic hook arrangement, polished studio mix"
        )
        coverage = tag_dimension_coverage(caption)

        self.assertEqual(coverage["status"], "pass")
        self.assertEqual(coverage["missing"], [])
        self.assertEqual(len(coverage["dimensions"]), 7)

    def test_long_vocal_track_with_short_nonsense_lyrics_fails_loudly(self):
        payload = {
            "caption": (
                "pop, steady groove, piano, clear lead vocal, uplifting mood, "
                "dynamic hook arrangement, polished studio mix"
            ),
            "lyrics": "[Verse]\nMorning finds the you on the floor\n[Chorus]\nThe you is here",
            "duration": 240,
            "language": "en",
            "instrumental": False,
        }

        report = evaluate_album_payload_quality(payload, repair=True)
        issue_ids = {issue["id"] for issue in report["issues"]}

        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["gate_passed"])
        self.assertIn("lyrics_under_length", issue_ids)
        self.assertIn("fallback_lyric_artifacts", issue_ids)
        self.assertIn("section_coverage_low", issue_ids)

    def test_complete_duration_fit_payload_passes(self):
        sections = ["Intro", "Verse", "Chorus", "Verse 2", "Final Chorus"]
        payload = {
            "caption": (
                "pop, steady groove, piano, clear lead vocal, uplifting mood, "
                "dynamic hook arrangement, polished studio mix"
            ),
            "lyrics": _lyrics_for_sections(sections, lines_per_section=4),
            "duration": 60,
            "language": "en",
        }

        report = evaluate_album_payload_quality(payload, repair=True)

        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["gate_passed"])
        self.assertEqual(report["tag_coverage"]["status"], "pass")
        self.assertEqual(report["caption_integrity"]["status"], "pass")
        self.assertEqual(report["lyric_duration_fit"]["status"], "pass")

    def test_markdown_bold_chorus_counts_as_hook(self):
        sections = ["Intro", "Verse", "Chorus", "Verse 2", "Final Chorus"]
        lyrics = _lyrics_for_sections(sections, lines_per_section=4).replace("[Chorus]", "**[Chorus]**")
        payload = {
            "caption": (
                "pop, steady groove, piano, clear lead vocal, uplifting mood, "
                "dynamic hook arrangement, polished studio mix"
            ),
            "lyrics": lyrics,
            "duration": 60,
            "language": "en",
        }

        report = evaluate_album_payload_quality(payload, repair=True)

        self.assertTrue(report["gate_passed"])
        self.assertGreaterEqual(report["lyric_duration_fit"]["stats"]["hook_count"], 1)
        self.assertNotIn("hook_missing", {issue["id"] for issue in report["issues"]})

    def test_song_model_metadata_in_lyrics_fails(self):
        payload = {
            "caption": (
                "pop, steady groove, piano, clear lead vocal, uplifting mood, "
                "dynamic hook arrangement, polished studio mix"
            ),
            "lyrics": _lyrics_for_sections(["Intro", "Verse", "Chorus", "Verse 2", "Final Chorus"], lines_per_section=4)
            + "\nSong Model: acestep-v15-turbo",
            "duration": 60,
            "language": "en",
        }

        report = evaluate_album_payload_quality(payload, repair=True)
        issue_ids = {issue["id"] for issue in report["issues"]}

        self.assertEqual(report["status"], "fail")
        self.assertIn("lyric_meta_leakage", issue_ids)

    def test_bold_ace_step_metadata_in_lyrics_fails(self):
        payload = {
            "caption": (
                "pop, steady groove, piano, clear lead vocal, uplifting mood, "
                "dynamic hook arrangement, polished studio mix"
            ),
            "lyrics": _lyrics_for_sections(["Intro", "Verse", "Chorus", "Verse 2", "Final Chorus"], lines_per_section=4)
            + "\n**ACE-Step Metadata:**\n- **Song Model:** acestep-v15-turbo",
            "duration": 60,
            "language": "en",
        }

        report = evaluate_album_payload_quality(payload, repair=True)
        issue_ids = {issue["id"] for issue in report["issues"]}

        self.assertEqual(report["status"], "fail")
        self.assertIn("lyric_meta_leakage", issue_ids)

    def test_ace_step_timing_block_in_lyrics_fails(self):
        payload = {
            "caption": (
                "pop, steady groove, piano, clear lead vocal, uplifting mood, "
                "dynamic hook arrangement, polished studio mix"
            ),
            "lyrics": _lyrics_for_sections(["Intro", "Verse", "Chorus", "Verse 2", "Final Chorus"], lines_per_section=4)
            + "\n[ACE-Step]\nTag: [Chorus]\nStart: 00:30\nEnd: 00:46\nVocal Role: lyrics",
            "duration": 60,
            "language": "en",
        }

        report = evaluate_album_payload_quality(payload, repair=True)
        issue_ids = {issue["id"] for issue in report["issues"]}

        self.assertEqual(report["status"], "fail")
        self.assertIn("lyric_meta_leakage", issue_ids)

    def test_bracketed_ace_step_metadata_tail_in_lyrics_fails(self):
        payload = {
            "caption": (
                "pop, steady groove, piano, clear lead vocal, uplifting mood, "
                "dynamic hook arrangement, polished studio mix"
            ),
            "lyrics": _lyrics_for_sections(["Intro", "Verse", "Chorus", "Verse 2", "Final Chorus"], lines_per_section=4)
            + "\n[ACE-Step metadata]\ntag_list: pop, funk, radio ready",
            "duration": 60,
            "language": "en",
        }

        report = evaluate_album_payload_quality(payload, repair=True)
        issue_ids = {issue["id"] for issue in report["issues"]}

        self.assertEqual(report["status"], "fail")
        self.assertIn("lyric_meta_leakage", issue_ids)

    def test_bracketed_contract_metadata_in_lyrics_fails(self):
        payload = {
            "caption": (
                "pop, steady groove, piano, clear lead vocal, uplifting mood, "
                "dynamic hook arrangement, polished studio mix"
            ),
            "lyrics": _lyrics_for_sections(["Intro", "Verse", "Chorus", "Verse 2", "Final Chorus"], lines_per_section=4)
            + "\n[Producer Credit: Studio House]\n[Locked Title: Neon Bakery Lights]\n[Duration: 60.0 seconds]",
            "duration": 60,
            "language": "en",
        }

        report = evaluate_album_payload_quality(payload, repair=True)
        issue_ids = {issue["id"] for issue in report["issues"]}

        self.assertEqual(report["status"], "fail")
        self.assertIn("lyric_meta_leakage", issue_ids)

    def test_hook_repetition_warns_without_blocking_render(self):
        hook_lines = [f"Hook returns {idx}" for idx in range(20)]
        lyrics = "\n".join(
            ["[Intro]"]
            + [f"Opening image starts {idx}" for idx in range(4)]
            + ["[Verse 1]"]
            + [f"Clear scene moves {idx}" for idx in range(14)]
            + ["[Pre-Chorus]"]
            + hook_lines
            + ["[Chorus]"]
            + hook_lines
            + ["[Verse 2]"]
            + [f"Second scene lands {idx}" for idx in range(14)]
            + ["[Pre-Chorus]"]
            + hook_lines
            + ["[Chorus]"]
            + hook_lines
            + ["[Bridge]"]
            + [f"Bridge opens door {idx}" for idx in range(6)]
            + ["[Breakdown]"]
            + [f"Breakdown leaves space {idx}" for idx in range(4)]
            + ["[Final Chorus]"]
            + hook_lines
            + ["[Outro]", "Last image lands"]
        )
        payload = {
            "caption": (
                "pop, steady groove, piano, clear lead vocal, uplifting mood, "
                "dynamic hook arrangement, polished studio mix"
            ),
            "lyrics": lyrics,
            "duration": 180,
            "language": "en",
        }

        report = evaluate_album_payload_quality(payload, repair=True)
        issue_ids = {issue["id"] for issue in report["issues"]}

        self.assertIn("lyric_repetition_warning", issue_ids)
        self.assertTrue(report["gate_passed"])
        self.assertNotEqual(report["status"], "fail")

    def test_low_unique_line_ratio_blocks_mechanical_lyrics(self):
        lines = (
            ["[Intro]"]
            + ["Bakery lights come back tonight"] * 8
            + ["[Verse]"]
            + ["Warm windows shine across the street"] * 8
            + ["[Chorus]"]
            + ["We gather where the ovens glow"] * 8
            + ["[Verse 2]"]
            + ["Warm windows shine across the street"] * 8
            + ["[Final Chorus]"]
            + ["We gather where the ovens glow"] * 8
        )
        payload = {
            "caption": (
                "pop, steady groove, piano, clear lead vocal, uplifting mood, "
                "dynamic hook arrangement, polished studio mix"
            ),
            "lyrics": "\n".join(lines),
            "duration": 60,
            "language": "en",
        }

        report = evaluate_album_payload_quality(payload, repair=True)
        issue_ids = {issue["id"] for issue in report["issues"]}

        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["gate_passed"])
        self.assertIn("too_many_repeated_lines", issue_ids)

    def test_debug_logger_writes_job_scoped_json_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = AlbumRunDebugLogger(Path(tmp), "job/with spaces")
            json_path = Path(logger.write_json("01_request.json", {"ok": True}))
            jsonl_path = Path(logger.append_jsonl("05_generation_payloads.jsonl", {"phase": "pre_gate"}))

            self.assertTrue(str(logger.root).endswith("job_with_spaces"))
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["ok"], True)
            record = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
            self.assertEqual(record["phase"], "pre_gate")
            self.assertEqual(ALBUM_PAYLOAD_GATE_VERSION, "album-payload-quality-gate-2026-04-29")


if __name__ == "__main__":
    unittest.main()
