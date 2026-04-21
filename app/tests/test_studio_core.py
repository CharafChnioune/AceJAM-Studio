import unittest

from studio_core import (
    build_task_instruction,
    ensure_task_supported,
    normalize_task_type,
    normalize_track_names,
    parse_timesteps,
    safe_id,
    supported_tasks_for_model,
)


class StudioCoreTest(unittest.TestCase):
    def test_task_aliases(self):
        self.assertEqual(normalize_task_type("simple"), "text2music")
        self.assertEqual(normalize_task_type("custom"), "text2music")
        self.assertEqual(normalize_task_type("remix"), "cover")

    def test_model_capabilities(self):
        self.assertIn("complete", supported_tasks_for_model("acestep-v15-base"))
        self.assertNotIn("complete", supported_tasks_for_model("acestep-v15-turbo"))
        with self.assertRaises(ValueError):
            ensure_task_supported("acestep-v15-xl-turbo", "extract")
        ensure_task_supported("acestep-v15-xl-base", "extract")

    def test_instruction_templates(self):
        self.assertIn("vocals", build_task_instruction("extract", "vocals"))
        self.assertIn("drums", build_task_instruction("lego", ["drums"]))
        self.assertIn("drums, bass", build_task_instruction("complete", ["drums", "bass"]))

    def test_track_normalization(self):
        self.assertEqual(normalize_track_names("backing vocals, drums, nope"), ["backing_vocals", "drums"])

    def test_timesteps(self):
        self.assertEqual(parse_timesteps("1, 0.5, 0"), [1.0, 0.5, 0.0])
        self.assertIsNone(parse_timesteps(""))

    def test_safe_id(self):
        self.assertEqual(safe_id("abc123_DEF-9"), "abc123_DEF-9")
        with self.assertRaises(ValueError):
            safe_id("../bad")


if __name__ == "__main__":
    unittest.main()
