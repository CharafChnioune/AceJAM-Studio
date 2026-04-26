from pathlib import Path
import unittest


class LauncherScriptTest(unittest.TestCase):
    def test_start_script_binds_lan_and_captures_url(self):
        root = Path(__file__).resolve().parents[2]
        start_js = (root / "start.js").read_text(encoding="utf-8")

        self.assertIn('GRADIO_SERVER_NAME: "0.0.0.0"', start_js)
        self.assertIn('event: "/(http:\\\\/\\\\/[0-9.:]+)/"', start_js)
        self.assertIn('bind_url: "{{input.event[1]}}"', start_js)
        self.assertIn("input.event[1].replace('0.0.0.0', '127.0.0.1')", start_js)


if __name__ == "__main__":
    unittest.main()
