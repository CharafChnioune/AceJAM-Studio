from pathlib import Path
import unittest


class LauncherScriptTest(unittest.TestCase):
    def test_install_script_downloads_all_acestep_weights(self):
        root = Path(__file__).resolve().parents[2]
        install_js = (root / "install.js").read_text(encoding="utf-8")

        self.assertIn('path: "app"', install_js)
        self.assertIn('"python download_models.py --all"', install_js)
        self.assertIn('PYTHONUNBUFFERED: "1"', install_js)

    def test_start_script_binds_localhost_and_captures_url(self):
        root = Path(__file__).resolve().parents[2]
        start_js = (root / "start.js").read_text(encoding="utf-8")

        self.assertIn('GRADIO_SERVER_NAME: "127.0.0.1"', start_js)
        self.assertIn('event: "/(http:\\\\/\\\\/[0-9.:]+)/"', start_js)
        self.assertIn('url: "{{input.event[1]}}"', start_js)
        self.assertIn('bind_url: "{{input.event[1]}}"', start_js)

    def test_song_intent_builder_is_schema_driven(self):
        root = Path(__file__).resolve().parents[2]
        index_html = (root / "app" / "index.html").read_text(encoding="utf-8")

        self.assertNotIn("INTENT_SUBGENRES", index_html)
        self.assertIn("song_intent_schema", index_html)
        self.assertIn("renderSongIntentBuilder", index_html)
        self.assertIn("data-intent-tab", index_html)
        self.assertIn("genre_modules", index_html)
        self.assertIn("model_strategies", index_html)
        self.assertIn("negative_control", index_html)


if __name__ == "__main__":
    unittest.main()
