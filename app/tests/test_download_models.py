import unittest

import download_models


class DownloadModelsTest(unittest.TestCase):
    def test_all_download_list_covers_supported_models_and_shared_components(self):
        models = download_models.default_download_models()

        self.assertIn("acestep-v15-turbo", models)
        self.assertIn("acestep-v15-xl-sft", models)
        self.assertIn("acestep-5Hz-lm-4B", models)
        self.assertIn("vae", models)
        self.assertIn("Qwen3-Embedding-0.6B", models)
        self.assertNotIn("auto", models)
        self.assertNotIn("none", models)
        self.assertNotIn("acestep-v15-turbo-rl", models)


if __name__ == "__main__":
    unittest.main()
