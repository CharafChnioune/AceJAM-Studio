from pathlib import Path
import unittest


class LauncherScriptTest(unittest.TestCase):
    def test_install_script_downloads_all_acestep_weights(self):
        root = Path(__file__).resolve().parents[2]
        install_js = (root / "install.js").read_text(encoding="utf-8")

        self.assertIn('path: "app"', install_js)
        self.assertIn('"python download_models.py --all"', install_js)
        self.assertIn('"python install_mflux.py"', install_js)
        self.assertIn('"python install_mlx_video.py"', install_js)
        self.assertIn('PYTHONUNBUFFERED: "1"', install_js)

    def test_mflux_runtime_has_dedicated_requirements(self):
        root = Path(__file__).resolve().parents[2]
        core_requirements = (root / "app" / "requirements.txt").read_text(encoding="utf-8")
        mflux_requirements = (root / "app" / "requirements-mflux.txt").read_text(encoding="utf-8")
        manager = (root / "app" / "mflux_manager.py").read_text(encoding="utf-8")

        self.assertNotIn("mflux>=", core_requirements)
        self.assertIn("mflux>=0.17.5,<0.18", mflux_requirements)
        self.assertIn("MFLUX_ENV_DIR", manager)
        self.assertIn("_command_path", manager)

    def test_torchao_is_pinned_for_torch_291_runtime(self):
        root = Path(__file__).resolve().parents[2]
        core_requirements = (root / "app" / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("torchao==0.15.0", core_requirements)
        self.assertNotIn("torchao>=0.16.0,<0.17.0", core_requirements)

    def test_vendor_patch_skips_bitsandbytes_warning_on_non_cuda(self):
        root = Path(__file__).resolve().parents[2]
        patcher = (root / "app" / "patch_ace_step_vendor.py").read_text(encoding="utf-8")

        self.assertIn("patch_bitsandbytes_non_cuda_warning", patcher)
        self.assertIn("if torch.cuda.is_available()", patcher)

    def test_training_bootstrap_preserves_cli_args_and_skips_prompt(self):
        root = Path(__file__).resolve().parents[2]
        bootstrap = (root / "app" / "_acejam_train_bootstrap.py").read_text(encoding="utf-8")

        self.assertIn("sys.argv = [str(_target)] + _user_args", bootstrap)
        self.assertIn("_config_panel.confirm_start = lambda skip=False: True", bootstrap)

    def test_start_script_binds_localhost_and_captures_url(self):
        root = Path(__file__).resolve().parents[2]
        start_js = (root / "start.js").read_text(encoding="utf-8")

        self.assertIn('GRADIO_SERVER_NAME: "127.0.0.1"', start_js)
        self.assertIn('event: "/(http:\\\\/\\\\/[0-9.:]+)/"', start_js)
        self.assertIn('url: "{{input.event[1]}}"', start_js)
        self.assertIn('bind_url: "{{input.event[1]}}"', start_js)

    # Removed: test_song_intent_builder_is_schema_driven asserted on the
    # legacy Python SPA (app/index.html) that was deleted in v0.2 when the
    # React + shadcn wizard UI took over. Equivalent intent builder now lives
    # in app/web/src/wizards/CustomWizard.tsx and is covered by TS type
    # checks plus Playwright smoke tests.
    pass


if __name__ == "__main__":
    unittest.main()
