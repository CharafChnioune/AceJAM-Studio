from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import install_mflux
import install_mlx_video
import sync_ace_step_vendor


class RuntimeInstallerTests(unittest.TestCase):
    def test_mflux_runtime_status_reports_new_version_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python = root / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("", encoding="utf-8")

            with patch.object(install_mflux, "MFLUX_ENV_DIR", root), \
                 patch.object(install_mflux, "_mflux_python", return_value=python), \
                 patch.object(install_mflux, "_python_status", return_value={"exists": True, "version": "3.10.17", "ok": True, "reason": ""}), \
                 patch.object(install_mflux, "_package_status", side_effect=[
                     {"available": True, "version": "0.18.0", "reason": ""},
                     {"available": True, "version": "0.31.2", "reason": ""},
                     {"available": True, "version": "5.10.1", "reason": ""},
                 ]):
                status = install_mflux.runtime_status()

        self.assertEqual(status["version_range"], ">=0.18,<0.19")
        self.assertTrue(status["ready"])
        self.assertEqual(status["packages"]["mflux"]["version"], "0.18.0")

    def test_mlx_video_runtime_status_reports_vendor_and_patch_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python = root / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("", encoding="utf-8")

            with patch.object(install_mlx_video, "_video_python", return_value=python), \
                 patch.object(
                     install_mlx_video,
                     "_vendor_status",
                     return_value={
                         "commit": "87db56a",
                         "dirty_files": [],
                         "target_ref": install_mlx_video.MLX_VIDEO_TARGET_REF,
                         "target_ref_short": install_mlx_video.MLX_VIDEO_TARGET_REF[:7],
                         "matches_target_ref": True,
                     },
                 ), \
                 patch.object(install_mlx_video, "_python_status", return_value={"exists": True, "version": "3.11.13", "ok": True, "reason": ""}), \
                 patch.object(install_mlx_video, "_package_status", side_effect=[
                     {"available": True, "version": "0.31.2", "reason": ""},
                     {"available": True, "version": "0.0.1", "reason": ""},
                     {"available": True, "version": "0.6.2", "reason": ""},
                     {"available": True, "version": "0.1.9", "reason": ""},
                 ]), \
                 patch.object(install_mlx_video, "_command_help", side_effect=[
                     {"command": ["python", "-m", "ltx"], "help_ok": True, "reason": ""},
                     {"command": ["python", "-m", "wan"], "help_ok": True, "reason": ""},
                 ]), \
                 patch.object(install_mlx_video, "_vae_patch_already_present", return_value=True), \
                 patch.object(install_mlx_video, "_sampling_patch_already_present", return_value=True), \
                 patch.object(install_mlx_video, "_end_image_patch_already_present", return_value=True):
                status = install_mlx_video.runtime_status()

        self.assertTrue(status["ready"])
        self.assertEqual(status["vendor"]["commit"], "87db56a")
        self.assertEqual(status["vendor"]["target_ref"], install_mlx_video.MLX_VIDEO_TARGET_REF)
        self.assertTrue(status["patch_status"]["pr23_ltx_i2v_end_frame"])

    def test_sync_vendor_status_separates_known_patches_from_unknown_drift(self):
        dirty = "\n".join(
            [
                " M acestep/training_v2/cli/args.py",
                " M acestep/training/trainer.py",
                " M unexpected/file.py",
            ]
        )
        with patch.object(sync_ace_step_vendor, "run", return_value=None), \
             patch.object(
                 sync_ace_step_vendor,
                 "capture",
                 side_effect=[
                     "dce621408bee8c31b4fcf4811682eb9359e1bc94",
                     "dce621408bee8c31b4fcf4811682eb9359e1bc94",
                     dirty,
                 ],
             ), \
             patch.object(sync_ace_step_vendor, "VENDOR_DIR", Path("/tmp/vendor")):
            with patch.object(Path, "is_dir", return_value=True):
                status = sync_ace_step_vendor.vendor_status()

        self.assertTrue(status["pinned_matches_upstream"])
        self.assertIn("acestep/training_v2/cli/args.py", status["known_patch_files"])
        self.assertIn("unexpected/file.py", status["unknown_drift_files"])

    def test_vendor_sync_scripts_avoid_force_resets_and_track_pins(self):
        root = Path(__file__).resolve().parents[2]
        ace_sync = (root / "app" / "sync_ace_step_vendor.py").read_text(encoding="utf-8")
        mlx_video_sync = (root / "app" / "install_mlx_video.py").read_text(encoding="utf-8")

        self.assertNotIn("reset --hard", ace_sync)
        self.assertNotIn('"checkout", "--force"', ace_sync)
        self.assertIn('ACE_STEP_VENDOR_RELEASE = "v0.1.8"', ace_sync)
        self.assertIn(f'MLX_VIDEO_TARGET_REF = "{install_mlx_video.MLX_VIDEO_TARGET_REF}"', mlx_video_sync)
        self.assertNotIn('"checkout", "main"', mlx_video_sync)
        self.assertNotIn('"pull", "--ff-only"', mlx_video_sync)


if __name__ == "__main__":
    unittest.main()
