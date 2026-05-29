import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "app" / "vendor" / "ACE-Step-1.5"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeLoraLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = torch.nn.Linear(2, 2, bias=False)
        self.lora_A = torch.nn.ModuleDict({"main": torch.nn.Linear(2, 1, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"main": torch.nn.Linear(1, 2, bias=False)})
        self.scaling = {"main": 0.5}
        self.lora_bias = {"main": False}
        self.active_adapters = ["main"]
        self.disable_adapters = False
        with torch.no_grad():
            self.base_layer.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            self.lora_A["main"].weight.copy_(torch.tensor([[2.0, 3.0]]))
            self.lora_B["main"].weight.copy_(torch.tensor([[5.0], [7.0]]))

    def get_base_layer(self):
        return self.base_layer

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        return self.lora_B[adapter].weight @ self.lora_A[adapter].weight * self.scaling[adapter]


class FakeDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = FakeLoraLinear()
        self.other = torch.nn.Linear(2, 2, bias=False)


class FakePeftWrapper:
    def __init__(self, decoder):
        self.decoder = decoder

    def get_base_model(self):
        return self.decoder


class MlxLoraSyncTests(unittest.TestCase):
    def test_converter_bakes_active_lora_delta_into_mlx_weights(self):
        module = _load_module(VENDOR / "acestep" / "models" / "mlx" / "dit_convert.py", "acejam_test_dit_convert")
        decoder = FakeDecoder()

        state = module._effective_decoder_state_dict(FakePeftWrapper(decoder))

        expected_delta = torch.tensor([[5.0, 7.5], [7.0, 10.5]])
        self.assertTrue(torch.allclose(state["proj.weight"], decoder.proj.base_layer.weight + expected_delta))
        self.assertIn("other.weight", state)
        self.assertFalse(any(".lora_" in key or ".base_layer." in key for key in state))

    def test_converter_exports_base_weight_when_lora_disabled_or_zero_scale(self):
        module = _load_module(VENDOR / "acestep" / "models" / "mlx" / "dit_convert.py", "acejam_test_dit_convert_base")
        decoder = FakeDecoder()
        decoder.proj.disable_adapters = True
        disabled_state = module._effective_decoder_state_dict(FakePeftWrapper(decoder))
        self.assertTrue(torch.allclose(disabled_state["proj.weight"], decoder.proj.base_layer.weight))

        decoder.proj.disable_adapters = False
        decoder.proj.scaling["main"] = 0.0
        zero_state = module._effective_decoder_state_dict(FakePeftWrapper(decoder))
        self.assertTrue(torch.allclose(zero_state["proj.weight"], decoder.proj.base_layer.weight))

    def test_mlx_dit_sync_refreshes_active_decoder_after_lora_change(self):
        package_names = [
            "acestep",
            "acestep.core",
            "acestep.core.generation",
            "acestep.core.generation.handler",
        ]
        saved = {name: sys.modules.get(name) for name in package_names}
        for name in package_names:
            pkg = types.ModuleType(name)
            pkg.__path__ = []
            sys.modules[name] = pkg
        try:
            module = _load_module(
                VENDOR / "acestep" / "core" / "generation" / "handler" / "mlx_dit_init.py",
                "acestep.core.generation.handler.mlx_dit_init",
            )

            class Host(module.MlxDitInitMixin):
                pass

            host = Host()
            host.model = types.SimpleNamespace(decoder=object())
            host.mlx_decoder = Mock()
            host.use_mlx_dit = True
            host.lora_loaded = True
            host.use_lora = True
            host._adapter_type = "lora"
            host._lora_active_adapter = "main"
            host._active_loras = {"main": 1.0}
            fake_convert = types.ModuleType("acestep.models.mlx.dit_convert")
            fake_convert.convert_and_load = Mock()

            with patch.dict(sys.modules, {"acestep.models.mlx.dit_convert": fake_convert}):
                result = host._sync_mlx_dit_weights_from_torch(reason="unit", force=True)

            self.assertTrue(result["synced"])
            fake_convert.convert_and_load.assert_called_once_with(host.model, host.mlx_decoder)
            host.mlx_decoder.materialize_static_buffers.assert_called_once_with()
        finally:
            for name, original in saved.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original

    def test_vendor_patcher_contains_mlx_lora_sync_patch(self):
        patcher = (ROOT / "app" / "patch_ace_step_vendor.py").read_text(encoding="utf-8")
        self.assertIn("patch_mlx_effective_lora_sync", patcher)
        self.assertIn("_effective_decoder_state_dict", patcher)
        self.assertIn("_sync_mlx_dit_weights_from_torch", patcher)


if __name__ == "__main__":
    unittest.main()
