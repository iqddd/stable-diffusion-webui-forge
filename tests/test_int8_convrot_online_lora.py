import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules_forge" / "packages"))

# Import operations first; it owns the intentional operations_mixed_precision cycle.
import backend.operations  # noqa: E402,F401
from backend.operations_mixed_precision import (  # noqa: E402
    TRITON_AVAILABLE,
    _prepare_int8_convrot_online_lora,
    mixed_precision_ops,
    prepare_int8_convrot_online_lora_for_compile,
)
from backend.operations_triton import triton_int8_linear, triton_int8_linear_per_row  # noqa: E402
from backend.patcher.base import WeightPatch, reset_weight_functions  # noqa: E402
from backend.quant_ops import QuantizedTensor, TensorWiseINT8Layout  # noqa: E402
from backend.quant_rotation import build_hadamard, rotate_activation  # noqa: E402
from modules_forge.packages.comfy.weight_adapter.lora import LoRAAdapter  # noqa: E402


@unittest.skipUnless(torch.cuda.is_available() and TRITON_AVAILABLE, "CUDA INT8 Triton override is required")
class Int8ConvRotOnlineLoRATest(unittest.TestCase):
    def make_layer(self, group_size=256, per_row=True, bias=True):
        in_features = 256
        out_features = 192
        qweight = torch.randint(-127, 128, (out_features, in_features), dtype=torch.int8)
        scale = torch.full((out_features, 1) if per_row else (), 0.001, dtype=torch.float32)
        config = torch.tensor(
            list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group_size}).encode()),
            dtype=torch.uint8,
        )
        state_dict = {"weight": qweight, "weight_scale": scale, "comfy_quant": config}
        if bias:
            state_dict["bias"] = torch.randn(out_features, dtype=torch.bfloat16)

        ops = mixed_precision_ops(compute_dtype=torch.bfloat16)
        layer = ops.Linear(in_features, out_features, bias=bias, device=torch.device("cuda"))
        layer.load_state_dict(state_dict, strict=True)
        layer.parameters_manual_cast = True
        return layer, qweight, scale

    def add_loras(self, layer, specs):
        patches = []
        adapters = []
        for rank, strength, alpha in specs:
            down = torch.randn(rank, layer.in_features, dtype=torch.float32) * 0.02
            up = torch.randn(layer.out_features, rank, dtype=torch.float32) * 0.02
            adapter = LoRAAdapter(set(), (up, down, alpha, None, None, None))
            adapters.append(adapter)
            patches.append((strength, adapter, 1.0, None, None, True))

        patch_dict = {"weight": patches}
        layer.weight_function = [WeightPatch("weight", patch_dict, online=True)]
        layer.bias_function = []
        return patch_dict, adapters

    def reference(self, x, layer, qweight, scale, patches):
        group_size = layer.weight.params.convrot_groupsize
        H = build_hadamard(group_size, device=x.device, dtype=x.dtype)
        x_rot = rotate_activation(x, H, group_size=group_size)
        bias = layer.bias
        if layer._per_row:
            result = triton_int8_linear_per_row(x_rot, qweight.cuda(), scale.cuda(), bias, torch.bfloat16)
        else:
            result = triton_int8_linear(x_rot, qweight.cuda(), scale.cuda(), bias, torch.bfloat16)

        for strength, adapter, *_ in patches["weight"]:
            up, down, alpha, _, _, _ = adapter.weights
            gamma = strength * (alpha / down.shape[0] if alpha is not None else 1.0)
            hidden = torch.nn.functional.linear(x, down.cuda().bfloat16())
            result = result + gamma * torch.nn.functional.linear(hidden, up.cuda().bfloat16())
        return result

    def test_additive_loras_for_2d_and_3d_inputs(self):
        for group_size, per_row in ((4, False), (16, True), (256, True)):
            with self.subTest(group_size=group_size, per_row=per_row):
                layer, qweight, scale = self.make_layer(group_size=group_size, per_row=per_row)
                patches, _ = self.add_loras(layer, ((8, 0.7, 4.0), (16, -0.25, 16.0), (4, 0.0, None)))

                for shape in ((5, layer.in_features), (2, 3, layer.in_features)):
                    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                    actual = layer(x)
                    expected = self.reference(x, layer, qweight, scale, patches)
                    error = (actual.float() - expected.float()).square().mean().sqrt()
                    relative_rmse = error / expected.float().square().mean().sqrt().clamp_min(1e-12)
                    self.assertLess(relative_rmse.item(), 0.01)

    def test_fast_path_does_not_dequantize_requantize_or_materialize_delta(self):
        layer, _, _ = self.make_layer()
        self.add_loras(layer, ((8, 1.0, 8.0),))
        x = torch.randn(4, layer.in_features, device="cuda", dtype=torch.bfloat16)

        with (
            mock.patch.object(QuantizedTensor, "dequantize", side_effect=AssertionError("dequantize called")),
            mock.patch.object(TensorWiseINT8Layout, "quantize", side_effect=AssertionError("requantize called")),
            mock.patch.object(LoRAAdapter, "calculate_weight", side_effect=AssertionError("full delta materialized")),
        ):
            output = layer(x)
        self.assertEqual(output.shape, (4, layer.out_features))

    def test_inference_tensors_with_torch_compile(self):
        layer, _, _ = self.make_layer()
        with torch.inference_mode():
            self.add_loras(layer, ((8, 1.0, 8.0), (16, -0.25, 8.0)))

        x = torch.randn(2, 3, layer.in_features, device="cuda", dtype=torch.bfloat16)
        prepare_int8_convrot_online_lora_for_compile(layer, x.float(), compute_dtype=torch.bfloat16)
        compiled = torch.compile(layer, backend="inductor", dynamic=False, fullgraph=False)
        with torch.inference_mode():
            output = compiled(x)
            output_again = compiled(x)
        self.assertEqual(output.shape, (2, 3, layer.out_features))
        self.assertTrue(torch.isfinite(output_again).all())

    def test_matrix_cache_reuse_strength_update_and_cleanup(self):
        layer, _, _ = self.make_layer()
        patches, _ = self.add_loras(layer, ((8, 0.5, 8.0),))
        # Low-VRAM patch wrappers are not marked online; tuple metadata is authoritative.
        layer.weight_function[0].online = False
        x = torch.randn(2, layer.in_features, device="cuda", dtype=torch.bfloat16)

        layer(x)
        cached_a = layer._forge_int8_convrot_lora_A
        cached_b = layer._forge_int8_convrot_lora_B
        old_scales = layer._forge_int8_convrot_lora_scales.clone()

        patch = patches["weight"][0]
        patches["weight"][0] = (1.25, *patch[1:])
        layer(x)
        self.assertIs(layer._forge_int8_convrot_lora_A, cached_a)
        self.assertIs(layer._forge_int8_convrot_lora_B, cached_b)
        self.assertFalse(torch.equal(layer._forge_int8_convrot_lora_scales, old_scales))

        self.add_loras(layer, ((8, 1.25, 8.0),))
        layer(x)
        self.assertIsNot(layer._forge_int8_convrot_lora_A, cached_a)
        self.assertIsNot(layer._forge_int8_convrot_lora_B, cached_b)

        reset_weight_functions(layer, wipe=True)
        self.assertFalse(any(name.startswith("_forge_int8_convrot_lora_") for name in vars(layer)))

    def test_unsupported_adapter_uses_fallback(self):
        layer, _, _ = self.make_layer()
        patches, _ = self.add_loras(layer, ((8, 1.0, 8.0),))
        patch = patches["weight"][0]
        patches["weight"][0] = (*patch[:4], lambda delta: delta, patch[5])
        x = torch.randn(2, layer.in_features, device="cuda", dtype=torch.bfloat16)

        *_, supported = _prepare_int8_convrot_online_lora(layer, x, 256)
        self.assertFalse(supported)
        output = layer(x)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(len(patches["weight"]), 1)


if __name__ == "__main__":
    unittest.main()
