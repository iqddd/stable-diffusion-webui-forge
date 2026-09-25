import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.argv = [sys.argv[0]]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modules_forge" / "packages"))

from backend.diffusion_engine.krea import Krea2, configure_filter_bypass
from backend.nn.krea import TextFusionTransformer


DELTA = torch.tensor([[0.0] * 8 + [-0.51171875, -0.890625, -0.609375, 0.0]])


def make_transformer():
    model = TextFusionTransformer(num_txt_layers=12, txt_dim=4, heads=1, multiplier=1)
    model.layerwise_blocks = nn.ModuleList()
    model.refiner_blocks = nn.ModuleList()
    with torch.no_grad():
        model.projector.weight.copy_(torch.arange(12, dtype=torch.float32).view(1, 12) / 32)
    return model


def projector_input(x):
    return x.permute(0, 1, 3, 2)


@pytest.mark.parametrize("strength", [0.0, 1.0, 3.0])
def test_bypass_matches_reference_weight_delta(strength):
    model = make_transformer()
    x = torch.arange(2 * 3 * 12 * 4, dtype=torch.float32).reshape(2, 3, 12, 4) / 128
    original_weight = model.projector.weight.detach().clone()
    buffer_id = id(model.filter_bypass_strength)

    model.set_filter_bypass_strength(strength)
    actual = model(x)
    expected = F.linear(projector_input(x), original_weight + strength * DELTA).squeeze(-1)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(model.projector.weight, original_weight)
    assert id(model.filter_bypass_strength) == buffer_id


def test_bypass_adds_to_quantized_projector_without_patching_its_weight():
    class QuantizedProjector(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.register_buffer("qweight", (weight * 128).round().to(torch.int8))

        def forward(self, x):
            return F.linear(x, self.qweight.float() / 128)

    model = make_transformer()
    model.projector = QuantizedProjector(model.projector.weight.detach())
    x = torch.randn(1, 2, 12, 4)
    original_qweight = model.projector.qweight.clone()
    model.set_filter_bypass_strength(1.5)

    actual = model(x)
    expected = F.linear(projector_input(x), original_qweight.float() / 128 + 1.5 * DELTA).squeeze(-1)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(model.projector.qweight, original_qweight)


def test_bypass_runs_with_bfloat16_model():
    model = make_transformer().to(dtype=torch.bfloat16)
    x = torch.randn(1, 2, 12, 4, dtype=torch.bfloat16)

    model.set_filter_bypass_strength(0.0)
    baseline = model(x)
    expected_baseline = model.projector(projector_input(x)).squeeze(-1)
    torch.testing.assert_close(baseline, expected_baseline)

    model.set_filter_bypass_strength(1.0)
    assert not torch.equal(model(x), baseline)


def make_krea_engine():
    engine = object.__new__(Krea2)
    transformer = make_transformer()
    engine.forge_objects = SimpleNamespace(unet=SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(txtfusion=transformer))))
    return engine, transformer


def test_setting_only_updates_the_loaded_krea_model():
    first, first_transformer = make_krea_engine()
    second, second_transformer = make_krea_engine()
    other_model = SimpleNamespace()

    configure_filter_bypass(first, 2.0)
    configure_filter_bypass(other_model, 3.0)
    configure_filter_bypass(second, 3.0)

    assert first_transformer.filter_bypass_strength.item() == 2.0
    assert second_transformer.filter_bypass_strength.item() == 3.0

    configure_filter_bypass(first, 0.0)
    assert first_transformer.filter_bypass_strength.item() == 0.0
    assert second_transformer.filter_bypass_strength.item() == 3.0


def test_changing_strength_does_not_recompile():
    torch._dynamo.reset()
    model = make_transformer()
    x = torch.randn(1, 2, 12, 4)
    graph_count = 0

    def counting_backend(graph, example_inputs):
        nonlocal graph_count
        graph_count += 1
        return graph.forward

    compiled = torch.compile(model, backend=counting_backend)
    baseline = compiled(x)
    compiled_graphs = graph_count
    assert compiled_graphs > 0

    model.set_filter_bypass_strength(1.0)
    with_bypass = compiled(x)
    model.set_filter_bypass_strength(3.0)
    stronger_bypass = compiled(x)

    assert graph_count == compiled_graphs
    assert not torch.equal(baseline, with_bypass)
    assert not torch.equal(with_bypass, stronger_bypass)


def test_inference_loaded_model_updates_strength_without_recompiling():
    torch._dynamo.reset()
    # forge_loader constructs the model inside inference_mode, but
    # process_images configures the strength outside that context.
    with torch.inference_mode():
        engine, model = make_krea_engine()
        x = torch.randn(1, 2, 12, 4)

    buffer = model.filter_bypass_strength
    assert buffer.is_inference()
    assert not torch.is_inference_mode_enabled()
    graph_count = 0

    def counting_backend(graph, example_inputs):
        nonlocal graph_count
        graph_count += 1
        return graph.forward

    compiled = torch.compile(model, backend=counting_backend, fullgraph=True)
    outputs = []
    for strength in (0.0, 1.0, 3.0, 0.0):
        configure_filter_bypass(engine, strength)
        assert model.filter_bypass_strength is buffer
        assert buffer.item() == strength
        assert not torch.is_inference_mode_enabled()
        with torch.inference_mode():
            outputs.append(compiled(x).clone())

    assert graph_count == 1
    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[1], outputs[2])
    torch.testing.assert_close(outputs[0], outputs[3], rtol=0, atol=0)
