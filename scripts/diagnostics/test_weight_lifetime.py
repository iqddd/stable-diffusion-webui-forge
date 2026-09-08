"""CUDA regression checks for loading references and offline LoRA restoration."""
import gc
import sys
import weakref
import tempfile
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'modules_forge' / 'packages')]

import torch
import backend.operations
from backend.operations_mixed_precision import mixed_precision_ops
from backend.patcher.base import ModelPatcher
from backend import quant_ops
from modules_forge.packages.comfy.weight_adapter.lora import LoRAAdapter


QuantizedTensor: Any = getattr(quant_ops, 'QuantizedTensor')


def check_loader():
    from backend import utils
    from safetensors.torch import save_file

    values = {'weight': torch.arange(1024, dtype=torch.int32), 'scale': torch.tensor(0.25)}
    with tempfile.TemporaryDirectory(prefix='forge-loader-test-') as directory:
        path = str(Path(directory) / 'test.safetensors')
        save_file(values, path, metadata={'fixture': 'lifetime'})
        for platform, disabled, expected in [('nt', False, 'pread'), ('posix', False, None), ('posix', True, 'pread')]:
            with patch.object(utils, 'os', SimpleNamespace(name=platform)), patch.object(utils, 'DISABLE_MMAP', disabled), patch.object(utils.safetensors, 'safe_open', wraps=utils.safetensors.safe_open) as opened:
                loaded, metadata = cast(tuple[dict[str, torch.Tensor], dict[str, str]], utils.load_torch_file(path, return_metadata=True))
                assert opened.call_args.kwargs.get('backend') == expected
                assert metadata == {'fixture': 'lifetime'}
                assert all(torch.equal(values[k], loaded[k]) for k in values)
        loaded = cast(dict[str, torch.Tensor], utils.load_torch_file(path, device=torch.device('cuda:0')))
        assert all(t.device == torch.device('cuda:0') for t in loaded.values())
    print('PASS loader backends, metadata and explicit CUDA device', flush=True)


def payload(weight):
    if isinstance(weight, QuantizedTensor):
        return [weight._qdata, weight.params.scale]
    return [weight]


def snapshot(model):
    return [[t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).clone() for t in payload(m.weight)] for m in model]


def equal(a, b):
    return all(torch.equal(x, y) for aa, bb in zip(a, b) for x, y in zip(aa, bb))


@torch.inference_mode()
def check(layout, dtype):
    torch.manual_seed(120)
    ops = mixed_precision_ops(compute_dtype=dtype)
    def make():
        model = torch.nn.ModuleList()
        for i in range(3):
            m: Any = ops.Linear(1024, 1024, bias=False, device='cpu')
            w = torch.randn(1024, 1024, dtype=dtype) * 0.1
            if layout:
                w = QuantizedTensor.from_float(w.cuda(), layout).cpu()
                m.layout_type = layout
                m.quant_format = {'TensorCoreConvRotW4A4Layout': 'int4_tensorwise', 'TensorWiseINT8Layout': 'int8_tensorwise', 'TensorCoreFP8Layout': 'float8_e4m3fn'}[layout]
            m.weight = torch.nn.Parameter(w, requires_grad=False)
            model.append(m)
        return model
    model = make()
    original = snapshot(model)
    root = ModelPatcher(model, torch.device('cuda'), torch.device('cpu'))
    # Keep the actual loading list alive while replacing every Parameter.
    loading = root._load_list()
    refs = [weakref.ref(payload(m.weight)[0]) for m in model]
    for m in model:
        m.to('cuda')
    gc.collect()
    assert all(r() is None for r in refs), 'load list retained old payload'
    del loading
    model.cpu()
    # Two adapters share one layer and each has another independent target.
    def adapter():
        return LoRAAdapter(set(), (torch.randn(1024, 4) * .1, torch.randn(4, 1024) * .1, 4., None, None, None))
    foo = {'0.weight': adapter(), '1.weight': adapter()}
    bar = {'1.weight': adapter(), '2.weight': adapter()}
    def patched(patches, strength=1.):
        p = root.clone()
        p.add_patches(cast(Any, patches), strength_patch=strength, online_mode=False)
        return p
    def run(p):
        p.partially_load(torch.device('cuda'), 10**12)
        for key, backup in p.backup.items():
            i = int(key.split('.')[0])
            actual = [t.detach().cpu().contiguous().reshape(-1).view(torch.uint8) for t in payload(backup.weight)]
            assert all(torch.equal(x, y) for x, y in zip(actual, original[i]))
        for m in model:
            assert isinstance(m.weight, QuantizedTensor) == bool(layout)
            weight: Any = m.weight
            w = weight.dequantize() if layout else weight
            assert torch.isfinite(w).all()
        return snapshot(model)
    f = patched(foo)
    first = run(f)
    assert not equal(first, original)
    assert equal(run(f), first), 'identical request changed weights'
    b = patched(bar)
    switched = run(b)
    run(root)
    assert equal(snapshot(model), original), 'unpatch did not restore original bytes'
    assert equal(run(patched(bar)), switched), 'bar depends on previous foo'
    run(root)
    assert equal(run(patched(foo)), first), 'foo accumulated error'
    half = run(patched(foo, .5))
    assert not equal(half, first), 'strength change had no effect'
    run(root)
    assert equal(run(patched(foo, .5)), half), 'strength change depends on history'
    both = patched(foo)
    both.add_patches(cast(Any, bar), online_mode=False)
    combined = run(both)
    both.partially_unload(torch.device('cpu'), 10**12)
    assert equal(run(both), combined), 'offload/reload changed merged weights'
    run(root)
    assert equal(snapshot(model), original)
    root.unpatch_model(torch.device('cpu'))
    print('PASS', layout or str(dtype), flush=True)


if __name__ == '__main__':
    check_loader()
    for layout, dtype in [
        ('TensorCoreConvRotW4A4Layout', torch.bfloat16),
        ('TensorWiseINT8Layout', torch.bfloat16),
        ('TensorCoreFP8Layout', torch.bfloat16),
        (None, torch.float16), (None, torch.bfloat16),
    ]:
        check(layout, dtype)
