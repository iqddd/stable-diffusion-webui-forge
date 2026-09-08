"""Diagnostic launch wrapper; records payload ownership at transfer events."""
import functools
import json
import runpy
import sys
import weakref
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from modules_forge.initialization import initialize_forge

initialize_forge()

import psutil
import torch
import backend.operations
from backend.patcher.base import ModelPatcher
from backend import operations_mixed_precision as mixed
from backend import quant_ops


QuantizedTensor: Any = getattr(quant_ops, 'QuantizedTensor')


def plain(tensor):
    if isinstance(tensor, QuantizedTensor):
        yield tensor._qdata
        yield from (x for x in vars(tensor.params).values() if isinstance(x, torch.Tensor))
    else:
        yield tensor


def amounts(tensors):
    sizes = {}
    for tensor in tensors:
        for value in plain(tensor):
            storage = value.untyped_storage()
            sizes[(value.device.type, storage.data_ptr())] = storage.nbytes()
    return {device: sum(size for (kind, _), size in sizes.items() if kind == device) / 1024**3 for device in ('cpu', 'cuda')}


active = []


def snapshot(event, patcher, refs):
    model = amounts(patcher.model.parameters())
    backup = amounts(item.weight for item in patcher.backup.values())
    old = amounts(tensor for ref in refs if (tensor := ref()) is not None)
    memory = psutil.Process().memory_info()
    print('[WEIGHTLIFE] ' + json.dumps(dict(
        event=event, model=type(patcher.model).__name__, model_gib=model,
        backup_gib=backup, surviving_original_objects_gib=old,
        private_gib=memory.private / 1024**3,
        cuda_allocated_gib=torch.cuda.memory_allocated() / 1024**3,
        cuda_reserved_gib=torch.cuda.memory_reserved() / 1024**3,
    )), flush=True)


original_load = ModelPatcher.load


@functools.wraps(original_load)
def load(patcher, *args, **kwargs):
    refs = [weakref.ref(value) for tensor in patcher.model.parameters() for value in plain(tensor)]
    state = [patcher, refs, 0, 1]
    active.append(state)
    snapshot('before-load', patcher, refs)
    try:
        return original_load(patcher, *args, **kwargs)
    finally:
        active.pop()
        snapshot('load-returned', patcher, refs)


original_apply = mixed._quantized_apply
original_patch = ModelPatcher.patch_weight_to_device


@functools.wraps(original_patch)
def patch_weight(patcher, key, *args, **kwargs):
    result = original_patch(patcher, key, *args, **kwargs)
    if active and key in patcher.patches:
        # Merge events expose necessary backup allocations independently from
        # the transfer list. No tensor references are retained by the probe.
        backup_bytes = amounts(item.weight for item in patcher.backup.values())['cpu']
        state = active[-1]
        bucket = int(backup_bytes)
        previous = state[4] if len(state) > 4 else -1
        if bucket != previous:
            if len(state) == 4:
                state.append(bucket)
            else:
                state[4] = bucket
            snapshot('merge-backup-progress', patcher, state[1])
    return result


@functools.wraps(original_apply)
def apply(module, *args, **kwargs):
    result = original_apply(module, *args, **kwargs)
    if active:
        state = active[-1]
        state[2] += sum(value.numel() * value.element_size() for tensor in module.parameters(recurse=False) for value in plain(tensor))
        if state[2] >= state[3] * 1024**3:
            snapshot('transfer-progress', state[0], state[1])
            state[3] = state[2] // 1024**3 + 1
    return result


setattr(ModelPatcher, 'load', load)
setattr(ModelPatcher, 'patch_weight_to_device', patch_weight)
mixed._quantized_apply = apply
runpy.run_path(str(ROOT / 'launch.py'), run_name='__main__')
