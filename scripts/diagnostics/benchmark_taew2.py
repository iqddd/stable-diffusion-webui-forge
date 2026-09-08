r"""Benchmark Forge's taew2_1 TAEHV decoder for a single output frame.

Run from the repository root::

    .\venv\Scripts\python.exe scripts/diagnostics/benchmark_taew2.py

Environment overrides: TAEW2_WIDTH, TAEW2_HEIGHT, TAEW2_WARMUP,
TAEW2_REPEATS and TAEW2_MODEL.
"""

from __future__ import annotations

import gc
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules_forge.initialization import initialize_forge

initialize_forge()

import torch

from modules import devices
from modules.sd_vae_taesd import TAEHVDecoder, _TAEW2OneFrame


MODEL = Path(os.environ.get("TAEW2_MODEL", ROOT / "models/VAE-taesd/taew2_1.pth"))
WIDTH = int(os.environ.get("TAEW2_WIDTH", "1408"))
HEIGHT = int(os.environ.get("TAEW2_HEIGHT", "1024"))
WARMUP = int(os.environ.get("TAEW2_WARMUP", "4"))
REPEATS = int(os.environ.get("TAEW2_REPEATS", "20"))
COMPILE_MODE = os.environ.get("TAEW2_COMPILE", "none").strip().lower()
LATENT_CHANNELS = 16
SPATIAL_SCALE = 8


def mib(value: int) -> float:
    return value / 1024**2


def tensor_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def module_bytes(module: torch.nn.Module) -> int:
    tensors = list(module.parameters()) + list(module.buffers())
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    return sum(storages.values())


@torch.inference_mode()
def main():
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    if WIDTH % SPATIAL_SCALE or HEIGHT % SPATIAL_SCALE:
        raise ValueError(f"Width and height must be divisible by {SPATIAL_SCALE}")

    device = devices.device
    dtype = devices.dtype
    model = TAEHVDecoder(MODEL, LATENT_CHANNELS).eval().to(device=device, dtype=dtype)
    generator = torch.Generator(device=device).manual_seed(12345)
    latent = torch.randn(
        (1, LATENT_CHANNELS, 1, HEIGHT // SPATIAL_SCALE, WIDTH // SPATIAL_SCALE),
        device=device,
        dtype=dtype,
        generator=generator,
    )

    decode = model
    compile_mean_abs = None
    compile_max_abs = None
    if COMPILE_MODE != "none":
        unrolled = _TAEW2OneFrame(model.decoder).eval()
        reference = model(latent)
        unrolled_reference = unrolled(latent)
        if not torch.equal(reference, unrolled_reference):
            raise RuntimeError(
                f"One-frame specialization mismatch: max_abs={(reference - unrolled_reference).abs().max().item()}"
            )
        if COMPILE_MODE not in {"static", "dynamic"}:
            raise ValueError("TAEW2_COMPILE must be one of: none, static, dynamic")
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.suppress_errors = False
        decode = torch.compile(
            unrolled,
            backend="inductor",
            dynamic=COMPILE_MODE == "dynamic",
            fullgraph=True,
        )
        if COMPILE_MODE == "dynamic":
            torch._dynamo.mark_dynamic(latent, 3)
            torch._dynamo.mark_dynamic(latent, 4)
        compiled_reference = decode(latent)
        torch.cuda.synchronize(device)
        compile_diff = (reference - compiled_reference).abs().float()
        compile_mean_abs = compile_diff.mean().item()
        compile_max_abs = compile_diff.max().item()
        del reference, unrolled_reference, compiled_reference, compile_diff

    for _ in range(WARMUP):
        output = decode(latent)
        del output
    torch.cuda.synchronize(device)

    # Empty unused cached blocks so reserved memory describes this decoder,
    # then measure a single forward while retaining its output.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    output = decode(latent)
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    output_size = tensor_bytes(output)
    incremental_peak = peak_allocated - baseline_allocated
    workspace_peak = max(0, incremental_peak - output_size)

    expected_shape = (1, 3, HEIGHT, WIDTH)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(f"Unexpected decoder output: expected {expected_shape}, got {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise RuntimeError("Decoder produced non-finite values")
    del output

    samples = []
    for _ in range(REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = decode(latent)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del output
    samples.sort()

    print(f"model={MODEL}")
    print(f"device={device} dtype={dtype} compile={COMPILE_MODE}")
    print(f"latent_shape={tuple(latent.shape)} output_shape={expected_shape}")
    print(
        f"time_ms median={statistics.median(samples):.3f} "
        f"mean={statistics.fmean(samples):.3f} min={samples[0]:.3f} max={samples[-1]:.3f}"
    )
    if compile_max_abs is not None:
        print(f"compile_difference mean_abs={compile_mean_abs:.9g} max_abs={compile_max_abs:.9g}")
    print("samples_ms=" + ",".join(f"{sample:.3f}" for sample in samples))
    print(
        f"vram_mib model={mib(module_bytes(model)):.3f} input={mib(tensor_bytes(latent)):.3f} "
        f"baseline_allocated={mib(baseline_allocated):.3f} peak_allocated={mib(peak_allocated):.3f} "
        f"incremental_peak={mib(incremental_peak):.3f} output={mib(output_size):.3f} "
        f"workspace_peak={mib(workspace_peak):.3f} baseline_reserved={mib(baseline_reserved):.3f} "
        f"peak_reserved={mib(peak_reserved):.3f}"
    )


if __name__ == "__main__":
    main()
