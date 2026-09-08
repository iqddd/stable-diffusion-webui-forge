r"""Continuously execute one compiled Krea 2 block until interrupted.

This is a sustained GPU-load diagnostic for external sensor monitoring.  It
uses the same 1408x1024 token geometry as benchmark_krea2_block.py and the
Torch Compile Integrated ``dynamic`` preset configuration.  Pass
``--int4-convrot-compile-precision`` to enable Forge's precision-preserving
Inductor settings.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.diagnostics.benchmark_krea2_block import BLOCK_INDEX, load_block, make_inputs


SYNC_INTERVAL = 32
REPORT_INTERVAL_SECONDS = 10.0
PROFILE_SECONDS = float(os.environ.get("KREA_STRESS_PROFILE_SECONDS", "0"))


def skip_torch_compile_dict(guard_entries):
    return [("transformer_options" not in entry.name) for entry in guard_entries]


def main():
    device = torch.device("cuda:0")
    torch.set_grad_enabled(False)

    block = load_block(BLOCK_INDEX).to(device).eval()
    x_seed, vec, freqs, image_tokens = make_inputs(device)
    work = torch.empty_like(x_seed)

    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.suppress_errors = True
    compiled = torch.compile(
        block,
        backend="inductor",
        dynamic=True,
        fullgraph=False,
        options={"guard_filter_fn": skip_torch_compile_dict},
    )

    # Compile and warm all paths before sensor statistics begin to matter.
    with torch.inference_mode():
        for _ in range(4):
            work.copy_(x_seed)
            compiled(work, vec, freqs, None, transformer_options={})
    torch.cuda.synchronize(device)

    print(
        f"STRESS_READY block={BLOCK_INDEX} tokens={x_seed.shape[1]} "
        f"image_tokens={image_tokens} batch_sync={SYNC_INTERVAL}",
        flush=True,
    )
    started = time.perf_counter()
    last_report = started
    iterations = 0
    profiling = PROFILE_SECONDS > 0
    cudart = torch.cuda.cudart()

    if profiling:
        if cudart is None:
            raise RuntimeError("CUDA runtime is unavailable")
        cudart.cudaProfilerStart()
        torch.cuda.nvtx.range_push("krea2_block_steady_state")

    try:
        with torch.inference_mode():
            while True:
                for _ in range(SYNC_INTERVAL):
                    work.copy_(x_seed)
                    compiled(work, vec, freqs, None, transformer_options={})
                torch.cuda.synchronize(device)
                iterations += SYNC_INTERVAL

                now = time.perf_counter()
                if now - last_report >= REPORT_INTERVAL_SECONDS:
                    elapsed = now - started
                    print(
                        f"STRESS_ALIVE iterations={iterations} elapsed_s={elapsed:.1f} "
                        f"avg_ms_per_block={elapsed * 1000.0 / iterations:.3f}",
                        flush=True,
                    )
                    last_report = now
                if profiling and now - started >= PROFILE_SECONDS:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        torch.cuda.synchronize(device)
        if profiling:
            torch.cuda.nvtx.range_pop()
            assert cudart is not None
            cudart.cudaProfilerStop()
        elapsed = time.perf_counter() - started
        print(
            f"STRESS_STOP iterations={iterations} elapsed_s={elapsed:.1f} "
            f"avg_ms_per_block={elapsed * 1000.0 / max(iterations, 1):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
