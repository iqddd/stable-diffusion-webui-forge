"""Prewarm Forge's disk-backed INT8 autotune buckets without loading Krea or TE.

Run from the repository venv. Defaults cover batch=1 text-to-image, both image
dimensions in [960, 1216], up to 512 prompt tokens plus 5 template positions.
Only INT8 layers in the main DiT blocks are covered (not INT4 or attention).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/Stable-diffusion/krea2_turbo_int4_tensorwise_mixed.safetensors")
    parser.add_argument("--min-side", type=int, default=960)
    parser.add_argument("--max-side", type=int, default=1216)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--template-tokens", type=int, default=5)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--verify-cache-only", action="store_true", help="Fail on any autotune disk miss; use a different M within each bucket.")
    args = parser.parse_args()
    if not 0 < args.min_side <= args.max_side or args.min_side % 16 or args.max_side % 16:
        parser.error("Image bounds must be positive, ordered multiples of 16")
    if args.prompt_tokens < 0 or args.template_tokens < 1:
        parser.error("Token bounds must be nonnegative with at least one template token")

    import torch
    import triton
    from safetensors import safe_open
    from backend import operations_triton as ops

    # Read only small metadata/shape tensors; do not materialize checkpoint weights.
    shapes = set()
    with safe_open(args.checkpoint, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        for key in sorted(keys):
            if not key.endswith(".comfy_quant"):
                continue
            prefix = key.removesuffix(".comfy_quant")
            conf = json.loads(bytes(checkpoint.get_tensor(key).tolist()))
            if conf.get("format") != "int8_tensorwise":
                continue
            if not prefix.startswith("blocks."):
                raise ValueError(f"INT8 layer outside main DiT blocks needs a separate M range: {prefix}")
            n, k = checkpoint.get_slice(prefix + ".weight").get_shape()
            scale_shape = checkpoint.get_slice(prefix + ".weight_scale").get_shape()
            scale_count = 1
            for dim in scale_shape:
                scale_count *= dim
            if scale_count not in (1, n):
                raise ValueError(f"Unsupported scale shape {scale_shape}: {prefix}")
            shapes.add((n, k, scale_count != 1, prefix + ".bias" in keys))
    if not shapes:
        raise ValueError("Checkpoint contains no supported INT8 layers")

    min_m = (args.min_side // 16) ** 2 + args.template_tokens
    max_m = (args.max_side // 16) ** 2 + args.prompt_tokens + args.template_tokens
    # Enumerate through the production helper so coverage tracks its actual keys.
    buckets = {}
    for m in range(min_m, max_m + 1):
        buckets.setdefault(ops._int8_autotune_m_bucket(m), []).append(m)

    if args.verify_cache_only:
        def fail_on_benchmark(*unused_args, **unused_kwargs):
            raise RuntimeError("Autotune disk cache miss")
        ops._int8_matmul_dequant_kernel._bench = fail_on_benchmark
        ops._int8_matmul_dequant_per_row_kernel._bench = fail_on_benchmark

    torch.manual_seed(12345)
    dtype = getattr(torch, args.dtype)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "dtype": args.dtype,
                      "cache_dir": triton.knobs.cache.dir, "M_range": [min_m, max_m],
                      "buckets": len(buckets), "shapes": sorted(shapes),
                      "total": len(buckets) * len(shapes), "verify": args.verify_cache_only}), flush=True)
    started = time.perf_counter()
    done = 0
    with torch.inference_mode():
        for n, k, per_row, has_bias in sorted(shapes):
            weight = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
            scale = torch.rand((n, 1) if per_row else (1,), device="cuda", dtype=torch.float32) * 0.01
            bias = torch.randn(n, device="cuda", dtype=dtype) if has_bias else None
            wrapper = ops.triton_int8_linear_per_row if per_row else ops.triton_int8_linear
            for bucket, members in buckets.items():
                # Tune near the centre, not exclusively on aligned boundary sizes.
                # Verification exercises a different actual length with the same key.
                m = members[0] if args.verify_cache_only else members[len(members) // 2]
                x = torch.randn((m, k), device="cuda", dtype=dtype)
                torch.cuda.synchronize()
                call_started = time.perf_counter()
                output = wrapper(x, weight, scale, bias, dtype)
                torch.cuda.synchronize()
                done += 1
                print(json.dumps({"done": done, "bucket": bucket, "M": m, "N": n, "K": k,
                                  "seconds": round(time.perf_counter() - call_started, 4)}), flush=True)
                del x, output
            del weight, scale, bias
    print(json.dumps({"complete": done, "seconds": round(time.perf_counter() - started, 2)}), flush=True)


if __name__ == "__main__":
    main()
