r"""Benchmark one real Krea 2 image-transformer block: quantized vs BF16.

Only the selected block is read from the checkpoint.  The BF16 baseline is
materialized by dequantizing that block's checkpoint weights before timing;
therefore a separate full-precision checkpoint is not required.

Run from the repository root, optionally with the same attention flags used by
Forge, for example::

    .\venv\Scripts\python.exe scripts/diagnostics/benchmark_krea2_block.py --sage

Benchmark settings can be overridden with environment variables:
KREA_BENCH_CHECKPOINT, KREA_BENCH_BLOCK, KREA_BENCH_WIDTH,
KREA_BENCH_HEIGHT, KREA_BENCH_PROMPT, KREA_BENCH_TEXT_TOKENS,
KREA_BENCH_WARMUP and
KREA_BENCH_REPEATS.
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import sys
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules_forge.initialization import initialize_forge

initialize_forge()

import torch
from safetensors import safe_open

# Transformers is installed in Forge's runtime environment, which Pyright does
# not necessarily use when it analyzes this standalone diagnostic.
no_init_weights = import_module("transformers.modeling_utils").no_init_weights

from backend.nn.flux import EmbedND
from backend.nn.krea import SingleStreamBlock, TextFusionTransformer
from backend.operations import using_forge_operations
from backend.operations_mixed_precision import mixed_precision_ops
from backend import quant_ops


# comfy-kitchen dynamically exports this class, so its static module surface is
# intentionally incomplete.
QuantizedTensor: Any = getattr(quant_ops, "QuantizedTensor")


CHECKPOINT = Path(
    os.environ.get(
        "KREA_BENCH_CHECKPOINT",
        ROOT / "models/Stable-diffusion/krea2_turbo_int4_tensorwise_mixed.safetensors",
    )
)
BLOCK_INDEX = int(os.environ.get("KREA_BENCH_BLOCK", "1"))
OUTPUT_WIDTH = int(os.environ.get("KREA_BENCH_WIDTH", "1408"))
OUTPUT_HEIGHT = int(os.environ.get("KREA_BENCH_HEIGHT", "1024"))
PROMPT = os.environ.get("KREA_BENCH_PROMPT", "diagnostic test image")
TEXT_TOKENS_OVERRIDE = os.environ.get("KREA_BENCH_TEXT_TOKENS")
WARMUP = int(os.environ.get("KREA_BENCH_WARMUP", "4"))
REPEATS = int(os.environ.get("KREA_BENCH_REPEATS", "12"))

FEATURES = 6144
HEADS = 48
KV_HEADS = 12
MLP_MULTIPLIER = 4
VAE_DOWNSAMPLE = 8
PATCH_SIZE = 2

PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


def forge_prompt_token_count(prompt: str) -> int:
    """Mirror Qwen3VLTextProcessingEngine.strip_template token slicing."""
    AutoTokenizer = import_module("transformers").AutoTokenizer

    tokenizer_path = ROOT / "backend/huggingface/krea/Krea-2-Raw/tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokens = tokenizer(PROMPT_TEMPLATE.format(prompt))["input_ids"]
    template_end = 0
    count_im_start = 0
    for index, token in enumerate(tokens):
        if int(token) == 151644 and count_im_start < 2:
            template_end = index
            count_im_start += 1
    if len(tokens) > template_end + 3 and int(tokens[template_end + 1]) == 872 and int(tokens[template_end + 2]) == 198:
        template_end += 3
    return len(tokens) - template_end


TEXT_TOKENS = int(TEXT_TOKENS_OVERRIDE) if TEXT_TOKENS_OVERRIDE is not None else forge_prompt_token_count(PROMPT)


def load_block(index: int) -> SingleStreamBlock:
    ops = mixed_precision_ops(compute_dtype=torch.bfloat16)
    with no_init_weights(), using_forge_operations(
        operations=ops,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        manual_cast_enabled=False,
    ):
        block = SingleStreamBlock(
            FEATURES,
            HEADS,
            MLP_MULTIPLIER,
            bias=False,
            kvheads=KV_HEADS,
        )

    prefix = f"blocks.{index}."
    state_dict = {}
    # pread avoids mapping the entire 8.6 GiB file into this diagnostic process.
    with safe_open(CHECKPOINT, framework="pt", device="cpu", backend="pread") as f:
        for key in f.keys():
            if key.startswith(prefix):
                state_dict[key[len(prefix) :]] = f.get_tensor(key)

    missing, unexpected = block.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Block load mismatch: missing={missing}, unexpected={unexpected}")
    return block


def load_text_fusion() -> TextFusionTransformer:
    ops = mixed_precision_ops(compute_dtype=torch.bfloat16)
    with no_init_weights(), using_forge_operations(
        operations=ops,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        manual_cast_enabled=False,
    ):
        model = TextFusionTransformer(
            num_txt_layers=12,
            txt_dim=2560,
            heads=20,
            multiplier=4,
            bias=False,
            kvheads=20,
        )

    prefix = "txtfusion."
    state_dict = {}
    with safe_open(CHECKPOINT, framework="pt", device="cpu", backend="pread") as f:
        for key in f.keys():
            if key.startswith(prefix):
                state_dict[key[len(prefix) :]] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Text-fusion load mismatch: missing={missing}, unexpected={unexpected}")
    return model


def quantized_linears(block: SingleStreamBlock) -> Iterator[tuple[str, Any]]:
    for name, module in block.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, QuantizedTensor):
            yield name, module


def checkpoint_block_signatures():
    projection_names = (
        "attn.wq",
        "attn.wk",
        "attn.wv",
        "attn.gate",
        "attn.wo",
        "mlp.gate",
        "mlp.up",
        "mlp.down",
    )
    signatures = {}
    with safe_open(CHECKPOINT, framework="pt", device="cpu", backend="pread") as f:
        for index in range(28):
            formats = []
            for name in projection_names:
                key = f"blocks.{index}.{name}.comfy_quant"
                conf = json.loads(f.get_tensor(key).numpy().tobytes())
                formats.append(conf["format"])
            signatures[index] = tuple(formats)
    return signatures


def make_inputs(device: torch.device):
    if OUTPUT_WIDTH % (VAE_DOWNSAMPLE * PATCH_SIZE) or OUTPUT_HEIGHT % (VAE_DOWNSAMPLE * PATCH_SIZE):
        raise ValueError("Output width and height must be divisible by 16 for this benchmark")

    grid_w = OUTPUT_WIDTH // VAE_DOWNSAMPLE // PATCH_SIZE
    grid_h = OUTPUT_HEIGHT // VAE_DOWNSAMPLE // PATCH_SIZE
    image_tokens = grid_h * grid_w
    sequence = TEXT_TOKENS + image_tokens

    generator = torch.Generator(device=device).manual_seed(12345)
    x = torch.randn((1, sequence, FEATURES), device=device, dtype=torch.bfloat16, generator=generator) * 0.02
    vec = torch.randn((1, 1, 6 * FEATURES), device=device, dtype=torch.bfloat16, generator=generator) * 0.02

    text_pos = torch.zeros((1, TEXT_TOKENS, 3), device=device, dtype=torch.float32)
    image_pos = torch.zeros((grid_h, grid_w, 3), device=device, dtype=torch.float32)
    image_pos[..., 1] = torch.arange(grid_h, device=device, dtype=torch.float32)[:, None]
    image_pos[..., 2] = torch.arange(grid_w, device=device, dtype=torch.float32)[None, :]
    positions = torch.cat((text_pos, image_pos.reshape(1, image_tokens, 3)), dim=1)
    freqs = EmbedND(dim=128, theta=1000, axes_dim=[32, 48, 48])(positions)
    return x, vec, freqs, image_tokens


@torch.inference_mode()
def measure(block, x_seed, vec, freqs, warmup: int, repeats: int):
    work = torch.empty_like(x_seed)

    def run_once():
        work.copy_(x_seed)
        return block(work, vec, freqs, None, transformer_options={})

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    values = []
    output: torch.Tensor | None = None
    for _ in range(repeats):
        work.copy_(x_seed)
        # Exclude input restoration from the measured block forward.
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = block(work, vec, freqs, None, transformer_options={})
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))

    values.sort()
    assert output is not None
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "min_ms": values[0],
        "max_ms": values[-1],
        "samples_ms": values,
    }, output.detach().clone()


@torch.inference_mode()
def measure_linear(module, value, warmup: int, repeats: int):
    for _ in range(warmup):
        module(value)
    torch.cuda.synchronize()

    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        module(value)
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    values.sort()
    return statistics.median(values)


@torch.inference_mode()
def measure_text_fusion(model, value, warmup: int, repeats: int):
    work = torch.empty_like(value)
    for _ in range(warmup):
        work.copy_(value)
        model(work, mask=None, transformer_options={})
    torch.cuda.synchronize()

    values = []
    for _ in range(repeats):
        work.copy_(value)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(work, mask=None, transformer_options={})
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    values.sort()
    del work
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "min_ms": values[0],
        "max_ms": values[-1],
        "samples_ms": values,
    }


@torch.inference_mode()
def measure_text_block(block, value, warmup: int, repeats: int):
    work = torch.empty_like(value)
    for _ in range(warmup):
        work.copy_(value)
        block(work, mask=None, transformer_options={})
    torch.cuda.synchronize()

    values = []
    for _ in range(repeats):
        work.copy_(value)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        block(work, mask=None, transformer_options={})
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    del work
    return statistics.median(values)


def measure_projections(block, tokens: int, device: torch.device):
    """Measure the eight projection calls separately from attention/norms."""
    rows = []
    inputs = {}
    generator = torch.Generator(device=device).manual_seed(54321)
    for name, module in list(quantized_linears(block)):
        in_features = module.in_features
        if in_features not in inputs:
            inputs[in_features] = (
                torch.randn((1, tokens, in_features), device=device, dtype=torch.bfloat16, generator=generator) * 0.02
            )
        value = inputs[in_features]
        quant_ms = measure_linear(module, value, WARMUP, REPEATS)

        original = module.weight
        original_full = getattr(module, "_full_precision_mm", False)
        dense = original.dequantize().to(device=device, dtype=torch.bfloat16)
        module.weight = torch.nn.Parameter(dense, requires_grad=False)
        module._full_precision_mm = True
        try:
            bf16_ms = measure_linear(module, value, WARMUP, REPEATS)
        finally:
            module.weight = original
            module._full_precision_mm = original_full
        rows.append((name, getattr(module, "quant_format", "unknown"), quant_ms, bf16_ms))
        del dense
    return rows


def materialize_bf16(block: SingleStreamBlock):
    """Replace quantized weights with reconstructed BF16, returning originals."""
    originals = []
    formats = {}
    for name, module in list(quantized_linears(block)):
        original = module.weight
        formats[name] = getattr(module, "quant_format", "unknown")
        # Dequantization is deliberately outside the timed region.
        dense = original.dequantize().to(device=original.device, dtype=torch.bfloat16)
        originals.append((module, original, getattr(module, "_full_precision_mm", False)))
        module.weight = torch.nn.Parameter(dense, requires_grad=False)
        module._full_precision_mm = True
    torch.cuda.synchronize()
    return originals, formats


def restore_quantized(originals):
    for module, weight, full_precision_mm in originals:
        module.weight = weight
        module._full_precision_mm = full_precision_mm


def fmt(result):
    return (
        f"median={result['median_ms']:.3f} ms, mean={result['mean_ms']:.3f} ms, "
        f"range={result['min_ms']:.3f}..{result['max_ms']:.3f} ms"
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    device = torch.device("cuda:0")
    torch.set_grad_enabled(False)
    block = load_block(BLOCK_INDEX).to(device).eval()
    x, vec, freqs, image_tokens = make_inputs(device)

    layer_formats = {name: getattr(module, "quant_format", "unknown") for name, module in quantized_linears(block)}
    counts = {kind: list(layer_formats.values()).count(kind) for kind in sorted(set(layer_formats.values()))}
    print(
        f"Krea2 block {BLOCK_INDEX}; output={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}; "
        f"image_tokens={image_tokens}; text_tokens={TEXT_TOKENS}; total_tokens={x.shape[1]}"
    )
    print(f"Quantized projections: {counts}")
    print("Projection formats: " + ", ".join(f"{name}={kind}" for name, kind in layer_formats.items()))

    quant_result, quant_output = measure(block, x, vec, freqs, WARMUP, REPEATS)
    print("Checkpoint mixed INT4/INT8: " + fmt(quant_result))

    originals, _ = materialize_bf16(block)
    try:
        torch.cuda.reset_peak_memory_stats(device)
        bf16_result, bf16_output = measure(block, x, vec, freqs, WARMUP, REPEATS)
        bf16_peak = torch.cuda.max_memory_allocated(device) / 1024**3
    finally:
        restore_quantized(originals)

    speedup = bf16_result["median_ms"] / quant_result["median_ms"]
    delta = (quant_result["median_ms"] / bf16_result["median_ms"] - 1.0) * 100.0
    diff = (quant_output.float() - bf16_output.float()).abs()
    print("Reconstructed BF16:       " + fmt(bf16_result))
    print(f"BF16 / quantized speed ratio: {speedup:.3f}x ({delta:+.1f}% quantized time vs BF16)")
    print(f"Output difference: mean_abs={diff.mean().item():.6g}, max_abs={diff.max().item():.6g}")
    print(f"BF16 phase CUDA peak allocated: {bf16_peak:.3f} GiB")

    projection_rows = measure_projections(block, x.shape[1], device)
    print("Projection medians (quantized -> BF16):")
    for name, kind, quant_ms, bf16_ms in projection_rows:
        print(f"  {name:10s} {kind:19s} {quant_ms:8.3f} -> {bf16_ms:8.3f} ms  ({bf16_ms / quant_ms:.2f}x)")
    quant_projection_ms = sum(row[2] for row in projection_rows)
    bf16_projection_ms = sum(row[3] for row in projection_rows)
    print(
        f"Projection sum: {quant_projection_ms:.3f} -> {bf16_projection_ms:.3f} ms; "
        f"non-projection remainder estimate: "
        f"{quant_result['median_ms'] - quant_projection_ms:.3f} / "
        f"{bf16_result['median_ms'] - bf16_projection_ms:.3f} ms"
    )

    # Krea 2 executes this BF16 text-fusion transformer inside every diffusion
    # forward.  Its two layerwise blocks see B=text_tokens, L=12; its two
    # refiner blocks see B=1, L=text_tokens.
    # Time one representative of every actual INT4/INT8 projection signature,
    # then weight it by the number of blocks using that signature.  Weight
    # values do not affect GEMM scheduling, so duplicate signatures need not be
    # loaded repeatedly.
    signatures = checkpoint_block_signatures()
    signature_counts = Counter(signatures.values())
    signature_representatives = {signature: index for index, signature in signatures.items()}
    selected_signature = signatures[BLOCK_INDEX]
    signature_times = {selected_signature: quant_result["median_ms"]}

    del bf16_output, quant_output, originals, projection_rows, block
    gc.collect()
    torch.cuda.empty_cache()

    for signature, representative in signature_representatives.items():
        if signature in signature_times:
            continue
        candidate = load_block(representative).to(device).eval()
        result, output = measure(candidate, x, vec, freqs, WARMUP, REPEATS)
        signature_times[signature] = result["median_ms"]
        del output, candidate
        gc.collect()
        torch.cuda.empty_cache()

    image_blocks_ms = sum(signature_counts[signature] * signature_times[signature] for signature in signature_counts)
    print(f"Image-block signature-weighted total ({len(signature_counts)} signatures): {image_blocks_ms:.3f} ms")
    for signature, count in sorted(signature_counts.items(), key=lambda item: (-item[1], item[0])):
        short = "".join("4" if kind == "int4_tensorwise" else "8" for kind in signature)
        print(f"  {count:2d}x {short}: {signature_times[signature]:.3f} ms/block")

    text_fusion = load_text_fusion().to(device).eval()
    text_generator = torch.Generator(device=device).manual_seed(67890)
    text_input = torch.randn(
        (1, TEXT_TOKENS, 12, 2560),
        device=device,
        dtype=torch.bfloat16,
        generator=text_generator,
    ) * 0.02
    text_result = measure_text_fusion(text_fusion, text_input, WARMUP, REPEATS)
    layerwise_input = text_input.reshape(TEXT_TOKENS, 12, 2560).contiguous()
    refiner_input = torch.randn(
        (1, TEXT_TOKENS, 2560),
        device=device,
        dtype=torch.bfloat16,
        generator=text_generator,
    ) * 0.02
    special_block_times = [
        measure_text_block(text_fusion.layerwise_blocks[0], layerwise_input, WARMUP, REPEATS),
        measure_text_block(text_fusion.layerwise_blocks[1], layerwise_input, WARMUP, REPEATS),
        measure_text_block(text_fusion.refiner_blocks[0], refiner_input, WARMUP, REPEATS),
        measure_text_block(text_fusion.refiner_blocks[1], refiner_input, WARMUP, REPEATS),
    ]
    special_blocks_ms = sum(special_block_times)
    bf16_image_blocks_ms = 28 * bf16_result["median_ms"]
    print("Complete text-fusion path: " + fmt(text_result))
    print(
        "Four special BF16 blocks: "
        + " + ".join(f"{value:.3f}" for value in special_block_times)
        + f" = {special_blocks_ms:.3f} ms"
    )
    print(
        f"28 image blocks + 4 special blocks estimate: "
        f"quantized={image_blocks_ms + special_blocks_ms:.3f} ms; "
        f"BF16={bf16_image_blocks_ms + special_blocks_ms:.3f} ms"
    )

    del x, vec, freqs, text_fusion, text_input, layerwise_input, refiner_input
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
