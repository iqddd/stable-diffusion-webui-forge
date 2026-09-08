# https://github.com/Comfy-Org/ComfyUI/blob/v0.27.0/comfy/quant_ops.py

import comfy_kitchen as ck
import torch
from comfy_kitchen.tensor import (  # noqa
    QuantizedTensor,
    TensorCoreConvRotW4A4Layout,
    TensorCoreFP8Layout,
    TensorCoreMXFP8Layout,
    TensorCoreNVFP4Layout,
    TensorWiseINT8Layout,
    get_layout_class,
    register_layout_class,
)

if torch.version.cuda is None:
    ck.registry.disable("cuda")
else:
    cuda_version = tuple(map(int, str(torch.version.cuda).split(".")))
    if cuda_version < (13,):
        ck.registry.disable("cuda")

from backend.args import args

if args.int4_convrot_compile_precision:
    # These are Inductor-wide compilation settings.  The flag is named for the
    # ConvRot use case because W4A4 activation quantization amplifies small
    # numerical changes in operations that Inductor fuses ahead of the opaque
    # kernel.  The CUDA kernel itself is unaffected by these settings.
    torch._inductor.config.force_same_precision = True
    torch._inductor.config.emulate_precision_casts = True
    print("INT4 ConvRot compile precision preservation enabled")

if args.enable_triton_backend:
    try:
        import triton  # noqa
    except ImportError:
        ck.registry.disable("triton")
else:
    ck.registry.disable("triton")

import importlib.metadata

ver = importlib.metadata.version("comfy-kitchen")

print(f"Comfy-Kitchen {ver}:", {k: v["available"] and not v["disabled"] for k, v in ck.list_backends().items()})


# comfy-kitchen 0.2.x exposes ConvRot through Python registry and DLPack
# bindings.  Those calls cannot be traced with FakeTensor, so expose the
# existing CUDA implementation as an opaque dispatcher operator.  This does
# not replace or emulate the kernel: eager execution enters the same
# comfy-kitchen implementation and its already-built native extension.
@torch.library.custom_op("forge_quant::convrot_w4a4_linear", mutates_args=())
def convrot_w4a4_linear_compile_safe(
    input: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    convrot_groupsize: int,
    quant_group_size: int,
    linear_dtype: str,
) -> torch.Tensor:
    return ck.convrot_w4a4_linear(
        input,
        qweight,
        weight_scale,
        bias=bias,
        convrot_groupsize=convrot_groupsize,
        quant_group_size=quant_group_size,
        linear_dtype=linear_dtype,
    )


@convrot_w4a4_linear_compile_safe.register_fake
def _convrot_w4a4_linear_compile_safe_fake(
    input,
    qweight,
    weight_scale,
    bias,
    convrot_groupsize,
    quant_group_size,
    linear_dtype,
):
    return input.new_empty((*input.shape[:-1], qweight.shape[0]))


# region Registry


register_layout_class("TensorCoreFP8Layout", TensorCoreFP8Layout)
register_layout_class("TensorCoreFP8E4M3Layout", TensorCoreFP8Layout)
register_layout_class("TensorCoreFP8E5M2Layout", TensorCoreFP8Layout)
register_layout_class("TensorCoreNVFP4Layout", TensorCoreNVFP4Layout)
register_layout_class("TensorCoreMXFP8Layout", TensorCoreMXFP8Layout)
register_layout_class("TensorWiseINT8Layout", TensorWiseINT8Layout)
register_layout_class("TensorCoreConvRotW4A4Layout", TensorCoreConvRotW4A4Layout)


QUANT_ALGOS = {
    "float8_e4m3fn": {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E4M3Layout",
    },
    "float8_e5m2": {
        "storage_t": torch.float8_e5m2,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E5M2Layout",
    },
    "nvfp4": {
        "storage_t": torch.uint8,
        "parameters": {"weight_scale", "weight_scale_2", "input_scale"},
        "comfy_tensor_layout": "TensorCoreNVFP4Layout",
        "group_size": 16,
    },
    "mxfp8": {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreMXFP8Layout",
        "group_size": 32,
    },
    "int8_tensorwise": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale"},
        "comfy_tensor_layout": "TensorWiseINT8Layout",
        "quantize_input": False,
    },
    "convrot_w4a4": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale"},
        "comfy_tensor_layout": "TensorCoreConvRotW4A4Layout",
        "quantize_input": False,
    },
    # Compatibility name used by tensor-wise INT4 ConvRot checkpoints.  The
    # packed weights and per-output scales are exactly the layout consumed by
    # comfy-kitchen's TensorCoreConvRotW4A4Layout.
    "int4_tensorwise": {
        "storage_t": torch.int8,
        "parameters": {"weight_scale"},
        "comfy_tensor_layout": "TensorCoreConvRotW4A4Layout",
        "quantize_input": False,
    },
}


# region float


def stochastic_rounding(value: torch.Tensor, dtype: torch.dtype, seed: int = 0):
    if dtype is torch.float32:
        return value.to(dtype=torch.float32)
    if dtype is torch.float16:
        return value.to(dtype=torch.float16)
    if dtype is torch.bfloat16:
        return value.to(dtype=torch.bfloat16)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed)
        rng = torch.randint(0, 256, value.size(), dtype=torch.uint8, layout=value.layout, device=value.device, generator=generator)
        return ck.stochastic_rounding_fp8(value, rng, dtype)

    return value.to(dtype=dtype)
