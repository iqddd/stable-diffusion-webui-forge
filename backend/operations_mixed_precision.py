# https://github.com/Comfy-Org/ComfyUI/blob/v0.27.0/comfy/ops.py#L1163

import json

import torch

from backend.args import args
from backend.memory_management import cast_to_device, logger

from .operations import (
    ForgeOperations,
    ForgeWeights,
    main_stream_worker,
    weights_manual_cast,
)
from .quant_ops import (  # noqa
    QUANT_ALGOS,
    QuantizedTensor,
    TensorCoreFP8Layout,
    TensorWiseINT8Layout,
    convrot_w4a4_linear_compile_safe,
    get_layout_class,
)

INT4_CONVROT_FORMATS = frozenset(("convrot_w4a4", "int4_tensorwise"))
INT4_QUANT_GROUP_SIZE = 64
INT4_CONVROT_GROUP_SIZES = frozenset((16, 64, 256))


def _quant_conf_value(layer_conf: dict, name: str, default=None):
    params_conf = layer_conf.get("params", {})
    if not isinstance(params_conf, dict):
        params_conf = {}
    return layer_conf.get(name, params_conf.get(name, default))


def _int4_convrot_params(
    *,
    layer_name: str,
    quant_format: str,
    layer_conf: dict,
    weight: torch.Tensor,
    scale: torch.Tensor,
    orig_shape: tuple[int, int],
) -> dict:
    out_features, in_features = orig_shape

    if weight.dtype is not torch.int8:
        raise ValueError(f"INT4 ConvRot weight for layer {layer_name} must use packed torch.int8 storage, got {weight.dtype}")
    if in_features % 2 != 0:
        raise ValueError(f"INT4 ConvRot input size for layer {layer_name} must be even, got {in_features}")

    expected_weight_shape = (out_features, in_features // 2)
    if weight.dim() != 2 or tuple(weight.shape) != expected_weight_shape:
        raise ValueError(
            f"Invalid packed INT4 ConvRot weight shape for layer {layer_name}: "
            f"expected {expected_weight_shape}, got {tuple(weight.shape)}"
        )

    if not isinstance(scale, torch.Tensor) or scale.dtype is not torch.float32:
        scale_dtype = getattr(scale, "dtype", type(scale).__name__)
        raise ValueError(f"INT4 ConvRot weight scale for layer {layer_name} must be torch.float32, got {scale_dtype}")
    if scale.numel() != out_features:
        raise ValueError(
            f"Invalid INT4 ConvRot weight scale for layer {layer_name}: "
            f"expected {out_features} values, got {scale.numel()}"
        )

    # convrot_w4a4 implies rotation for compatibility with checkpoints that
    # predate the explicit flag. int4_tensorwise must opt into it explicitly.
    convrot = bool(_quant_conf_value(layer_conf, "convrot", quant_format == "convrot_w4a4"))
    if not convrot:
        raise ValueError(f"INT4 tensor-wise layer {layer_name} is unsupported without ConvRot")

    convrot_groupsize = int(_quant_conf_value(layer_conf, "convrot_groupsize", 256))
    if convrot_groupsize not in INT4_CONVROT_GROUP_SIZES:
        raise ValueError(
            f"Unsupported INT4 ConvRot group size for layer {layer_name}: {convrot_groupsize}; "
            f"expected one of {sorted(INT4_CONVROT_GROUP_SIZES)}"
        )
    if in_features % convrot_groupsize != 0:
        raise ValueError(
            f"INT4 ConvRot group size {convrot_groupsize} does not divide input size "
            f"{in_features} for layer {layer_name}"
        )

    quant_group_size = int(_quant_conf_value(layer_conf, "quant_group_size", INT4_QUANT_GROUP_SIZE))
    if quant_group_size != INT4_QUANT_GROUP_SIZE:
        raise ValueError(
            f"Unsupported INT4 quantization group size for layer {layer_name}: {quant_group_size}; "
            f"expected {INT4_QUANT_GROUP_SIZE}"
        )

    linear_dtype = _quant_conf_value(layer_conf, "linear_dtype", "int4")
    if linear_dtype not in ("int4", "int8"):
        raise ValueError(f"Unsupported INT4 ConvRot linear dtype for layer {layer_name}: {linear_dtype!r}")

    return {
        "scale": scale.reshape(-1).contiguous(),
        "convrot_groupsize": convrot_groupsize,
        "quant_group_size": quant_group_size,
        "linear_dtype": linear_dtype,
    }

# TODO: Delete all these junks once comfy_kitchen fix AMD support...

if args.disable_int8_override:
    TRITON_AVAILABLE = False
else:
    try:
        from .operations_triton import triton_int8_linear, triton_int8_linear_per_row
    except ImportError:
        TRITON_AVAILABLE = False
    else:
        TRITON_AVAILABLE = True

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties()
            if props.major < 8:
                TRITON_AVAILABLE = False

        if TRITON_AVAILABLE:
            from .quant_rotation import build_hadamard, rotate_activation


def _quantized_apply(module: torch.nn.Module, fn, recurse=True):
    if recurse:
        for child in module.children():
            child._apply(fn)
    for key, param in module._parameters.items():
        if param is None:
            continue
        p: torch.Tensor = fn(param)
        try:
            module.register_parameter(key, torch.nn.Parameter(p, requires_grad=False))
        except RuntimeError:
            module.register_parameter(key, torch.nn.Parameter(p.clone(), requires_grad=False))
    for key, buf in module._buffers.items():
        if buf is not None:
            module._buffers[key] = fn(buf)
    return module


def _load_quantized_module(module: torch.nn.Module, super_load, state_dict: dict[str, torch.Tensor], prefix: str, local_metadata, strict, missing_keys, unexpected_keys, error_msgs, load_extra_params=False):
    device = module.factory_kwargs["device"]
    compute_dtype = module.factory_kwargs["dtype"]
    disabled_formats = module._disabled_formats
    layer_name = prefix.rstrip(".")

    weight = state_dict.pop(f"{prefix}weight", None)
    if weight is None:
        module.weight = None
        return
    manually_loaded_keys = [f"{prefix}weight"]

    def pop_scale(name, dtype=None):
        key = f"{prefix}{name}"
        v = state_dict.pop(key, None)
        if v is not None:
            v = v.to(device=device)
            if dtype is not None:
                v = v.view(dtype=dtype)
            manually_loaded_keys.append(key)
        return v

    layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
    if layer_conf is not None:
        layer_conf = json.loads(layer_conf.numpy().tobytes())
        if not isinstance(layer_conf, dict):
            raise ValueError(f"Invalid quantization metadata for layer {layer_name}: expected a JSON object")

    if layer_conf is None:
        module.weight = torch.nn.Parameter(weight.to(device=device, dtype=compute_dtype), requires_grad=False)
    else:
        module.quant_format = layer_conf.get("format", None)
        module._full_precision_mm_config = layer_conf.get("full_precision_matrix_mult", False)
        if not module._full_precision_mm:
            module._full_precision_mm = module._full_precision_mm_config
        if module.quant_format in disabled_formats:
            module._full_precision_mm = True
        if module.quant_format is None:
            raise ValueError(f"Unknown quantization format for layer {layer_name}")

        qconfig = QUANT_ALGOS.get(module.quant_format)
        if qconfig is None:
            raise ValueError(f"Unsupported quantization format {module.quant_format!r} for layer {layer_name}")
        module.layout_type = qconfig["comfy_tensor_layout"]
        layout_cls = get_layout_class(module.layout_type)

        # Per-format scales; fp8 dtype views handle both legacy uint8-on-disk and native fp8.
        if module.quant_format in ("float8_e4m3fn", "float8_e5m2"):
            scales = {"scale": pop_scale("weight_scale")}
        elif module.quant_format == "mxfp8":
            bs = pop_scale("weight_scale", torch.float8_e8m0fnu)
            if bs is None:
                raise ValueError(f"Missing MXFP8 block scales for layer {layer_name}")
            scales = {"scale": bs}
        elif module.quant_format == "nvfp4":
            ts = pop_scale("weight_scale_2")
            bs = pop_scale("weight_scale", torch.float8_e4m3fn)
            if ts is None or bs is None:
                raise ValueError(f"Missing NVFP4 scales for layer {layer_name}")
            scales = {"scale": ts, "block_scale": bs}
        elif module.quant_format == "int8_tensorwise":
            scale = pop_scale("weight_scale")
            if scale is None:
                raise ValueError(f"Missing INT8 weight scale for layer {layer_name}")
            module._per_row = scale.dim() == 2 and scale.shape[1] == 1
            scales = {"scale": scale}
            params_conf = layer_conf.get("params", {})
            if not isinstance(params_conf, dict):
                params_conf = {}
            if layer_conf.get("convrot", params_conf.get("convrot", False)):
                scales["convrot"] = True
                scales["convrot_groupsize"] = int(layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256)))
        elif module.quant_format in INT4_CONVROT_FORMATS:
            scale = pop_scale("weight_scale")
            if scale is None:
                raise ValueError(f"Missing INT4 ConvRot weight scale for layer {layer_name}")
            scales = _int4_convrot_params(
                layer_name=layer_name,
                quant_format=module.quant_format,
                layer_conf=layer_conf,
                weight=weight,
                scale=scale,
                orig_shape=module._orig_shape,
            )
        else:
            raise ValueError(f"Unsupported quantization format: {module.quant_format}")

        params = layout_cls.Params(**scales, orig_dtype=compute_dtype, orig_shape=module._orig_shape)
        module.weight = torch.nn.Parameter(
            QuantizedTensor(weight.to(device=device, dtype=qconfig["storage_t"]), module.layout_type, params),
            requires_grad=False,
        )
        refresh_runtime_args = getattr(module, "_refresh_int4_convrot_runtime_args", None)
        if refresh_runtime_args is not None:
            refresh_runtime_args()

        if load_extra_params:
            for param_name in qconfig["parameters"]:
                if param_name in {"weight_scale", "weight_scale_2"}:
                    continue
                param_key = f"{prefix}{param_name}"
                _v = state_dict.pop(param_key, None)
                if _v is None:
                    continue
                module.register_parameter(param_name, torch.nn.Parameter(_v.to(device=device), requires_grad=False))
                manually_loaded_keys.append(param_key)

    super_load(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    for key in manually_loaded_keys:
        if key in missing_keys:
            missing_keys.remove(key)


def _quantized_weight_state_dict(module: torch.nn.Module, sd: dict[str, torch.Tensor], prefix: str, extra_quant_conf: dict = None, extra_quant_params: tuple[str] = ()):
    if not hasattr(module, "weight"):
        logger.warning(f"uninitialized op {prefix}")
        return sd
    bias = getattr(module, "bias", None)
    if bias is not None:
        sd[f"{prefix}bias"] = bias
    if module.weight is None:
        return sd

    if not isinstance(module.weight, QuantizedTensor):
        sd[f"{prefix}weight"] = module.weight
    else:
        sd.update(module.weight.state_dict(f"{prefix}weight"))
        quant_conf = {"format": module.quant_format}
        if getattr(module, "_full_precision_mm_config", False):
            quant_conf["full_precision_matrix_mult"] = True
        params = getattr(module.weight, "_params", None)
        if module.quant_format == "int8_tensorwise" and getattr(params, "convrot", False):
            quant_conf["convrot"] = True
            quant_conf["convrot_groupsize"] = getattr(params, "convrot_groupsize", 256)
        elif module.quant_format in INT4_CONVROT_FORMATS:
            if module.quant_format == "int4_tensorwise":
                quant_conf["convrot"] = True
            quant_conf["convrot_groupsize"] = getattr(params, "convrot_groupsize", 256)
            quant_group_size = getattr(params, "quant_group_size", INT4_QUANT_GROUP_SIZE)
            if quant_group_size != INT4_QUANT_GROUP_SIZE:
                quant_conf["quant_group_size"] = quant_group_size
            linear_dtype = getattr(params, "linear_dtype", "int4")
            if linear_dtype != "int4":
                quant_conf["linear_dtype"] = linear_dtype
        if extra_quant_conf:
            quant_conf.update(extra_quant_conf)
        sd[f"{prefix}comfy_quant"] = torch.tensor(list(json.dumps(quant_conf).encode("utf-8")), dtype=torch.uint8)
        for name in extra_quant_params:
            value = getattr(module, name, None)
            if value is not None:
                sd[f"{prefix}{name}"] = value

    return sd


def mixed_precision_ops(quant_config={}, compute_dtype=torch.bfloat16, full_precision_mm=False, disabled=[]):
    class MixedPrecisionOps(ForgeOperations):
        _quant_config = quant_config
        _compute_dtype = compute_dtype
        _full_precision_mm = full_precision_mm
        _disabled = disabled

        class Linear(torch.nn.Module, ForgeWeights):
            _disabled_formats = disabled

            def __setattr__(self, name, value):
                super().__setattr__(name, value)
                # Catch patcher replacement/restoration of the Parameter.  The
                # runtime aliases contain no additional tensor storage.
                if name == "weight" and "_parameters" in self.__dict__:
                    self._refresh_int4_convrot_runtime_args()

            def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
                super().__init__()

                self.factory_kwargs = {"device": device, "dtype": MixedPrecisionOps._compute_dtype}

                self.in_features = in_features
                self.out_features = out_features
                self._orig_shape = (out_features, in_features)
                if bias:
                    self.bias = torch.nn.Parameter(torch.empty(out_features, **self.factory_kwargs))
                else:
                    self.register_parameter("bias", None)

                self._full_precision_mm = MixedPrecisionOps._full_precision_mm
                self._full_precision_mm_config = False

            def reset_parameters(self):
                return None

            def _refresh_int4_convrot_runtime_args(self):
                weight = self.__dict__.get("_parameters", {}).get("weight")
                if isinstance(weight, QuantizedTensor) and getattr(self, "quant_format", None) in INT4_CONVROT_FORMATS:
                    params = weight._params
                    self._int4_convrot_qweight = weight._qdata
                    self._int4_convrot_scale = params.scale
                    self._int4_convrot_groupsize = params.convrot_groupsize
                    self._int4_quant_group_size = params.quant_group_size
                    self._int4_linear_dtype = params.linear_dtype
                else:
                    self._int4_convrot_qweight = None
                    self._int4_convrot_scale = None

            def _load_from_state_dict(self, *args):
                _load_quantized_module(self, super()._load_from_state_dict, *args, load_extra_params=True)

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                sd = destination if destination is not None else {}
                return _quantized_weight_state_dict(self, sd, prefix, extra_quant_params=("input_scale",))

            def forward(self, input, *args, **kwargs):
                input_shape = input.shape
                reshaped_nd = False

                # TODO: Add a factorized online LoRA path for INT4 ConvRot.
                # Until then weight functions deliberately use Forge's correct,
                # but slower, dequantized fallback below.
                _use_quantized = getattr(self, "layout_type", None) is not None and not isinstance(input, QuantizedTensor) and not self._full_precision_mm and not getattr(self, "forge_force_cast_weights", False) and len(self.weight_function) == 0 and len(self.bias_function) == 0
                quantize_input = QUANT_ALGOS.get(getattr(self, "quant_format", None), {}).get("quantize_input", True)

                assert not input.requires_grad

                if _use_quantized and quantize_input:
                    input_reshaped = input.reshape(-1, input_shape[-1]) if input.ndim >= 3 else input

                    if input_reshaped.ndim == 2:
                        reshaped_nd = input.ndim >= 3
                        scale = getattr(self, "input_scale", None)
                        if scale is not None:
                            scale = cast_to_device(scale, input.device, None)
                        input = QuantizedTensor.from_float(input_reshaped, self.layout_type, scale=scale)

                _double_cast = self.parameters_manual_cast and (len(self.weight_function) > 0 or len(self.bias_function) > 0)

                if TRITON_AVAILABLE and getattr(self, "quant_format", None) == "int8_tensorwise" and not (_double_cast or self._full_precision_mm):
                    if len(self.weight_function) > 0 or len(self.bias_function) > 0:
                        _weight, bias, signal = weights_manual_cast(self, x=None, dtype=self.weight.dtype, device=input.device, bias_dtype=input.dtype)
                        weight, params = TensorWiseINT8Layout.quantize(
                            tensor=_weight,
                            scale="recalculate",
                            is_weight=True,
                            per_channel=True,
                            convrot=getattr(self.weight.params, "convrot", False),
                            convrot_groupsize=getattr(self.weight.params, "convrot_groupsize", 256),
                        )
                        scale: torch.Tensor = params.scale.to(device=input.device, non_blocking=True)
                    elif self.parameters_manual_cast:
                        weight, bias, signal = weights_manual_cast(self, x=None, dtype=torch.int8, device=input.device, bias_dtype=input.dtype)
                        scale: torch.Tensor = self.weight.params.scale.to(device=input.device, non_blocking=True)
                    else:
                        weight, bias, signal = self.weight._qdata, self.bias, None
                        scale: torch.Tensor = self.weight.params.scale.to(device=input.device, non_blocking=True)

                    if getattr(self.weight.params, "convrot", False):
                        group_size: int = getattr(self.weight.params, "convrot_groupsize", 256)
                        H = build_hadamard(group_size, device=input.device, dtype=input.dtype)
                        input = rotate_activation(input, H, group_size=group_size)

                    compute_dtype: torch.dtype = input.dtype if input.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

                    with main_stream_worker(weight, bias, signal):
                        if self._per_row:
                            output = triton_int8_linear_per_row(input, weight, scale, bias, compute_dtype)
                        else:
                            output = triton_int8_linear(input, weight, scale, bias, compute_dtype)
                else:
                    weight_only_quant = _use_quantized and not quantize_input and isinstance(self.weight, QuantizedTensor)

                    int4_convrot_fast_path = weight_only_quant and getattr(self, "quant_format", None) in INT4_CONVROT_FORMATS

                    if int4_convrot_fast_path:
                        # Plain tensor aliases keep comfy-kitchen's Params
                        # dataclass and QuantizedTensor subclass out of the
                        # Dynamo graph. In low-VRAM mode copy the two actual
                        # kernel operands directly on the current stream.
                        weight = cast_to_device(self._int4_convrot_qweight, input.device, torch.int8)
                        scale = cast_to_device(self._int4_convrot_scale, input.device, torch.float32)
                        bias = None if self.bias is None else cast_to_device(self.bias, input.device, input.dtype)
                        signal = None
                    elif weight_only_quant:
                        weight, bias, signal = weights_manual_cast(self, x=None, dtype=self.weight.dtype, device=input.device, bias_dtype=input.dtype)
                        weight = weight.to(dtype=input.dtype)
                    else:
                        weight, bias, signal = weights_manual_cast(self, x=input)

                    with main_stream_worker(weight, bias, signal):
                        if int4_convrot_fast_path:
                            output = convrot_w4a4_linear_compile_safe(
                                input,
                                weight,
                                scale,
                                bias,
                                self._int4_convrot_groupsize,
                                self._int4_quant_group_size,
                                self._int4_linear_dtype,
                            )
                        else:
                            output = torch.nn.functional.linear(input, weight, bias)

                if reshaped_nd:
                    output = output.reshape((*input_shape[:-1], self.weight.shape[0]))

                return output

            def convert_weight(self, weight, inplace=False, **kwargs):
                if isinstance(weight, QuantizedTensor):
                    return weight.dequantize()
                else:
                    return weight

            def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
                if getattr(self, "layout_type", None) is not None:
                    weight = self.weight.requantize_from_float(weight, scale="recalculate", stochastic_rounding=seed, inplace_ops=True).to(self.weight.dtype)
                else:
                    weight = weight.to(self.weight.dtype)
                if return_weight:
                    return weight

                assert inplace_update is False
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

            def _apply(self, fn, recurse=True):
                result = _quantized_apply(self, fn, recurse)
                self._refresh_int4_convrot_runtime_args()
                return result

        class Embedding(ForgeOperations.Embedding):
            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                weight_key = f"{prefix}weight"
                layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
                if layer_conf is not None:
                    layer_conf = json.loads(layer_conf.numpy().tobytes())

                quant_format = layer_conf.get("format") if layer_conf is not None else None
                manually_loaded_keys = []

                if quant_format in ("float8_e4m3fn", "float8_e5m2") and weight_key in state_dict:
                    self.quant_format = quant_format
                    qconfig = QUANT_ALGOS[quant_format]
                    self.layout_type = qconfig["comfy_tensor_layout"]
                    layout_cls = get_layout_class(self.layout_type)
                    weight = state_dict.pop(weight_key)
                    manually_loaded_keys.append(weight_key)

                    scale_key = f"{prefix}weight_scale"
                    scale = state_dict.pop(scale_key, None)
                    if scale is not None:
                        scale = scale.float()
                        manually_loaded_keys.append(scale_key)

                    params = layout_cls.Params(
                        scale=scale if scale is not None else torch.ones((), dtype=torch.float32),
                        orig_dtype=MixedPrecisionOps._compute_dtype,
                        orig_shape=(self.num_embeddings, self.embedding_dim),
                    )
                    self.weight = torch.nn.Parameter(QuantizedTensor(weight.to(dtype=qconfig["storage_t"]), qconfig["comfy_tensor_layout"], params), requires_grad=False)
                elif layer_conf is not None:
                    state_dict[f"{prefix}comfy_quant"] = torch.tensor(list(json.dumps(layer_conf).encode("utf-8")), dtype=torch.uint8)

                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

                for k in manually_loaded_keys:
                    if k in missing_keys:
                        missing_keys.remove(k)

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                sd = destination if destination is not None else {}
                return _quantized_weight_state_dict(self, sd, prefix)

            def forward(self, input):
                weight = self.weight

                if isinstance(weight, QuantizedTensor) and len(self.weight_function) == 0:
                    qdata, _, signal = weights_manual_cast(self, device=input.device, dtype=weight.dtype)
                    if isinstance(qdata, QuantizedTensor):
                        scale = qdata._params.scale
                        qdata = qdata._qdata
                    else:
                        scale = None

                    with main_stream_worker(qdata, None, signal):
                        x = torch.nn.functional.embedding(input, qdata, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

                    target_dtype = weight._params.orig_dtype
                    x = x.to(dtype=target_dtype)
                    if scale is not None and scale != 1.0:
                        x = x * scale.to(dtype=target_dtype)

                    return x

                return super().forward(input)

    return MixedPrecisionOps
