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
    get_layout_class,
)

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


_INT8_CONVROT_LORA_CACHE_PREFIX = "_forge_int8_convrot_lora_"


def _tensor_cache_version(tensor: torch.Tensor):
    try:
        return tensor._version
    except RuntimeError:
        # Tensors created in inference_mode are immutable but have no version counter.
        return None


@torch.compiler.disable
def _prepare_int8_convrot_online_lora(linear: torch.nn.Linear, x: torch.Tensor, group_size: int):
    """Prepare additive classic LoRAs without materializing their full weight deltas."""
    # Runtime imports avoid the operations/model-patcher import cycle.
    from backend.patcher.base import WeightPatch
    from modules_forge.packages.comfy.weight_adapter.lora import LoRAAdapter

    def unsupported():
        setattr(linear, f"{_INT8_CONVROT_LORA_CACHE_PREFIX}supported", False)
        for suffix in ("matrix_key", "scale_key", "A", "B", "scales"):
            setattr(linear, f"{_INT8_CONVROT_LORA_CACHE_PREFIX}{suffix}", None)
        return None, None, None, False

    weight_functions = tuple(getattr(linear, "weight_function", ()))
    if not weight_functions:
        setattr(linear, f"{_INT8_CONVROT_LORA_CACHE_PREFIX}supported", True)
        return None, None, None, True
    if tuple(getattr(linear, "bias_function", ())):
        return unsupported()

    adapters = []
    matrix_signature = []
    scale_signature = []

    for weight_patch in weight_functions:
        if not isinstance(weight_patch, WeightPatch):
            return unsupported()

        patches = weight_patch.patches.get(weight_patch.key)
        if not isinstance(patches, list):
            return unsupported()

        for patch in patches:
            if not isinstance(patch, (tuple, list)) or len(patch) < 6:
                return unsupported()

            strength, adapter, strength_model, offset, function, online = patch[:6]
            if not online or float(strength_model) != 1.0 or offset is not None or function is not None:
                return unsupported()
            if not isinstance(adapter, LoRAAdapter):
                return unsupported()

            up, down, alpha, mid, dora_scale, reshape = adapter.weights
            if mid is not None or dora_scale is not None or reshape is not None:
                return unsupported()
            if not isinstance(up, torch.Tensor) or not isinstance(down, torch.Tensor):
                return unsupported()
            if up.ndim != 2 or down.ndim != 2:
                return unsupported()
            if up.shape[1] != down.shape[0] or down.shape[1] != linear.in_features or up.shape[0] != linear.out_features:
                return unsupported()
            if down.shape[1] % group_size != 0:
                return unsupported()

            rank = down.shape[0]
            if rank == 0:
                return unsupported()
            alpha_value = None if alpha is None else float(alpha)
            gamma = float(strength) * ((alpha_value / rank) if alpha_value is not None else 1.0)
            adapters.append((up, down, rank, gamma))
            matrix_signature.append(
                (
                    id(adapter),
                    id(up),
                    _tensor_cache_version(up),
                    tuple(up.shape),
                    id(down),
                    _tensor_cache_version(down),
                    tuple(down.shape),
                )
            )
            scale_signature.append((rank, gamma))

    matrix_cache_key = (tuple(matrix_signature), x.device, x.dtype, group_size)
    if getattr(linear, f"{_INT8_CONVROT_LORA_CACHE_PREFIX}matrix_key", None) != matrix_cache_key:
        if adapters:
            all_down = [cast_to_device(down, x.device, x.dtype) for _, down, _, _ in adapters]
            all_up = [cast_to_device(up, x.device, x.dtype) for up, _, _, _ in adapters]
            lora_a = torch.cat(all_down, dim=0) if len(all_down) > 1 else all_down[0]
            lora_b = torch.cat(all_up, dim=1) if len(all_up) > 1 else all_up[0]
            H = build_hadamard(group_size, device=x.device, dtype=x.dtype)
            lora_a = rotate_activation(lora_a, H, group_size=group_size)
        else:
            lora_a = None
            lora_b = None

        setattr(linear, f"{_INT8_CONVROT_LORA_CACHE_PREFIX}matrix_key", matrix_cache_key)
        setattr(linear, f"{_INT8_CONVROT_LORA_CACHE_PREFIX}A", lora_a)
        setattr(linear, f"{_INT8_CONVROT_LORA_CACHE_PREFIX}B", lora_b)

    scale_cache_key = (tuple(scale_signature), x.device)
    if getattr(linear, f"{_INT8_CONVROT_LORA_CACHE_PREFIX}scale_key", None) != scale_cache_key:
        # TODO(torch.compile): when ranks are unchanged, update this tensor in-place.
        # Replacing the module attribute currently recompiles on strength-only changes.
        if adapters:
            rank_scales = torch.cat(
                [torch.full((rank,), gamma, device=x.device, dtype=torch.float32) for _, _, rank, gamma in adapters]
            )
        else:
            rank_scales = None
        setattr(linear, f"{_INT8_CONVROT_LORA_CACHE_PREFIX}scale_key", scale_cache_key)
        setattr(linear, f"{_INT8_CONVROT_LORA_CACHE_PREFIX}scales", rank_scales)

    setattr(linear, f"{_INT8_CONVROT_LORA_CACHE_PREFIX}supported", True)
    return (
        getattr(linear, f"{_INT8_CONVROT_LORA_CACHE_PREFIX}A"),
        getattr(linear, f"{_INT8_CONVROT_LORA_CACHE_PREFIX}B"),
        getattr(linear, f"{_INT8_CONVROT_LORA_CACHE_PREFIX}scales"),
        True,
    )


def prepare_int8_convrot_online_lora_for_compile(
    model: torch.nn.Module,
    x: torch.Tensor,
    compute_dtype: torch.dtype | None = None,
):
    """Prepare online-LoRA tensors before entering a torch.compile'd diffusion model."""
    if not TRITON_AVAILABLE or not isinstance(x, torch.Tensor):
        return

    compute_dtype = x.dtype if compute_dtype is None else compute_dtype
    if compute_dtype is not torch.bfloat16:
        return

    cache_spec = x if x.dtype is compute_dtype else torch.empty((), device=x.device, dtype=compute_dtype)

    for module in model.modules():
        weight = getattr(module, "weight", None)
        weight_params = getattr(weight, "params", None)
        if (
            getattr(module, "quant_format", None) == "int8_tensorwise"
            and getattr(weight_params, "convrot", False)
            and not getattr(module, "_full_precision_mm", True)
            and not getattr(module, "forge_force_cast_weights", False)
            and len(getattr(module, "weight_function", ())) > 0
        ):
            group_size = getattr(weight_params, "convrot_groupsize", 256)
            _prepare_int8_convrot_online_lora(module, cache_spec, group_size)


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

        qconfig = QUANT_ALGOS[module.quant_format]
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
        elif module.quant_format == "convrot_w4a4":
            scale = pop_scale("weight_scale")
            if scale is None:
                raise ValueError(f"Missing ConvRot W4A4 weight scale for layer {layer_name}")
            params_conf = layer_conf.get("params", {})
            if not isinstance(params_conf, dict):
                params_conf = {}
            scales = {
                "scale": scale,
                "convrot_groupsize": int(layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))),
                "quant_group_size": 64,
                "linear_dtype": layer_conf.get("linear_dtype", params_conf.get("linear_dtype", "int4")),
            }
        else:
            raise ValueError(f"Unsupported quantization format: {module.quant_format}")

        params = layout_cls.Params(**scales, orig_dtype=compute_dtype, orig_shape=module._orig_shape)
        module.weight = torch.nn.Parameter(
            QuantizedTensor(weight.to(device=device, dtype=qconfig["storage_t"]), module.layout_type, params),
            requires_grad=False,
        )

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
        elif module.quant_format == "convrot_w4a4":
            quant_conf["convrot_groupsize"] = getattr(params, "convrot_groupsize", 256)
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

            def _load_from_state_dict(self, *args):
                _load_quantized_module(self, super()._load_from_state_dict, *args, load_extra_params=True)

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                sd = destination if destination is not None else {}
                return _quantized_weight_state_dict(self, sd, prefix, extra_quant_params=("input_scale",))

            def forward(self, input, *args, **kwargs):
                input_shape = input.shape
                reshaped_nd = False

                int8_convrot_lora = (None, None, None)
                int8_convrot_lora_supported = False
                weight_params = getattr(self.weight, "params", None)
                if (
                    TRITON_AVAILABLE
                    and getattr(self, "quant_format", None) == "int8_tensorwise"
                    and getattr(weight_params, "convrot", False)
                    and not self._full_precision_mm
                    and not getattr(self, "forge_force_cast_weights", False)
                    and not isinstance(input, QuantizedTensor)
                    and input.dtype is torch.bfloat16
                    and len(self.weight_function) > 0
                ):
                    group_size = getattr(weight_params, "convrot_groupsize", 256)
                    if torch.compiler.is_compiling():
                        int8_convrot_lora_supported = getattr(
                            self,
                            f"{_INT8_CONVROT_LORA_CACHE_PREFIX}supported",
                            False,
                        )
                        lora_a = getattr(self, f"{_INT8_CONVROT_LORA_CACHE_PREFIX}A", None)
                        lora_b = getattr(self, f"{_INT8_CONVROT_LORA_CACHE_PREFIX}B", None)
                        lora_scales = getattr(self, f"{_INT8_CONVROT_LORA_CACHE_PREFIX}scales", None)
                    else:
                        lora_a, lora_b, lora_scales, int8_convrot_lora_supported = _prepare_int8_convrot_online_lora(self, input, group_size)
                    int8_convrot_lora = (lora_a, lora_b, lora_scales)

                _use_quantized = getattr(self, "layout_type", None) is not None and not isinstance(input, QuantizedTensor) and not self._full_precision_mm and not getattr(self, "forge_force_cast_weights", False) and (len(self.weight_function) == 0 or int8_convrot_lora_supported) and len(self.bias_function) == 0
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

                _double_cast = self.parameters_manual_cast and ((len(self.weight_function) > 0 and not int8_convrot_lora_supported) or len(self.bias_function) > 0)

                if TRITON_AVAILABLE and getattr(self, "quant_format", None) == "int8_tensorwise" and not (_double_cast or self._full_precision_mm):
                    if int8_convrot_lora_supported:
                        if self.weight.device == input.device:
                            weight, bias, signal = self.weight._qdata, self.bias, None
                        else:
                            weight, bias, signal = weights_manual_cast(
                                self,
                                x=None,
                                dtype=torch.int8,
                                device=input.device,
                                bias_dtype=input.dtype,
                                apply_weight_functions=False,
                                apply_bias_functions=False,
                            )
                            if isinstance(weight, QuantizedTensor):
                                weight = weight._qdata
                        scale: torch.Tensor = self.weight.params.scale.to(device=input.device, non_blocking=True)
                    elif len(self.weight_function) > 0 or len(self.bias_function) > 0:
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

                    lora_a, lora_b, lora_scales = int8_convrot_lora
                    if int8_convrot_lora_supported and lora_a is not None:
                        lora_hidden = torch.nn.functional.linear(input, lora_a)
                        lora_hidden.mul_(lora_scales)
                        lora_output = torch.nn.functional.linear(lora_hidden, lora_b)
                        output.add_(lora_output.to(dtype=output.dtype))
                else:
                    weight_only_quant = _use_quantized and not quantize_input and isinstance(self.weight, QuantizedTensor)

                    if weight_only_quant:
                        weight, bias, signal = weights_manual_cast(self, x=None, dtype=self.weight.dtype, device=input.device, bias_dtype=input.dtype)
                        weight = weight.to(dtype=input.dtype)
                    else:
                        weight, bias, signal = weights_manual_cast(self, x=input)

                    with main_stream_worker(weight, bias, signal):
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
                return _quantized_apply(self, fn, recurse)

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
