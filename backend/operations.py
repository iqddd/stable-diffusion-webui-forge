# Copyright Forge 2024

import time
import torch
import torch.nn as nn
from torch import __version__
import contextlib

from backend import stream, memory_management, utils
from backend.patcher.lora import merge_lora_to_weight

stash = []


def get_weight_and_bias(layer, weight_args=None, bias_args=None, weight_fn=None, bias_fn=None):
    scale_weight = getattr(layer, 'scale_weight', None)
    patches = getattr(layer, 'forge_online_loras', None)
    weight_patches, bias_patches = None, None

    if patches is not None:
        weight_patches = patches.get('weight', None)

    if patches is not None:
        bias_patches = patches.get('bias', None)

    weight = None
    if layer.weight is not None:
        weight = layer.weight
        if weight_fn is not None:
            if weight_args is not None:
                fn_device = weight_args.get('device', None)
                if fn_device is not None:
                    weight = weight.to(device=fn_device)
            weight = weight_fn(weight)
        if weight_args is not None:
            weight = weight.to(**weight_args)
        if scale_weight is not None:
            weight = weight*scale_weight.to(device=weight.device, dtype=weight.dtype)
        if weight_patches is not None:
            weight = merge_lora_to_weight(patches=weight_patches, weight=weight, key="online weight lora", computation_dtype=weight.dtype)

    bias = None
    if layer.bias is not None:
        bias = layer.bias
        if bias_fn is not None:
            if bias_args is not None:
                fn_device = bias_args.get('device', None)
                if fn_device is not None:
                    bias = bias.to(device=fn_device)
            bias = bias_fn(bias)
        if bias_args is not None:
            bias = bias.to(**bias_args)
        if bias_patches is not None:
            bias = merge_lora_to_weight(patches=bias_patches, weight=bias, key="online bias lora", computation_dtype=bias.dtype)
    return weight, bias


def weights_manual_cast(layer, x, skip_weight_dtype=False, skip_bias_dtype=False, weight_fn=None, bias_fn=None):
    weight, bias, signal = None, None, None
    non_blocking = True

    if getattr(x.device, 'type', None) == 'mps':
        non_blocking = False

    target_dtype = x.dtype
    target_device = x.device

    if skip_weight_dtype:
        weight_args = dict(device=target_device, non_blocking=non_blocking)
    else:
        weight_args = dict(device=target_device, dtype=target_dtype, non_blocking=non_blocking)

    if skip_bias_dtype:
        bias_args = dict(device=target_device, non_blocking=non_blocking)
    else:
        bias_args = dict(device=target_device, dtype=target_dtype, non_blocking=non_blocking)

    if stream.should_use_stream():
        with stream.stream_context()(stream.mover_stream):
            weight, bias = get_weight_and_bias(layer, weight_args, bias_args, weight_fn=weight_fn, bias_fn=bias_fn)
            signal = stream.mover_stream.record_event()
    else:
        weight, bias = get_weight_and_bias(layer, weight_args, bias_args, weight_fn=weight_fn, bias_fn=bias_fn)

    return weight, bias, signal



@contextlib.contextmanager
def main_stream_worker(weight, bias, signal):
    global stash

    if signal is None or not stream.should_use_stream():
        yield
        return

    with stream.stream_context()(stream.current_stream):
        stream.current_stream.wait_event(signal)
        yield
        finished_signal = stream.current_stream.record_event()
        stash.append((id(finished_signal), weight, bias, finished_signal))
        # stash[id(finished_signal)] = (weight, bias, finished_signal)

    if len(stash) > 50:
        stash[25][3].synchronize()
    stash = [item for item in stash if not item[3].query()]


def cleanup_cache():
    if not stream.should_use_stream():
        return

    stream.current_stream.synchronize()
    stream.mover_stream.synchronize()
    stash.clear()
    return


current_device = None
current_dtype = None
current_manual_cast_enabled = False
current_bnb_dtype = None
current_fp8_mode = None

IS_TORCH_2_4 = __version__ < (2, 4, 9)

class ForgeOperations:
    class Linear(torch.nn.Module):
        def __init__(self, in_features, out_features, float_weight = None, float_bias = None, *args, **kwargs):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.dummy = torch.nn.Parameter(torch.empty(1, device=current_device, dtype=current_dtype))
            self.weight = None
            self.scale_weight = None
            self.bias = None
            self.parameters_manual_cast = current_manual_cast_enabled
            self.fp8_mode = current_fp8_mode

            if current_fp8_mode:
                self.float8_dtype = torch.float8_e4m3fn
                self.input_float8_dtype = torch.float8_e5m2
                self.input_scale_initialized = False
                self.weight_initialized = False
                self.max_value = torch.finfo(self.float8_dtype).max
                self.input_max_value = torch.finfo(self.input_float8_dtype).max
                # factory_kwargs = {"dtype": current_dtype, "device": current_device}
                # if float_weight is None:
                #     self.weight = nn.Parameter(
                #         torch.empty((out_features, in_features), **factory_kwargs)
                #     )
                # else:
                #     self.weight = nn.Parameter(
                #         float_weight, requires_grad=float_weight.requires_grad
                #     )
                # if float_bias is None:
                #     if bias:
                #         self.bias = nn.Parameter(
                #             torch.empty(out_features, **factory_kwargs),
                #         )
                #     else:
                #         self.register_parameter("bias", None)
                # else:
                #     self.bias = nn.Parameter(float_bias, requires_grad=float_bias.requires_grad)
                self.num_scale_trials = 12
                self.register_buffer(
                    "input_amax_trials",
                    torch.zeros(
                        12, requires_grad=False, device=current_device, dtype=torch.float32
                    ),
                )
                self.trial_index = 0
                self.register_buffer("scale", None)
                self.register_buffer(
                    "input_scale",
                    None,
                )
                self.register_buffer(
                    "float8_data",
                    None,
                )
                self.scale_reciprocal = self.register_buffer("scale_reciprocal", None)
                self.input_scale_reciprocal = self.register_buffer(
                    "input_scale_reciprocal", None
                )

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if hasattr(self, 'dummy'):
                if prefix + 'weight' in state_dict:
                    self.weight = torch.nn.Parameter(state_dict[prefix + 'weight'].to(self.dummy))
                if prefix + 'scale_weight' in state_dict:
                     self.scale_weight = torch.nn.Parameter(state_dict[prefix + 'scale_weight'])                    
                if prefix + 'bias' in state_dict:
                    self.bias = torch.nn.Parameter(state_dict[prefix + 'bias'].to(self.dummy))
                del self.dummy

                if self.fp8_mode:
                    self.quantize_weight()
            else:
                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        def quantize_weight(self):
            if self.weight_initialized:
                return
            amax = torch.max(torch.abs(self.weight.data)).float()
            self.scale = self.amax_to_scale(amax, self.max_value)
            self.float8_data = self.to_fp8_saturated(
                self.weight.data, self.scale, self.max_value
            ).to(self.float8_dtype)
            self.scale_reciprocal = self.scale.reciprocal()
            self.weight.data = torch.zeros(
                1, dtype=self.weight.dtype, device=self.weight.device, requires_grad=False
            )
            self.weight_initialized = True

        def amax_to_scale(self, amax, max_val):
            return (max_val / torch.clamp(amax, min=1e-12)).clamp(max=max_val)

        def to_fp8_saturated(self, x, scale, max_val):
            return (x * scale).clamp(-max_val, max_val)
        
        def quantize_input(self, x: torch.Tensor):
            if self.input_scale_initialized:
                return self.to_fp8_saturated(x, self.input_scale, self.input_max_value).to(
                    self.input_float8_dtype
                )
            elif self.trial_index < self.num_scale_trials:

                amax = torch.max(torch.abs(x)).float()

                self.input_amax_trials[self.trial_index] = amax
                self.trial_index += 1
                self.input_scale = self.amax_to_scale(
                    self.input_amax_trials[: self.trial_index].max(), self.input_max_value
                )
                self.input_scale_reciprocal = self.input_scale.reciprocal()
                return self.to_fp8_saturated(x, self.input_scale, self.input_max_value).to(
                    self.input_float8_dtype
                )
            else:
                self.input_scale = self.amax_to_scale(
                    self.input_amax_trials.max(), self.input_max_value
                )
                self.input_scale_reciprocal = self.input_scale.reciprocal()
                self.input_scale_initialized = True
                return self.to_fp8_saturated(x, self.input_scale, self.input_max_value).to(
                    self.input_float8_dtype
                )
        
        def fp8_forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.input_scale_initialized:
                x = self.to_fp8_saturated(x, self.input_scale, self.input_max_value).to(
                    self.input_float8_dtype
                )
            else:
                x = self.quantize_input(x)

            prev_dims = x.shape[:-1]
            x = x.view(-1, self.in_features)

            # float8 matmul, much faster than float16 matmul w/ float32 accumulate on ADA devices!
            out = torch._scaled_mm(
                x,
                self.float8_data.T,
                scale_a=self.input_scale_reciprocal,
                scale_b=self.scale_reciprocal,
                bias=self.bias,
                out_dtype=self.weight.dtype,
                use_fast_accum=True,
            )
            if IS_TORCH_2_4:
                out = out[0]
            out = out.view(*prev_dims, self.out_features)
            return out

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    if self.fp8_mode:
                        return self.fp8_forward(x)
                    else:
                        return torch.nn.functional.linear(x, weight, bias)
            else:
                weight, bias = get_weight_and_bias(self)
                if self.fp8_mode:
                    return self.fp8_forward(x)
                else:
                    return torch.nn.functional.linear(x, weight, bias)

    class Conv2d(torch.nn.Conv2d):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return self._conv_forward(x, weight, bias)
            else:
                weight, bias = get_weight_and_bias(self)
                return super()._conv_forward(x, weight, bias)

    class Conv3d(torch.nn.Conv3d):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return self._conv_forward(x, weight, bias)
            else:
                weight, bias = get_weight_and_bias(self)
                return super()._conv_forward(input, weight, bias)

    class Conv1d(torch.nn.Conv1d):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return self._conv_forward(x, weight, bias)
            else:
                weight, bias = get_weight_and_bias(self)
                return super()._conv_forward(input, weight, bias)

    class ConvTranspose2d(torch.nn.ConvTranspose2d):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x, output_size=None):
            if self.parameters_manual_cast:
                num_spatial_dims = 2
                output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims, self.dilation)

                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.conv_transpose2d(x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
            else:
                weight, bias = get_weight_and_bias(self)
                num_spatial_dims = 2
                output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims, self.dilation)
                return torch.nn.functional.conv_transpose2d(x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

    class ConvTranspose1d(torch.nn.ConvTranspose1d):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x, output_size=None):
            if self.parameters_manual_cast:
                num_spatial_dims = 1
                output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims, self.dilation)

                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.conv_transpose1d(x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
            else:
                weight, bias = get_weight_and_bias(self)
                num_spatial_dims = 1
                output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims, self.dilation)
                return torch.nn.functional.conv_transpose2d(x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

    class ConvTranspose3d(torch.nn.ConvTranspose3d):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x, output_size=None):
            if self.parameters_manual_cast:
                num_spatial_dims = 3
                output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims, self.dilation)

                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.conv_transpose3d(x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
            else:
                weight, bias = get_weight_and_bias(self)
                num_spatial_dims = 3
                output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims, self.dilation)
                return torch.nn.functional.conv_transpose2d(x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

    class GroupNorm(torch.nn.GroupNorm):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.group_norm(x, self.num_groups, weight, bias, self.eps)
            else:
                return super().forward(x)

    class LayerNorm(torch.nn.LayerNorm):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
            else:
                return super().forward(x)

    class Embedding(torch.nn.Embedding):

        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled
            self.bias = None

        def reset_parameters(self):
            self.bias = None
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x, skip_weight_dtype=True, skip_bias_dtype=True)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.embedding(x, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
            else:
                return super().forward(x)


def weights_manual_cast_qlora(layer, x):
    target_dtype = x.dtype
    target_device = x.device

    weight_args = dict(device=target_device, dtype=target_dtype, non_blocking=True)
    bias_args = dict(device=target_device, non_blocking=True)

    if stream.should_use_stream():
        with stream.stream_context()(stream.mover_stream):
            weight, bias, lora_params = get_weight_and_bias_qlora(layer, weight_args, bias_args)
            signal = stream.mover_stream.record_event()
    else:
        weight, bias, lora_params = get_weight_and_bias_qlora(layer, weight_args, bias_args)

    return weight, bias, lora_params, signal


def get_weight_and_bias_qlora(layer, weight_args=None, bias_args=None):
    patches = getattr(layer, 'forge_online_loras', None)
    weight_patches = patches.get('weight', None)

    lora_params = None
    if layer.weight is None:
        raise "Not implemented"

    # weight = layer.weight.to(**weight_args)

    p = weight_patches[0]
    strength = p[0]
    v = p[1][1]
    mat1 = memory_management.cast_to_device(v[0], weight_args["device"], weight_args["dtype"])
    mat2 = memory_management.cast_to_device(v[1], weight_args["device"], weight_args["dtype"])
    if v[2] is not None:
        alpha = v[2] / mat2.shape[0]
    else:
        alpha = 1.0

    lora_params = [alpha * strength, mat1, mat2]

    bias = None
    if layer.bias is not None:
        bias = layer.bias
        bias = bias.to(**bias_args)

    return layer.weight, bias, lora_params


try:
    from backend.operations_bnb import ForgeLoader4Bit, ForgeParams4bit, functional_linear_4bits, functional_dequantize_4bit

    class ForgeOperationsBNB4bits(ForgeOperations):
        class Linear(ForgeLoader4Bit):
            def __init__(self, *args, **kwargs):
                super().__init__(device=current_device, dtype=current_dtype, quant_type=current_bnb_dtype)
                self.parameters_manual_cast = current_manual_cast_enabled

            def forward(self, x):
                if self.bias is not None and self.bias.dtype != x.dtype:
                    # Maybe this can also be set to all non-bnb ops since the cost is very low.
                    # And it only invokes one time, and most linear does not have bias
                    self.bias = utils.tensor2parameter(self.bias.to(x.dtype))

                if hasattr(self, 'forge_online_loras'):
                    weight, bias, lora_params, signal = weights_manual_cast_qlora(self, x)
                    with main_stream_worker(weight, bias, signal):
                        output_base = functional_linear_4bits(x, weight, bias)
                        temp_lora_activation = x @ lora_params[2].T
                        output_lora = temp_lora_activation @ lora_params[1].T
                        output_base.add_(output_lora, alpha=lora_params[0])
                        return output_base
                        # return torch.nn.functional.linear(x, weight, bias)

                if not self.parameters_manual_cast:
                    return functional_linear_4bits(x, self.weight, self.bias)
                elif not self.weight.bnb_quantized:
                    assert x.device.type == 'cuda', 'BNB Must Use CUDA as Computation Device!'
                    layer_original_device = self.weight.device
                    self.weight = self.weight._quantize(x.device)
                    bias = self.bias.to(x.device) if self.bias is not None else None
                    out = functional_linear_4bits(x, self.weight, bias)
                    self.weight = self.weight.to(layer_original_device)
                    return out
                else:
                    weight, bias, signal = weights_manual_cast(self, x, skip_weight_dtype=True, skip_bias_dtype=True)
                    with main_stream_worker(weight, bias, signal):
                        return functional_linear_4bits(x, weight, bias)

    bnb_avaliable = True
except:
    bnb_avaliable = False


from backend.operations_gguf import dequantize_tensor


class ForgeOperationsGGUF(ForgeOperations):
    class Linear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.empty(1, device=current_device, dtype=current_dtype))
            self.weight = None
            self.bias = None
            self.parameters_manual_cast = current_manual_cast_enabled

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if hasattr(self, 'dummy'):
                computation_dtype = self.dummy.dtype
                if computation_dtype not in [torch.float16, torch.bfloat16]:
                    # GGUF cast only supports 16bits otherwise super slow
                    computation_dtype = torch.float16
                if prefix + 'weight' in state_dict:
                    self.weight = state_dict[prefix + 'weight'].to(device=self.dummy.device)
                    self.weight.computation_dtype = computation_dtype
                if prefix + 'bias' in state_dict:
                    self.bias = state_dict[prefix + 'bias'].to(device=self.dummy.device)
                    self.bias.computation_dtype = computation_dtype
                del self.dummy
            else:
                if prefix + 'weight' in state_dict:
                    self.weight = state_dict[prefix + 'weight']
                if prefix + 'bias' in state_dict:
                    self.bias = state_dict[prefix + 'bias']
            return

        def _apply(self, fn, recurse=True):
            for k, p in self.named_parameters(recurse=False, remove_duplicate=True):
                setattr(self, k, utils.tensor2parameter(fn(p)))
            return self

        def forward(self, x):
            if self.bias is not None and self.bias.dtype != x.dtype:
                self.bias = utils.tensor2parameter(dequantize_tensor(self.bias).to(x.dtype))

            if self.weight is not None and self.weight.dtype != x.dtype and getattr(self.weight, 'gguf_cls', None) is None:
                self.weight = utils.tensor2parameter(self.weight.to(x.dtype))

            weight, bias, signal = weights_manual_cast(self, x, weight_fn=dequantize_tensor, bias_fn=None, skip_bias_dtype=True)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.linear(x, weight, bias)


@contextlib.contextmanager
def using_forge_operations(operations=None, device=None, dtype=None, manual_cast_enabled=False, bnb_dtype=None, fp8_mode=None):
    global current_device, current_dtype, current_manual_cast_enabled, current_bnb_dtype, current_fp8_mode

    current_device, current_dtype, current_manual_cast_enabled, current_bnb_dtype, current_fp8_mode = device, dtype, manual_cast_enabled, bnb_dtype, fp8_mode

    if operations is None:
        if bnb_dtype in ['gguf']:
            operations = ForgeOperationsGGUF
        elif bnb_avaliable and bnb_dtype in ['nf4', 'fp4']:
            operations = ForgeOperationsBNB4bits
        else:
            operations = ForgeOperations

    op_names = ['Linear', 'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d', 'GroupNorm', 'LayerNorm', 'Embedding']
    backups = {op_name: getattr(torch.nn, op_name) for op_name in op_names}

    try:
        for op_name in op_names:
            setattr(torch.nn, op_name, getattr(operations, op_name))

        yield

    finally:
        for op_name in op_names:
            setattr(torch.nn, op_name, backups[op_name])
    return


def shift_manual_cast(model, enabled):
    for m in model.modules():
        if hasattr(m, 'parameters_manual_cast'):
            m.parameters_manual_cast = enabled
    return


@contextlib.contextmanager
def automatic_memory_management():
    memory_management.free_memory(
        memory_required=3 * 1024 * 1024 * 1024,
        device=memory_management.get_torch_device()
    )

    module_list = []

    original_init = torch.nn.Module.__init__
    original_to = torch.nn.Module.to

    def patched_init(self, *args, **kwargs):
        module_list.append(self)
        return original_init(self, *args, **kwargs)

    def patched_to(self, *args, **kwargs):
        module_list.append(self)
        return original_to(self, *args, **kwargs)

    try:
        torch.nn.Module.__init__ = patched_init
        torch.nn.Module.to = patched_to
        yield
    finally:
        torch.nn.Module.__init__ = original_init
        torch.nn.Module.to = original_to

    start = time.perf_counter()
    module_list = set(module_list)

    for module in module_list:
        module.cpu()

    memory_management.soft_empty_cache()
    end = time.perf_counter()

    print(f'Automatic Memory Management: {len(module_list)} Modules in {(end - start):.2f} seconds.')
    return


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, target_device: torch.device):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(target_device), requires_grad=p.requires_grad)
                    else:
                        return p.to(target_device)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(target_device)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

        return

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')
        return

    @staticmethod
    def install_model(model: torch.nn.Module, target_device: torch.device):
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, target_device)
        return

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)
        return

