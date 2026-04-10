# https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_torch_compile.py

import logging
from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.modules.k_model import KModel

import gradio as gr
import torch

from backend.args import args as cmd_args
from backend.logging import setup_logger
from backend.utils import get_attr, set_attr_raw
from modules import scripts

TRITON_AVAILABLE = find_spec("triton") is not None

_COMPILE_CONFIG_KEY = "_torch_compile_config"
_COMPILED_BACKUP_KEY = "_compiled_backup"
_MODEL_BACKUP_KEY = "_model_backup"

logger = logging.getLogger("compile")
setup_logger(logger)


def skip_torch_compile_dict(guard_entries):
    return [("transformer_options" not in entry.name) for entry in guard_entries]


class TorchCompileForForge(scripts.Script):
    sorting_priority = 99999

    def __init__(self):
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.suppress_errors = True

    def title(self):
        return "Torch Compile Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if TRITON_AVAILABLE else None

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            preset = gr.Dropdown(
                label="Preset",
                value="Automatic",
                choices=[
                    "Automatic",
                    "Disable",
                    "guard_filter_fn",
                    "dynamic",
                    "max-autotune",
                    "max-autotune-no-cudagraphs",
                    "reduce-overhead",
                ],
                info='"Automatic" maintains the current compile status',
            )

            _dynamic = "Support any Resolution / Batch Size"
            _indynamic = "Require recompilation if Resolution / Batch Size is changed"
            _no_malloc = "Does not work with --cuda-malloc"

            gr.Markdown(rf"""
**torch.compile** speeds up the inference by compiling the model ahead of time
- **guard_filter_fn:** Compile the Fastest ; {_indynamic}
- **dynamic:** {_dynamic} ; Slower to Compile
- **max-autotune:** Best Runtime Speed ; {_indynamic} ; {_no_malloc}
- **max-autotune-no-cudagraphs:** {_dynamic} ; Faster than **dynamic** ; Even Slower to Compile
- **reduce-overhead:** Similar to **max-autotune** ; {_indynamic} ; {_no_malloc}
            """)

        return [preset]

    @staticmethod
    def restore(kmodel: "KModel"):
        model = get_attr(kmodel, _MODEL_BACKUP_KEY)
        set_attr_raw(kmodel, "diffusion_model", model)
        for attr in (_COMPILE_CONFIG_KEY, _COMPILED_BACKUP_KEY, _MODEL_BACKUP_KEY):
            if hasattr(kmodel, attr):
                delattr(kmodel, attr)

    def before_process_batch(self, p, *args, **kwargs):
        # temporarily restores the original model so LoRA can apply
        # (otherwise "keys mismatched")

        kmodel: "KModel" = p.sd_model.forge_objects.unet.model
        if not hasattr(kmodel, _COMPILE_CONFIG_KEY):
            return

        c_model = get_attr(kmodel, "diffusion_model")
        set_attr_raw(kmodel, _COMPILED_BACKUP_KEY, c_model)
        model = get_attr(kmodel, _MODEL_BACKUP_KEY)
        set_attr_raw(kmodel, "diffusion_model", model)

    def process_batch(self, p, preset: str, **kwargs):
        kmodel: "KModel" = p.sd_model.forge_objects.unet.model
        compiled: tuple[str, str] = getattr(kmodel, _COMPILE_CONFIG_KEY, None)
        enable: bool = (compiled is not None) if preset == "Automatic" else (preset != "Disable")

        if not enable:
            if compiled is not None:
                self.restore(kmodel)
            return

        if preset in ("max-autotune", "reduce-overhead") and cmd_args.cuda_malloc:
            logger.error(f"{preset} does not support --cuda-malloc\nModel is not compiled...")
            return

        _config: tuple[str, str] = (preset, p.sd_model.current_lora_hash)

        if compiled is not None:
            if preset in (compiled[0], "Automatic") and _config[1] == compiled[1]:
                _model = get_attr(kmodel, _COMPILED_BACKUP_KEY)
                set_attr_raw(kmodel, "diffusion_model", _model)
                delattr(kmodel, _COMPILED_BACKUP_KEY)
                return

            self.restore(kmodel)

        setattr(kmodel, _COMPILE_CONFIG_KEY, _config)

        match preset:
            case "guard_filter_fn":
                config = dict(backend="inductor", dynamic=False, fullgraph=False, options={"guard_filter_fn": skip_torch_compile_dict})
            case "dynamic":
                config = dict(backend="inductor", dynamic=True, fullgraph=False)
            case "max-autotune":
                config = dict(backend="inductor", dynamic=False, fullgraph=False, options={"coordinate_descent_tuning": True, "max_autotune": True, "triton.cudagraphs": True})
            case "max-autotune-no-cudagraphs":
                config = dict(backend="inductor", dynamic=True, fullgraph=False, options={"coordinate_descent_tuning": True, "max_autotune": True})
            case "reduce-overhead":
                config = dict(backend="inductor", mode="reduce-overhead", dynamic=False, fullgraph=False)

        kmodel = p.sd_model.forge_objects.unet.detach().model
        model = get_attr(kmodel, "diffusion_model")
        set_attr_raw(kmodel, _MODEL_BACKUP_KEY, model)

        # patch LoRA ahead-of-time
        p.sd_model.forge_objects.unet.refresh_loras()

        set_attr_raw(
            kmodel,
            "diffusion_model",
            torch.compile(model, **config),
        )

        logger.info(f"Model Compiled ({preset})")
