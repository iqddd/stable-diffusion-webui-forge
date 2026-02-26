from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.modules.k_model import KModel

import logging
import os

import gradio as gr
import torch

from backend.utils import get_attr, set_attr_raw
from modules import scripts


def skip_torch_compile_dict(guard_entries):
    # https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_torch_compile.py#L5
    return [("transformer_options" not in entry.name) for entry in guard_entries]


class TorchCompileForForge(scripts.Script):
    sorting_priority = 67
    _compile_log_initialized = False

    def __init__(self):
        torch._dynamo.config.cache_size_limit = 256
        self._initialize_compile_logging()

    @classmethod
    def _initialize_compile_logging(cls):
        if cls._compile_log_initialized:
            return

        log_path = os.environ.get("FORGE_TORCH_COMPILE_LOG_PATH", "").strip()
        if not log_path:
            return

        abs_log_path = os.path.abspath(log_path)
        os.makedirs(os.path.dirname(abs_log_path), exist_ok=True)

        file_handler = logging.FileHandler(abs_log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s"))

        for logger_name in ("torch._dynamo", "torch._inductor", "torch.fx.experimental.symbolic_shapes"):
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
            logger.addHandler(file_handler)

        set_logs = getattr(getattr(torch, "_logging", None), "set_logs", None)
        if callable(set_logs):
            set_logs(
                dynamo=logging.INFO,
                inductor=logging.INFO,
                dynamic=logging.INFO,
                graph_breaks=True,
                guards=True,
                recompiles=True,
                recompiles_verbose=True,
            )

        cls._compile_log_initialized = True
        print(f'[Torch Compile Integrated] compile logs -> "{abs_log_path}"')

    def title(self):
        return "Torch Compile Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.Markdown(
                """
**torch.compile** speeds up the Inference by compiling the model ahead of time
- **guard_filter_fn:** Compile the Fastest ; Require recompilation if Resolution / Batch Size is changed
- **dynamic:** Longer to Compile ; Support any Resolution / Batch Size
- **cudagraphs:** Not Recommended
                """
            )
            preset = gr.Dropdown(
                label="Preset",
                value="Automatic",
                choices=["Automatic", "Disable", "guard_filter_fn", "dynamic", "cudagraphs"],
                info='"Automatic" maintains the current compile status',
            )

        return [preset]

    @staticmethod
    def restore(kmodel: "KModel"):
        model = get_attr(kmodel, "_model_backup")
        set_attr_raw(kmodel, "diffusion_model", model)
        del kmodel._compile_config
        del kmodel._compiled_backup
        del kmodel._model_backup

    def before_process_batch(self, p, *args, **kwargs):
        kmodel: "KModel" = p.sd_model.forge_objects.unet.model
        if not hasattr(kmodel, "_compile_config"):
            return

        c_model = get_attr(kmodel, "diffusion_model")
        set_attr_raw(kmodel, "_compiled_backup", c_model)
        # temporarily restores the original model so LoRA can apply
        model = get_attr(kmodel, "_model_backup")
        set_attr_raw(kmodel, "diffusion_model", model)

    def process_batch(self, p, preset: str, **kwargs):
        kmodel: "KModel" = p.sd_model.forge_objects.unet.model
        compiled: bool = hasattr(kmodel, "_compile_config")
        enable: bool = compiled if preset == "Automatic" else (preset != "Disable")

        if not enable:
            if compiled:
                self.restore(kmodel)
            return

        match preset:
            case "guard_filter_fn":
                config = dict(backend="inductor", dynamic=False, fullgraph=False, options={"guard_filter_fn": skip_torch_compile_dict})
            case "dynamic":
                config = dict(backend="inductor", dynamic=True, fullgraph=False)
            case "cudagraphs":
                config = dict(backend="cudagraphs", dynamic=True, fullgraph=True)
            case _:
                config: dict = kmodel._compile_config

        if compiled:
            if kmodel._compile_config == config:
                c_model = get_attr(kmodel, "_compiled_backup")
                set_attr_raw(kmodel, "diffusion_model", c_model)
                del kmodel._compiled_backup
                return

            self.restore(kmodel)

        model = get_attr(kmodel, "diffusion_model")
        set_attr_raw(kmodel, "_model_backup", model)

        set_attr_raw(
            kmodel,
            "diffusion_model",
            torch.compile(model, **config),
        )

        kmodel._compile_config = config
