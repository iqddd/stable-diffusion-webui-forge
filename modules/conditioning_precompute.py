"""Request-local prompt-conditioning cache used by Prompts from File.

The cache deliberately lives outside the normal persistent conditioning cache.
It owns CPU copies of the conditioning objects and is discarded at the end of a
single Prompts from File request.

Maintenance constraints, observed runtime behaviour, and GPU-smoke instructions
are documented in ``docs/conditioning-precompute.md``.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from typing import Any

import torch

from backend import args, memory_management
from backend.logging import setup_logger
import modules.shared as shared
from modules import extra_networks, prompt_parser, script_callbacks
from modules.shared import opts, state


logger = logging.getLogger("conditioning_precompute")
setup_logger(logger)

_UNSET = object()


_SUPPORTED_ENGINES = {
    ("backend.diffusion_engine.sd15", "StableDiffusion"),
    ("backend.diffusion_engine.sdxl", "StableDiffusionXL"),
    ("backend.diffusion_engine.flux", "Flux"),
    ("backend.diffusion_engine.flux2", "Flux2"),
    ("backend.diffusion_engine.chroma", "Chroma"),
    ("backend.diffusion_engine.zimage", "ZImage"),
    ("backend.diffusion_engine.krea", "Krea2"),
}

def _script_id(script):
    return script.__class__.__module__, script.__class__.__name__


def _matches(script, relative_filename, class_name):
    filename = str(getattr(script, "filename", "")).replace("\\", "/").lower()
    return script.__class__.__name__ == class_name and filename.endswith(relative_filename.lower())


def _is_one_of(script, choices):
    return any(_matches(script, filename, class_name) for filename, class_name in choices)


_SAFE_SCRIPTS = {
    ("modules/processing_scripts/seed.py", "ScriptSeed"),
    ("modules/processing_scripts/sampler.py", "ScriptSampler"),
    ("modules/processing_scripts/comments.py", "ScriptComments"),
    ("modules/processing_scripts/rescale_cfg.py", "ScriptRescaleCFG"),
    ("modules/processing_scripts/mahiro.py", "ScriptMahiro"),
    ("extensions-builtin/sd_forge_compile/scripts/compile.py", "TorchCompileForForge"),
}


def _script_args(p, script):
    return (p.script_args or [])[script.args_from : script.args_to]


def _is_disabled(script, p):
    """Known Forge integrations have an explicit enable checkbox as arg 0."""
    values = _script_args(p, script)
    return bool(values) and not bool(values[0])


def _active_script_reason(p):
    """Return a conservative incompatibility reason for active always-on scripts."""
    for script in getattr(getattr(p, "scripts", None), "alwayson_scripts", []):
        script_id = _script_id(script)
        if _is_one_of(script, _SAFE_SCRIPTS):
            continue

        module, name = script_id
        values = _script_args(p, script)

        if _matches(script, "modules/processing_scripts/refiner.py", "ScriptRefiner"):
            if getattr(p, "refiner_checkpoint", None) in (None, "", "None", "none"):
                continue
            return "Refiner is enabled"

        if _matches(script, "extensions-builtin/extra-options-section/scripts/extra_options_section.py", "ExtraOptionsSection"):
            if not getattr(script, "setting_names", None):
                continue
            return "Extra options has configured fields"

        if _matches(script, "extensions-builtin/sd_forge_controlnet/scripts/controlnet.py", "ControlNetForForgeOfficial"):
            if not any(getattr(unit, "enabled", False) for unit in values):
                continue
            return "ControlNet has an enabled unit"

        if _matches(script, "extensions-builtin/sd_forge_image_stitch/scripts/image_stitch.py", "ImageStitch"):
            if _is_disabled(script, p) and getattr(script.__class__, "cached_parameters", None) is None:
                continue
            return "ImageStitch is enabled or has saved reference state"

        if _matches(script, "extensions-builtin/sd_forge_neveroom/scripts/forge_never_oom.py", "NeverOOMForForge"):
            if not any(values) and not getattr(script, "previous_unet_enabled", False):
                continue
            return "NeverOOM is enabled or waiting to switch state"

        disabled_builtin = {
            ("extensions-builtin/sd_forge_spectrum/scripts/spectrum.py", "SpectrumForForge"),
            ("extensions-builtin/sd_forge_radial/scripts/forge_radial.py", "RadialAttentionForForge"),
            ("extensions-builtin/sd_forge_multidiffusion/scripts/forge_multidiffusion.py", "MultiDiffusionForForge"),
            ("extensions-builtin/soft-inpainting/scripts/soft_inpainting.py", "Script"),
            ("extensions-builtin/sd_forge_pid/scripts/pid.py", "PiDForForge"),
        }
        if _is_one_of(script, disabled_builtin) and _is_disabled(script, p):
            continue

        return f"unsupported active script: {name} ({getattr(script, 'filename', module)})"
    return None


def _generation_callback_reason():
    # These hooks can alter conditioning tensors or model state while this request
    # is in flight. Forge itself does not register any of them.
    categories = ("cfg_denoiser", "cfg_denoised", "cfg_after_cfg", "extra_noise", "model_loaded")
    for category in categories:
        if script_callbacks.callback_map.get(f"callbacks_{category}"):
            return f"registered generation callback: {category}"
    return None


def _has_extra_network_tag(text):
    _, data = extra_networks.parse_prompt(text)
    return bool(data)


def _common_lora_reason(extra_network_data):
    """Validate tags allowed in the shared positive prompt."""
    unexpected = [name for name in extra_network_data if name != "lora"]
    if unexpected:
        return f"main prompt contains unsupported extra-network tag: {unexpected[0]}"
    return None


def _lora_state_snapshot(p):
    """The state Forge actually activated, including paths, weights and mode."""
    sd_model = getattr(p, "sd_model", None)
    return (
        getattr(sd_model, "current_lora_hash", None),
        bool(getattr(args.dynamic_args, "online_lora", False)),
    )


def _model_reason(p):
    engine = p.sd_model
    # The selectable script is entered before process_images() has performed its
    # first model load. Validate the real engine again at the commit point.
    if engine.__class__.__module__ == "modules.sd_models" and engine.__class__.__name__ == "FakeInitialModel":
        return None
    engine_id = engine.__class__.__module__, engine.__class__.__name__
    if engine_id not in _SUPPORTED_ENGINES:
        return f"unsupported engine: {engine_id[0]}.{engine_id[1]}"
    # Krea2 uses WanVAE tensor layout but is a text-to-image engine. Wan itself
    # remains excluded because its batch dimension represents video frames.
    if getattr(engine, "is_wan", False) and engine.__class__.__module__ != "backend.diffusion_engine.krea":
        return "video models are unsupported"
    if getattr(engine, "is_inpaint", False):
        return "image-conditioned model is unsupported"

    # Kontext/edit/reference engines expose their state in dynamic_args or on the
    # engine. Krea itself is valid when none of these are populated.
    reference_names = ("ref_latents", "references", "ini_latent", "image_prompts", "reference_images")
    if any(_has_value(getattr(args.dynamic_args, name, None)) for name in reference_names):
        return "reference conditioning is active"
    if any(_has_value(getattr(engine, name, None)) for name in reference_names):
        return "reference conditioning is active"
    if any(_has_value(getattr(p, name, None)) for name in reference_names):
        return "reference conditioning is active"
    if any(bool(getattr(args.dynamic_args, name, False)) for name in ("kontext", "edit", "anima")):
        return "image/reference mode is active"
    return None


def _has_value(value):
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    return bool(value)


def _additional_module_reason():
    modules = list(getattr(opts, "forge_additional_modules", []) or [])
    if not modules:
        return None
    try:
        from modules import sd_vae
        from modules.paths_internal import models_path

        vaes = {os.path.basename(name) for name in sd_vae.vae_dict}
        text_encoder_dir = os.path.normcase(os.path.abspath(os.path.join(models_path, "text_encoder")))
    except Exception:
        vaes = set()
        text_encoder_dir = ""
    for module in modules:
        absolute = os.path.normcase(os.path.abspath(module))
        if os.path.basename(module) in vaes:
            continue
        # Forge loads standalone text encoders through the same "additional
        # modules" setting as VAEs. They are part of the fixed TE configuration,
        # unlike a LoRA from models/Lora or any other arbitrary module.
        if text_encoder_dir and os.path.commonpath((absolute, text_encoder_dir)) == text_encoder_dir:
            continue
        return f"additional LoRA/module is active: {os.path.basename(module)}"
    return None


def _global_lora_reason():
    if getattr(opts, "sd_lora", "None") not in (None, "", "None", "none"):
        return f"global LoRA is active: {opts.sd_lora}"
    return None


def _extra_network_registry_reason():
    # `lora` is Forge's own registered network. It is called with an empty list
    # when no tag is present, which resets rather than changes the text encoder.
    for name, network in extra_networks.extra_network_registry.items():
        module = str(getattr(network, "__module__", ""))
        if name == "lora" and module in {"extra_networks_lora", "extensions_builtins.sd_forge_lora.extra_networks_lora"}:
            continue
        return f"registered extra network is not allowlisted: {name}"
    return None


def _prompt_source_reason(p, jobs):
    """Allow only Forge LoRA tags originating in the shared positive prompt."""
    if not isinstance(p.prompt, str) or not isinstance(p.negative_prompt, str):
        return "main prompt and negative prompt must be text"
    _, common_extra_network_data = extra_networks.parse_prompt(p.prompt)
    reason = _common_lora_reason(common_extra_network_data)
    if reason:
        return reason
    if _has_extra_network_tag(p.negative_prompt):
        return "main negative prompt contains an extra-network tag"

    for job in jobs:
        if job.unsupported_args:
            return f"unsupported per-line arguments: {', '.join(sorted(job.unsupported_args))}"
        if _has_extra_network_tag(job.line_prompt):
            return "per-line prompt contains an extra-network tag"
        if _has_extra_network_tag(job.line_negative_prompt):
            return "per-line negative prompt contains an extra-network tag"
        for style in getattr(job.p, "styles", []) or []:
            # Styles are materialised below. Inspecting their text here would tie
            # this module to StyleDatabase internals, so the materialised result is
            # validated again while building each snapshot.
            if not isinstance(style, str):
                return "unsupported style value"
    return None


def compatibility_reason(p, jobs) -> str | None:
    """Validate the narrow v1 contract before changing normal processing."""
    from modules.processing import StableDiffusionProcessingTxt2Img

    if not isinstance(p, StableDiffusionProcessingTxt2Img):
        return "only txt2img is supported"
    if getattr(p, "enable_hr", False):
        return "Hires fix is enabled"
    if getattr(p, "refiner_checkpoint", None) not in (None, "", "None", "none"):
        return "Refiner is enabled"
    if args.dynamic_args.pid:
        return "PiD is active"

    reason = _prompt_source_reason(p, jobs)
    if reason:
        return reason

    for reason in (_model_reason(p), _additional_module_reason(), _global_lora_reason(), _extra_network_registry_reason(), _active_script_reason(p), _generation_callback_reason()):
        if reason:
            return reason
    return None


def _freeze(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype), str(value.device), int(value.data_ptr()))
    if isinstance(value, extra_networks.ExtraNetworkParams):
        return (
            "extra-network-params",
            tuple(_freeze(x) for x in value.items),
            tuple(_freeze(x) for x in value.positional),
            tuple(sorted((str(k), _freeze(v)) for k, v in value.named.items())),
        )
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(x) for x in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def _embedding_snapshot():
    """A cheap immutable signature for textual-inversion files currently in use."""
    directory = getattr(args.dynamic_args, "embedding_dir", None)
    if not directory:
        return None
    try:
        directory = os.path.normcase(os.path.abspath(directory))
        entries = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                path = os.path.join(root, filename)
                stat = os.stat(path)
                entries.append((os.path.normcase(path), stat.st_mtime_ns, stat.st_size))
        return directory, tuple(sorted(entries))
    except OSError:
        # A transiently unavailable embedding directory is itself part of the
        # snapshot. If it becomes available later, lookup will fall back.
        return directory, None


def conditioning_key(p, required_prompts, steps, extra_network_data, hires_steps, kind, embedding_snapshot=None):
    """A value snapshot, not a reference to mutable p fields or prompt lists."""
    prompts = tuple(required_prompts)
    clip = getattr(getattr(p.sd_model, "forge_objects", None), "clip", None)
    prompt_meta = (
        getattr(required_prompts, "is_negative_prompt", False),
        getattr(required_prompts, "width", None),
        getattr(required_prompts, "height", None),
        getattr(required_prompts, "distilled_cfg_scale", None),
    )
    return (
        kind,
        prompts,
        prompt_meta,
        int(steps),
        hires_steps,
        _freeze(extra_network_data),
        int(p.width),
        int(p.height),
        p.distilled_cfg_scale,
        getattr(p, "hr_distilled_cfg", None),
        getattr(p, "sd_model_name", None),
        getattr(p, "sd_model_hash", None),
        id(p.sd_model),
        id(clip),
        _lora_state_snapshot(p),
        getattr(opts, "CLIP_stop_at_last_layers", None),
        getattr(opts, "sdxl_crop_left", None),
        getattr(opts, "sdxl_crop_top", None),
        getattr(opts, "sdxl_zero_neg", None),
        getattr(opts, "emphasis", None),
        _freeze(getattr(opts, "forge_additional_modules", [])),
        _embedding_snapshot() if embedding_snapshot is None else embedding_snapshot,
    )


def _clone_tree(value, *, to_cpu=False, devices=None, path=()):
    """Clone all tensors without allowing a sampling hook to mutate the cache."""
    if isinstance(value, torch.Tensor):
        if to_cpu:
            devices[path] = value.device
            return value.detach().to(device="cpu").clone()
        return value.detach().to(device=devices[path]).clone()
    if isinstance(value, dict):
        result = value.__class__() if value.__class__ is dict else copy.copy(value)
        if value.__class__ is not dict:
            result.clear()
        for key, item in value.items():
            result[key] = _clone_tree(item, to_cpu=to_cpu, devices=devices, path=path + (("key", str(key)),))
        return result
    if isinstance(value, list):
        return value.__class__([_clone_tree(item, to_cpu=to_cpu, devices=devices, path=path + (index,)) for index, item in enumerate(value)])
    if isinstance(value, tuple):
        items = [_clone_tree(item, to_cpu=to_cpu, devices=devices, path=path + (index,)) for index, item in enumerate(value)]
        return value.__class__(*items) if hasattr(value, "_fields") else tuple(items)
    if hasattr(value, "__dict__"):
        result = copy.copy(value)
        for key, item in vars(value).items():
            setattr(result, key, _clone_tree(item, to_cpu=to_cpu, devices=devices, path=path + (("attr", key),)))
        return result
    return value


@dataclass
class _StoredConditioning:
    value: Any
    devices: dict
    extra_generation_params: dict


@dataclass
class _MaterialisedPrompts:
    positive: list[str]
    negative: list[str]
    extra_network_data: dict


class PrecomputeContext:
    """A queue-scoped cache shared only by a Prompts from File invocation."""

    def __init__(self, jobs, common_prompt=""):
        self.jobs = jobs
        self.entries: dict[tuple, _StoredConditioning] = {}
        self.prepared = False
        self.preparing = False
        self.disabled = False
        self.warning_emitted = False
        self.failure_reason = None
        self.result_emitted = False
        _, common_extra_network_data = extra_networks.parse_prompt(common_prompt)
        self.common_extra_network_snapshot = _freeze(common_extra_network_data)
        self._materialised: dict[int, _MaterialisedPrompts] = {}
        self.embedding_snapshot = _embedding_snapshot()
        self.lora_state_snapshot = _UNSET

    def disable(self, reason):
        if self.disabled:
            return
        self.disabled = True
        self.failure_reason = reason
        self.entries.clear()
        if not self.warning_emitted:
            self.warning_emitted = True
            print(f"[Prompts from File] Precompute prompts disabled: {reason}. Continuing with normal text encoding.")

    def clear(self):
        entry_count = len(self.entries)
        if not self.result_emitted:
            self.result_emitted = True
            if self.prepared and not self.disabled:
                logger.debug(
                    "Prompts from File precompute result: success; batch precompute completed: prepared %d conditioning entries across %d jobs; sampling will use the request-local cache.",
                    entry_count,
                    len(self.jobs),
                )
            else:
                logger.debug(
                    "Prompts from File precompute result: fallback; reason=%s; legacy fallback enabled with TE encoding per job.",
                    self.failure_reason or "precompute did not reach the conditioning commit point",
                )
        self.entries.clear()
        self.disabled = True

    def lookup(self, p, required_prompts, steps, extra_network_data, hires_steps, kind):
        if self.disabled or self.preparing or not self.prepared:
            return None
        if _embedding_snapshot() != self.embedding_snapshot:
            self.disable("textual inversion embeddings changed after precompute")
            return None
        if _lora_state_snapshot(p) != self.lora_state_snapshot:
            self.disable("active LoRA state changed after precompute")
            return None
        key = conditioning_key(p, required_prompts, steps, extra_network_data, hires_steps, kind, self.embedding_snapshot)
        entry = self.entries.get(key)
        if entry is None:
            self.disable("conditioning inputs changed after precompute")
            return None
        p.sd_model.extra_generation_params.update(entry.extra_generation_params)
        return _clone_tree(entry.value, to_cpu=False, devices=entry.devices)

    def _store(self, p, required_prompts, steps, extra_network_data, hires_steps, kind, value):
        if self.lora_state_snapshot is _UNSET:
            self.lora_state_snapshot = _lora_state_snapshot(p)
        key = conditioning_key(p, required_prompts, steps, extra_network_data, hires_steps, kind, self.embedding_snapshot)
        if key in self.entries:
            return
        devices = {}
        self.entries[key] = _StoredConditioning(
            _clone_tree(value, to_cpu=True, devices=devices),
            devices,
            p.sd_model.extra_generation_params.copy(),
        )

    def _materialise_prompts(self, job):
        p = job.p
        count = p.batch_size * p.n_iter
        positive = [shared.prompt_styles.apply_styles_to_prompt(job.prompt, p.styles) for _ in range(count)]
        negative = [shared.prompt_styles.apply_negative_styles_to_prompt(job.negative_prompt, p.styles) for _ in range(count)]
        try:
            from modules.processing_scripts.comments import strip_comments

            positive = [strip_comments(x) for x in positive]
            negative = [strip_comments(x) for x in negative]
        except Exception:
            pass

        parsed_positive = []
        parsed_extra_network_data = []
        for text in positive:
            parsed_text, data = extra_networks.parse_prompt(text)
            reason = _common_lora_reason(data)
            if reason:
                raise ValueError(reason)
            if _freeze(data) != self.common_extra_network_snapshot:
                raise ValueError("a Style changed the common LoRA configuration")
            parsed_positive.append(parsed_text)
            parsed_extra_network_data.append(data)

        if any(_has_extra_network_tag(text) for text in negative):
            raise ValueError("a Style added an extra-network tag to the negative prompt")

        first_data = parsed_extra_network_data[0] if parsed_extra_network_data else {}
        return _MaterialisedPrompts(parsed_positive, negative, first_data)

    def _get_materialised_prompts(self, job):
        key = id(job)
        if key not in self._materialised:
            self._materialised[key] = self._materialise_prompts(job)
        return self._materialised[key]

    def prepare(self, p):
        if self.prepared or self.disabled:
            return
        self.preparing = True
        engine_params_before = p.sd_model.extra_generation_params.copy()
        job_before_precompute = state.job
        try:
            reason = _model_reason(p)
            if reason:
                self.disable(reason)
                return
            if p is not self.jobs[0].p or p.iteration != 0:
                self.disable("the first conditioning batch did not reach the expected commit point")
                return
            if state.skipped:
                # Preserve Forge's normal skip signal. It is consumed by the
                # ordinary batch loop, so preparation must not turn it into a
                # hidden successful cache hit.
                self.disable("precompute was skipped")
                return

            # The context is constructed by the selectable script before the
            # first process_images() call. At that point Forge may still have a
            # FakeInitialModel and dynamic_args.embedding_dir is not populated
            # until forge_model_reload(). Establish the baseline only now, at
            # the real conditioning commit point after model loading and
            # process_batch(), otherwise the first lookup falsely reports that
            # textual-inversion embeddings changed.
            self.embedding_snapshot = _embedding_snapshot()

            first = self._get_materialised_prompts(self.jobs[0])
            first_slice = slice(0, p.batch_size)
            if list(p.prompts) != first.positive[first_slice] or list(p.negative_prompts) != first.negative[first_slice]:
                self.disable("prompt or negative prompt changed before the conditioning commit point")
                return
            if _freeze(p.extra_network_data) != _freeze(first.extra_network_data):
                self.disable("LoRA configuration changed before the conditioning commit point")
                return
            self.lora_state_snapshot = _lora_state_snapshot(p)

            for index, job in enumerate(self.jobs, start=1):
                state.job = f"Precomputing prompts: {index}/{len(self.jobs)}"
                if state.interrupted or state.stopping_generation:
                    self.disable("precompute was interrupted")
                    return
                if state.skipped:
                    self.disable("precompute was skipped")
                    return
                try:
                    materialised = self._get_materialised_prompts(job)
                except ValueError as exc:
                    self.disable(str(exc))
                    return
                work = copy.copy(job.p)
                work.cached_c = [None, None, None]
                work.cached_uc = [None, None, None]
                work._precompute_context = self
                work.sd_model_name = p.sd_model_name
                work.sd_model_hash = p.sd_model_hash
                work.sd_vae_name = p.sd_vae_name
                work.sd_vae_hash = p.sd_vae_hash

                # Kept lazy so the small cache unit tests do not need optional
                # sampler dependencies loaded by the full web UI.
                from modules import sd_samplers

                sampler_config = sd_samplers.find_sampler_config(work.sampler_name)
                total_steps = sampler_config.total_steps(work.steps) if sampler_config else work.steps
                work.step_multiplier = total_steps // work.steps
                work.firstpass_steps = total_steps

                for iteration in range(work.n_iter):
                    work.prompts = materialised.positive[iteration * work.batch_size : (iteration + 1) * work.batch_size]
                    work.negative_prompts = materialised.negative[iteration * work.batch_size : (iteration + 1) * work.batch_size]
                    work.extra_network_data = materialised.extra_network_data
                    pos = prompt_parser.SdConditioning(work.prompts, width=work.width, height=work.height, distilled_cfg_scale=work.distilled_cfg_scale)
                    neg = prompt_parser.SdConditioning(work.negative_prompts, width=work.width, height=work.height, is_negative_prompt=True, distilled_cfg_scale=work.distilled_cfg_scale)

                    # setup_conds is the Forge implementation of schedules, AND,
                    # CFG=1 and all engine-specific conditioning metadata.
                    work.sd_model.extra_generation_params = {}
                    work.setup_conds()
                    self._store(work, pos, total_steps, work.extra_network_data, None, "positive", work.c)
                    if work.uc is not None:
                        self._store(work, neg, total_steps, work.extra_network_data, None, "negative", work.uc)
                print(f"[Prompts from File] Precomputed prompts: {index}/{len(self.jobs)}")

            self.prepared = True
            self._offload_text_encoder(p)
            print(f"[Prompts from File] Precomputed {len(self.entries)} conditioning entries; starting sampling.")
        except (MemoryError, torch.cuda.OutOfMemoryError) as exc:
            self.disable(f"not enough memory while preparing conditioning ({exc})")
        except Exception as exc:
            # Compatibility failures fall back; actual TE failures must remain
            # visible to the caller after the request result has been recorded.
            self.disable(f"text encoder preparation failed: {type(exc).__name__}: {exc}")
            raise
        finally:
            # setup_conds writes this dictionary on the shared engine. The first
            # real batch still needs its own metadata when it resumes below.
            p.sd_model.extra_generation_params = engine_params_before
            state.job = job_before_precompute
            self.preparing = False

    @staticmethod
    def _offload_text_encoder(p):
        clip = getattr(getattr(p.sd_model, "forge_objects", None), "clip", None)
        patcher = getattr(clip, "patcher", None)
        device = getattr(patcher, "load_device", None)
        if patcher is None or device is None or device.type == "cpu":
            return
        # This calls LoadedModel.model_unload through free_memory. Do not use
        # unload_model(), which only removes bookkeeping for the loaded model.
        memory_management.free_memory(1e30, device)


def maybe_precompute(p):
    context = getattr(p, "_precompute_context", None)
    if context is not None and not context.disabled and not context.prepared:
        context.prepare(p)


def log_initial_fallback(reason):
    """Record the one request result when no context can be constructed."""
    logger.debug(
        "Prompts from File precompute result: fallback; reason=%s; legacy fallback enabled with TE encoding per job.",
        reason,
    )


def lookup(p, required_prompts, steps, extra_network_data, hires_steps, kind):
    context = getattr(p, "_precompute_context", None)
    if context is None:
        return None
    return context.lookup(p, required_prompts, steps, extra_network_data, hires_steps, kind)
