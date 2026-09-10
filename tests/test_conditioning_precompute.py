import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

# backend.args parses the process command line at import time. Unit-test runners
# put their own module selectors there, which are unrelated to Forge options.
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modules
from modules import conditioning_precompute, prompt_parser


class _Engine:
    def __init__(self):
        self.forge_objects = SimpleNamespace(clip=object())
        self.extra_generation_params = {}


class _Processing:
    width = 512
    height = 512
    distilled_cfg_scale = 3.5
    hr_distilled_cfg = None
    sd_model_name = "test"
    sd_model_hash = "hash"

    def __init__(self):
        self.sd_model = _Engine()


class _PreparingProcessing(_Processing):
    batch_size = 1
    n_iter = 1
    iteration = 0
    prompts = ["one"]
    negative_prompts = [""]
    styles = []
    sampler_name = "Euler"
    steps = 4
    sd_vae_name = None
    sd_vae_hash = None

    def __init__(self):
        super().__init__()
        self.extra_network_data = {}

    def setup_conds(self):
        self.c = {"crossattn": torch.tensor([[1.0]])}
        self.uc = None


class ConditioningPrecomputeTest(unittest.TestCase):
    def test_prompt_sources_allow_multiple_main_loras_only(self):
        main = SimpleNamespace(
            prompt="cinematic <lora:first:0.7><lora:second:unet=0.8:te=0.4><lora:third:1>",
            negative_prompt="",
        )
        job = SimpleNamespace(
            p=SimpleNamespace(styles=[]),
            line_prompt="a subject",
            line_negative_prompt="",
            unsupported_args=set(),
        )

        self.assertIsNone(conditioning_precompute._prompt_source_reason(main, [job]))

        job.line_prompt = "a subject <lora:line-only:1>"
        self.assertIn("per-line prompt", conditioning_precompute._prompt_source_reason(main, [job]))
        job.line_prompt = "a subject"
        job.line_negative_prompt = "bad <lora:line-negative:1>"
        self.assertIn("per-line negative", conditioning_precompute._prompt_source_reason(main, [job]))
        job.line_negative_prompt = ""
        main.negative_prompt = "bad <lora:negative:1>"
        self.assertIn("negative prompt", conditioning_precompute._prompt_source_reason(main, [job]))
        main.negative_prompt = ""
        main.prompt = "cinematic <hypernet:unsupported:1>"
        self.assertIn("unsupported extra-network", conditioning_precompute._prompt_source_reason(main, [job]))

    def test_extra_network_params_have_stable_structural_keys(self):
        prompt = "<lora:first:0.7><lora:second:unet=0.8:te=0.4><lora:third:1>"
        _, first = conditioning_precompute.extra_networks.parse_prompt(prompt)
        _, equivalent = conditioning_precompute.extra_networks.parse_prompt(prompt)
        _, reordered = conditioning_precompute.extra_networks.parse_prompt(
            "<lora:second:unet=0.8:te=0.4><lora:first:0.7><lora:third:1>"
        )
        _, changed_weight = conditioning_precompute.extra_networks.parse_prompt(
            "<lora:first:0.8><lora:second:unet=0.8:te=0.4><lora:third:1>"
        )

        frozen = conditioning_precompute._freeze(first)
        self.assertEqual(frozen, conditioning_precompute._freeze(equivalent))
        self.assertNotEqual(frozen, conditioning_precompute._freeze(reordered))
        self.assertNotEqual(frozen, conditioning_precompute._freeze(changed_weight))

    def test_materialised_style_cannot_add_lora(self):
        p = _PreparingProcessing()
        p.styles = ["unsafe"]
        job = SimpleNamespace(p=p, prompt="a subject", negative_prompt="")
        prompt_styles = SimpleNamespace(
            apply_styles_to_prompt=lambda prompt, styles: prompt + " <lora:style-only:1>",
            apply_negative_styles_to_prompt=lambda prompt, styles: prompt,
        )

        with mock.patch.object(conditioning_precompute.shared, "prompt_styles", prompt_styles):
            cache = conditioning_precompute.PrecomputeContext([job], common_prompt="")
            with self.assertRaisesRegex(ValueError, "Style changed the common LoRA"):
                cache._materialise_prompts(job)

    def test_global_lora_setting_remains_unsupported(self):
        with mock.patch.object(conditioning_precompute, "opts", SimpleNamespace(sd_lora="global-lora")):
            self.assertEqual(conditioning_precompute._global_lora_reason(), "global LoRA is active: global-lora")

    def test_prepare_and_lookup_with_three_common_loras(self):
        common = "cinematic <lora:first:0.7><lora:second:unet=0.8:te=0.4><lora:third:1>"
        raw_prompt = f"a subject {common}"
        parsed_prompt, extra_network_data = conditioning_precompute.extra_networks.parse_prompt(raw_prompt)
        p = _PreparingProcessing()
        p.prompts = [parsed_prompt]
        p.extra_network_data = extra_network_data
        p.sd_model.current_lora_hash = "three-active-loras"
        job = SimpleNamespace(p=p, prompt=raw_prompt, negative_prompt="")
        request_state = SimpleNamespace(job="", skipped=False, interrupted=False, stopping_generation=False)
        prompt_styles = SimpleNamespace(
            apply_styles_to_prompt=lambda prompt, styles: prompt,
            apply_negative_styles_to_prompt=lambda prompt, styles: prompt,
        )
        sd_samplers = SimpleNamespace(find_sampler_config=lambda name: None)

        with mock.patch.object(conditioning_precompute, "_model_reason", return_value=None), mock.patch.object(
            conditioning_precompute.PrecomputeContext, "_offload_text_encoder"
        ), mock.patch.object(conditioning_precompute, "state", request_state), mock.patch.object(
            conditioning_precompute.shared, "prompt_styles", prompt_styles
        ), mock.patch.object(modules, "sd_samplers", sd_samplers, create=True), mock.patch.dict(
            sys.modules, {"modules.sd_samplers": sd_samplers}
        ):
            cache = conditioning_precompute.PrecomputeContext([job], common_prompt=common)
            cache.prepare(p)
            prompts = prompt_parser.SdConditioning([parsed_prompt], width=512, height=512, distilled_cfg_scale=3.5)
            result = cache.lookup(p, prompts, 4, extra_network_data, None, "positive")

        self.assertTrue(cache.prepared)
        self.assertFalse(cache.disabled)
        self.assertIsNotNone(result)

    def test_changed_active_lora_state_disables_request_cache(self):
        p = _Processing()
        p.sd_model.current_lora_hash = "first"
        prompts = prompt_parser.SdConditioning(["one"], width=512, height=512)
        cache = conditioning_precompute.PrecomputeContext([])
        cache._store(p, prompts, 20, {}, None, "positive", torch.tensor([[1.0]]))
        cache.prepared = True

        p.sd_model.current_lora_hash = "second"
        self.assertIsNone(cache.lookup(p, prompts, 20, {}, None, "positive"))
        self.assertEqual(cache.failure_reason, "active LoRA state changed after precompute")

    def test_embedding_snapshot_is_rebased_after_first_model_load(self):
        p = _PreparingProcessing()
        job = SimpleNamespace(p=p, prompt="one", negative_prompt="")
        before_model_load = None
        at_commit_point = ("embeddings", (("embedding.pt", 1, 2),))
        request_state = SimpleNamespace(job="", skipped=False, interrupted=False, stopping_generation=False)
        prompt_styles = SimpleNamespace(
            apply_styles_to_prompt=lambda prompt, styles: prompt,
            apply_negative_styles_to_prompt=lambda prompt, styles: prompt,
        )
        sd_samplers = SimpleNamespace(find_sampler_config=lambda name: None)

        with mock.patch.object(
            conditioning_precompute,
            "_embedding_snapshot",
            side_effect=[before_model_load, at_commit_point, at_commit_point],
        ), mock.patch.object(conditioning_precompute, "_model_reason", return_value=None), mock.patch.object(
            conditioning_precompute.PrecomputeContext, "_offload_text_encoder"
        ), mock.patch.object(conditioning_precompute, "state", request_state), mock.patch.object(
            conditioning_precompute.shared, "prompt_styles", prompt_styles
        ), mock.patch.object(modules, "sd_samplers", sd_samplers, create=True), mock.patch.dict(
            sys.modules, {"modules.sd_samplers": sd_samplers}
        ):
            cache = conditioning_precompute.PrecomputeContext([job])
            cache.prepare(p)

            prompts = prompt_parser.SdConditioning(["one"], width=512, height=512, distilled_cfg_scale=3.5)
            result = cache.lookup(p, prompts, 4, {}, None, "positive")

        self.assertEqual(cache.embedding_snapshot, at_commit_point)
        self.assertIsNotNone(result)

    def test_cpu_snapshot_is_immutable_and_lookup_returns_a_copy(self):
        p = _Processing()
        prompts = prompt_parser.SdConditioning(["one [red:blue:0.5] AND cat:0.4"], width=512, height=512)
        value = {"crossattn": torch.tensor([[1.0, 2.0]])}
        cache = conditioning_precompute.PrecomputeContext([])
        cache._store(p, prompts, 20, {}, None, "positive", value)
        cache.prepared = True

        result = cache.lookup(p, prompts, 20, {}, None, "positive")
        self.assertIsNotNone(result)
        self.assertEqual(result["crossattn"].device.type, "cpu")
        result["crossattn"].add_(10)

        second = cache.lookup(p, prompts, 20, {}, None, "positive")
        self.assertTrue(torch.equal(second["crossattn"], torch.tensor([[1.0, 2.0]])))

    def test_changed_conditioning_input_disables_only_request_cache(self):
        p = _Processing()
        prompts = prompt_parser.SdConditioning(["one"], width=512, height=512)
        cache = conditioning_precompute.PrecomputeContext([])
        cache._store(p, prompts, 20, {}, None, "positive", torch.tensor([[1.0]]))
        cache.prepared = True

        changed = prompt_parser.SdConditioning(["two"], width=512, height=512)
        self.assertIsNone(cache.lookup(p, changed, 20, {}, None, "positive"))
        self.assertTrue(cache.disabled)

    def test_parser_keeps_schedules_and_and_components_deterministic(self):
        schedules = prompt_parser.get_learned_conditioning_prompt_schedules(["[red:blue:0.5] AND [cat|dog]"], 4)
        self.assertEqual([end for end, _ in schedules[0]], [1, 2, 3, 4])
        indexes, flat, _ = prompt_parser.get_multicond_prompt_list(["cat AND dog:0.5"])
        self.assertEqual(indexes, [[(0, 1.0), (1, 0.5)]])
        self.assertEqual(flat, ["cat", " dog"])

    def test_engine_allowlist_rejects_same_module_refiner_and_unknown_classes(self):
        def engine(module, name):
            return type(name, (), {"__module__": module})()

        allowed = SimpleNamespace(sd_model=engine("backend.diffusion_engine.sdxl", "StableDiffusionXL"))
        refiner = SimpleNamespace(sd_model=engine("backend.diffusion_engine.sdxl", "StableDiffusionXLRefiner"))
        unknown = SimpleNamespace(sd_model=engine("backend.diffusion_engine.sdxl", "UnexpectedSDXL"))

        self.assertIsNone(conditioning_precompute._model_reason(allowed))
        self.assertIn("unsupported engine", conditioning_precompute._model_reason(refiner))
        self.assertIn("unsupported engine", conditioning_precompute._model_reason(unknown))


if __name__ == "__main__":
    unittest.main()
