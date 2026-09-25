import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from PIL import Image

# Forge parses the process command line while importing shared modules. Test
# selectors and pytest flags are not Forge launcher arguments.
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modules
from modules import shared_state


def make_opts(**overrides):
    values = {
        "live_previews_enable": True,
        "show_progress_every_n_steps": -1,
        "live_preview_use_completed_output": True,
        "show_progress_grid": False,
        "live_previews_image_format": "png",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_state():
    state = object.__new__(shared_state.State)
    state.current_latent = None
    state.current_image = None
    state.id_live_preview = 0
    state.job_no = 0
    state.sampling_step = 1
    state.preview_step = 1
    state.current_image_sampling_step = 0
    return state


def test_nextjob_skips_preview_decoder_for_completed_output(monkeypatch):
    monkeypatch.setattr(shared_state.shared, "opts", make_opts())
    state = make_state()
    state.current_latent = SimpleNamespace(ndim=4)
    calls = []
    monkeypatch.setattr(state, "do_set_current_image", lambda: calls.append(True))

    state.nextjob()

    assert calls == []
    assert state.job_no == 1


def test_nextjob_keeps_existing_decoder_when_option_is_disabled(monkeypatch):
    monkeypatch.setattr(shared_state.shared, "opts", make_opts(live_preview_use_completed_output=False))
    state = make_state()
    state.current_latent = SimpleNamespace(ndim=4)
    calls = []
    monkeypatch.setattr(state, "do_set_current_image", lambda: calls.append(True))

    state.nextjob()

    assert calls == [True]


def test_nextjob_keeps_existing_decoder_for_video(monkeypatch):
    monkeypatch.setattr(shared_state.shared, "opts", make_opts())
    state = make_state()
    state.current_latent = SimpleNamespace(ndim=5, size=lambda dim: 4 if dim == 2 else 1)
    calls = []
    monkeypatch.setattr(state, "do_set_current_image", lambda: calls.append(True))

    state.nextjob()

    assert calls == [True]


def test_assign_completed_images_uses_first_image_without_grid(monkeypatch):
    monkeypatch.setattr(shared_state.shared, "opts", make_opts(show_progress_grid=False))
    state = make_state()
    first = Image.new("RGB", (8, 8), "red")
    second = Image.new("RGB", (8, 8), "blue")

    state.assign_completed_images([first, second])

    assert state.current_image is first
    assert state.id_live_preview == 1


def test_assign_completed_images_uses_grid(monkeypatch):
    monkeypatch.setattr(shared_state.shared, "opts", make_opts(show_progress_grid=True))
    state = make_state()
    first = Image.new("RGB", (8, 8), "red")
    second = Image.new("RGB", (8, 8), "blue")
    expected = Image.new("RGB", (16, 8), "green")
    calls = []

    def image_grid(completed_images):
        calls.append(completed_images)
        return expected

    images_module = ModuleType("modules.images")
    images_module.image_grid = image_grid
    monkeypatch.setitem(sys.modules, "modules.images", images_module)
    monkeypatch.setattr(modules, "images", images_module, raising=False)
    state.assign_completed_images([first, second])

    assert calls == [[first, second]]
    assert state.current_image is expected
    assert state.id_live_preview == 1


def test_assign_completed_images_is_inactive_for_step_previews(monkeypatch):
    monkeypatch.setattr(shared_state.shared, "opts", make_opts(show_progress_every_n_steps=1))
    state = make_state()

    state.assign_completed_images([Image.new("RGB", (8, 8))])

    assert state.current_image is None
    assert state.id_live_preview == 0


def test_interrupted_result_is_published_without_another_preview_decode(monkeypatch):
    monkeypatch.setattr(shared_state.shared, "opts", make_opts(live_preview_fast_interrupt=True))
    state = make_state()
    state.interrupted = True
    state.current_latent = SimpleNamespace(ndim=4)
    decoder_calls = []
    monkeypatch.setattr(state, "do_set_current_image", lambda: decoder_calls.append(True))
    interrupted_result = Image.new("RGB", (8, 8), "purple")

    state.nextjob()
    state.assign_completed_images([interrupted_result])

    assert decoder_calls == []
    assert state.current_image is interrupted_result
    assert state.id_live_preview == 1
