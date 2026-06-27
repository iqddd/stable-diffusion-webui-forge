# Haoming Neo Sync WIP

Snapshot date: 2026-06-27 UTC

Current WIP branch: `wip/haoming-neo-sync-20260627`

Work branch before snapshot: `batch2/inference`

Target upstream: `haoming/neo`

Remote note: local `haoming` remote points to `https://github.com/Haoming02/sd-webui-forge-classic.git`, but the sync target for this stage is the `neo` branch, not the `classic` branch.

Base strategy from the original plan:

- Work from common ancestor `4b338a1c`.
- Sync `haoming/neo` into local `neo` in thematic batches.
- Prefer direct upstream cherry-pick batches where conflicts are simple.
- In hot zones, synthesize manually and preserve local behavior case by case.
- Intermediate state may be temporarily broken; final state must pass validation.
- Use `TODO Merging:` comments for known merge followups that are better fixed in a final pass.

## Current Git State

Last integration commit before this artifact:

```text
5522b327 ptl
```

Approximate divergence at snapshot:

```text
git log --left-right --cherry-pick --oneline --no-merges HEAD...haoming/neo
left/local unique:    50
right/upstream todo: 228
```

Diff size from local `neo` to current WIP:

```text
73 files changed, 3501 insertions(+), 1802 deletions(-)
```

The working tree was clean before adding this artifact.

## User Decisions To Preserve

- Move faster from here: larger upstream chunks are preferred.
- It is not necessary to keep every intermediate step runnable.
- Final state must be runnable and validated.
- Do not automatically let upstream replace local fixes in hot zones.
- Keep local Flux2 scheduler / sigma / TAESD-related fixes unless upstream clearly covers them.
- Keep Euler Negative fixes.
- Keep Flux LoRA sliced-qkv and RMSNorm key mapping fixes.
- Keep BOFT strength interpolation.
- Keep fast online FP8 LoRA behavior as a priority.
- Do not disable offline LoRA baking for `*FP8` ops. Upstream supports baking with requantization; local fast online BA behavior is still valuable.
- Flux.2 Small Decoder is a separate VAE/decoder path from TAESD. Deeper model-side review was postponed until after full merge.

## Already Landed

Batch 1 foundation is partly in:

- Launch entrypoint and main thread loop.
- Runtime/settings/hash/cache/logging foundations.
- Compile and mixed precision plumbing.
- New `dynamic_args` compatibility namespace that supports both dict-style and attribute-style access.

Batch 2 inference is in progress and has a sizable first wave:

- Mixed precision ops and quant ops groundwork.
- FP8/int8/manual cast plumbing.
- LoRA loading optimization and compatibility fixes.
- Local FP8 LoRA compatibility restored after upstream-shaped changes.
- Tiled VAE loading path.
- Scheduler and shift plumbing.
- SDXL RF / rectified loader detection updates.
- Flux2 small decoder and Flux dcfg support.
- Wan last-frame / reload state preservation.
- Qwen2D VAE loader support via `backend/nn/wan_vae_2d.py`.
- Precise inpaint mask path and related UI option.
- Anima simplification and alternating prompt support.
- Partial Mugen VAE/config groundwork.

Recent landed commits on top of `neo`:

```text
5522b327 ptl
6a1aa9d9 faster shut down
e68ed036 version
d0f65aae anima alternating prompts
07772cb7 loader
e3edb7f5 vae
800b457f simplify
e0a28637 Batch 2: sync precise inpaint mask and UI cleanup
b727b290 Batch 2: polish VAE overrides and Wan reload state
72eb5e1d Batch 2: add Qwen2D VAE loader support
b03c5461 Batch 2: add dynamic_args compatibility namespace
2e47dd83 Batch 2: keep FP8 LoRA online path enabled
48402cc0 tune
e88a32b7 steps
47de5e06 Spectrum
60e745a2 Batch 2: tighten FP8 mixed-precision LoRA guards
e9bf2fa2 Batch 2: finalize rectified SDXL loader detection
25820c75 Batch 2: sync shift plumbing and Wan last-frame flow
4b587dac Batch 2: sync Flux2 small decoder and Flux dcfg
```

Some upstream commits were attempted and skipped as empty/already covered:

- `4c55a97a small` - current loader already has broader Flux2 small decoder detection.
- `61a925c3 flux key detection` - current `flux_test_keys` already covers the relevant keys.
- `496106b5 donacdum` - local VAE override behavior intentionally differs and is more compatible with current code.

## Hot Zones

Treat these files as manual-synthesis zones:

- `backend/operations.py`
- `backend/operations_mixed_precision.py`
- `backend/patcher/lora.py`
- `modules_forge/packages/comfy/lora.py`
- `modules_forge/packages/comfy/weight_adapter/lora.py`
- `backend/loader.py`
- `modules/sd_models.py`
- `backend/sampling/*`
- `modules/sd_samplers_*`
- `modules/processing.py`
- `backend/nn/vae.py`
- `backend/patcher/vae.py`
- `modules/sd_vae.py`
- `modules/sd_vae_taesd.py`
- `modules_forge/packages/huggingface_guess/detection.py`

Rules of thumb:

- New model family support can be upstream-first.
- Local fixes in Flux2/LoRA/Euler/BOFT/scheduler behavior are local-first unless proven obsolete.
- Loader, scheduler, LoRA, ops, model detection, VAE paths need manual synthesis.
- Bundled configs/assets should land with the code that consumes them.

## Known Merge Followups

- `backend/nn/vae.py` likely needs a final syntax/import sanity pass after the Mugen VAE changes. In particular, verify `F.pad` has `torch.nn.functional as F` imported.
- `backend/nn/anima.py` may have leftover lint noise from direct upstream simplification, including an unused `F` import.
- `backend/patcher/vae.py` has existing `print` lint noise.
- Mugen support is partial at this snapshot: config and VAE flags are present, but full detection/loader wiring likely arrives in later upstream commits.
- Full compile/startup validation was not rerun after the latest direct cherry-picks because intermediate breakage was accepted.

Use `TODO Merging:` for any temporary marker that should be searchable before final validation.

## Recommended Next Session Flow

Start with:

```bash
git switch wip/haoming-neo-sync-20260627
git fetch haoming
git status --short --branch
git log --left-right --cherry-pick --oneline --no-merges HEAD...haoming/neo | awk '{if (substr($0,1,1)=="<") l++; else if (substr($0,1,1)==">") r++} END {print l+0, r+0}'
```

To list remaining upstream commits in cherry-pick order:

```bash
git log --right-only --cherry-pick --reverse --oneline --no-merges HEAD...haoming/neo
```

Suggested approach:

- Continue Batch 2 with larger direct cherry-pick groups.
- Resolve only real conflicts; leave searchable `TODO Merging:` notes where final synthesis is needed.
- Do not overvalidate each intermediate step.
- Before moving to Batch 3, do one focused import/syntax pass over VAE, loader, ops, LoRA, scheduler, and model detection.
- Batch 3 should pull new model families together with required `backend/huggingface/*` assets/configs.
- Batch 4 should pull ControlNet/preprocessor/postprocess pieces together with their UI and backend hooks.
- Batch 5 should handle UX/canvas/preview/infotext/docs/late cleanup.

Final validation target from the original plan:

```bash
./venv/bin/python -m compileall -q backend modules modules_forge extensions-builtin scripts launch.py webui.py
./venv/bin/python -m ruff check <changed python files>
./venv/bin/python launch.py --help
./venv/bin/python launch.py --skip-prepare-environment --skip-install --ui-debug-mode --disable-extra-extensions --port 7861
```
