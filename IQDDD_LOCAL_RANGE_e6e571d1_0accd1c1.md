# iqddd Local Range Notes

Scope reviewed:

- `e6e571d10539b93ef5452e96eb03125b80a3441d`
- through
- `0accd1c16015a427179121c35aab91fca700b4ac`

This file documents the local functionality that should be treated as the
actual "iqddd-owned" delta during the Haoming `neo` merge.

Merge rule to use from here:

- If an upstream change does not intersect with the functional areas below, take upstream.
- If an upstream change does intersect, resolve case-by-case in favor of preserving the behavior described here unless upstream clearly supersedes it.
- The only intentionally low-priority item in this range is the short Flux2 Automatic debug trace from `2867a1bf`; it is diagnostic, not core functionality.

## Functional Areas

### 1. Flux2 TAESD / latent adapter support

Primary commits:

- `e6e571d1` Apply local neo changes
- `877ecf8e` Fix Flux2 TAESD compatibility and latent adapters
- `cd52d01a` Use no-BN path for Flux2 TAESD latent adapters

Files:

- `backend/diffusion_engine/base.py`
- `backend/diffusion_engine/flux2.py`
- `modules/sd_vae_taesd.py`

Behavior to preserve:

- `ForgeDiffusionEngine` gains `is_flux2`.
- `Flux2` models explicitly set `is_flux2 = True`.
- TAESD recognizes Flux2-specific assets (`taef2_*`) and routes Flux2 through dedicated encoder/decoder handling.
- Flux2 TAESD uses packed/unpacked latent adapters between 32-channel TAESD internals and 128-channel Flux2 latents.
- Flux2 TAESD mid-blocks add the special GN-assisted path introduced in `877ecf8e`.
- Final intended behavior is the no-BN variant from `cd52d01a`: the temporary dependency on VAE batchnorm state added in `877ecf8e` is removed again and should not come back.

Practical merge note:

- Any upstream TAESD, decoder, encoder, latent adapter, or `is_flux2` changes intersect with this range.

### 2. Flux2 Automatic sigma schedule

Primary commits:

- `139eca36` Fix Automatic sigma schedule to avoid duplicate terminal zero
- `6fb77174` Use BFL Flux2 Automatic schedule for sigma generation
- `2867a1bf` Add short Flux2 Automatic trace for mu and sigma slices

Files:

- `modules_forge/packages/k_diffusion/external.py`
- `modules/sd_samplers_kdiffusion.py`

Behavior to preserve:

- Automatic sigma generation must not create duplicate terminal zero after `append_zero()`.
- Flux2 "Automatic" schedule should use the BFL/empirical schedule path instead of generic sigma generation.
- Flux2 Automatic computes image sequence length from effective latent resolution and VAE downscale ratio.
- Flux2 Automatic uses the empirical `mu` curve and generalized time/SNR shift schedule.
- Generation params should record Flux2 Automatic schedule metadata.

Low-priority detail:

- `2867a1bf` only adds a short debug print of `mu` and sigma head/tail slices. This is useful for tracing but is not core runtime behavior.

Practical merge note:

- Any upstream changes in Flux2 scheduler selection, Automatic scheduler behavior, sigma generation, or k-diffusion schedule linking intersect with this range.

### 3. Fast FP8 LoRA and fp8_scaled offline LoRA correctness

Primary commit:

- `598f0e73` Fix LoRA behavior in fast-fp8 and fp8_scaled offline path

Files:

- `backend/operations.py`
- `backend/patcher/lora.py`
- `extensions-builtin/sd_forge_lora/networks.py`

Behavior to preserve:

- Fast FP8 path can apply a restricted online LoRA residual for classic LoRA adapters by composing low-rank A/B matrices in `_fp8_prepare_online_lora()`.
- Fast FP8 falls back cleanly when online adapters are unsupported instead of silently misbehaving.
- fp8-scaled offline LoRA path preserves correct base dtype handling for unscaled BF16 weights inside mixed/fp8 checkpoints.
- fp8-scaled layers can re-quantize merged weights and recompute `scale_weight` inside `set_weight()`.
- Stochastic rounding helper logic was added for FP8 offline re-quantization.
- `LoraLoader` backs up auxiliary `scale_weight` parameters and restores them when switching LoRA sets.
- `LoraLoader` uses `convert_*` / `set_*` hooks when present, instead of treating quantized layers like plain parameters.
- `LoraLoader` seeds quantized restore/update paths deterministically from the layer key.
- The old blanket "disable online LoRA for FP8" behavior in `extensions-builtin/sd_forge_lora/networks.py` was intentionally removed by this local range. The point of this range is to keep the fast online FP8 LoRA path working, not to turn it off.

Practical merge note:

- Any upstream changes in `backend/operations.py`, `backend/patcher/lora.py`, or `extensions-builtin/sd_forge_lora/networks.py` intersect with this range if they touch FP8, mixed precision, quantized weight hooks, online LoRA gating, or LoRA merge/update semantics.

### 4. Euler Negative RF sampler

Primary commits:

- `f99b0498` Add Euler Negative RF sampler with strict s_tmax guard
- `660aef83` Remove Euler Negative RF diagnostics and keep smooth churn cap

Files:

- `modules/sd_samplers_kdiffusion.py`
- `modules_forge/packages/k_diffusion/sampling.py`

Behavior to preserve:

- A new sampler named `Euler Negative` is registered.
- It maps to `sample_euler_negative_RF`.
- `sample_euler_negative_RF` is a dedicated negative Euler sampler for linear Rectified Flow.
- `s_tmax` is sanitized/clamped to a safe `(0, 1)` RF range.
- The later commit keeps the sampler but removes noisy diagnostics and adds the smooth churn cap behavior:
  - `negative_rf_churn_curve_power`
  - weak churn near `t ~ 1`
  - more aggressive churn at lower `t`
  - `gamma` capped by a smooth target `t_hat`

Practical merge note:

- Any upstream changes to sampler registration, RF samplers, Euler variants, churn handling, or k-diffusion RF stepping intersect with this range.

### 5. CFG++ sigma-dependent CFG schedule

Primary commit:

- `0accd1c1` CFG++: sigma-dependent CFG schedule

Files:

- `modules/sd_samplers_cfg_denoiser.py`
- `modules/shared_options.py`

Behavior to preserve:

- When sampler name ends with `CFG++`, CFG scale becomes sigma-dependent during sampling.
- New options:
  - `cfgpp_low_cfg`
  - `cfgpp_beta`
- Effective `cond_scale` becomes:
  - `low + (high - low) * (sigma_norm ** (2 * beta))`
- Metadata for `CFG++ low CFG` and `CFG++ beta` is recorded in generation params on first step.

Practical merge note:

- Any upstream changes to CFG++, CFG denoiser behavior, sigma-conditioned CFG scheduling, or these options intersect with this range.

## Commit-by-Commit Summary

### `e6e571d1` Apply local neo changes

- Adds `is_flux2` plumbing.
- Adds Flux2 TAESD asset detection (`taef2_*`) and latent-channel selection.

### `139eca36` Fix Automatic sigma schedule to avoid duplicate terminal zero

- Prevents duplicated trailing zero in sigma schedules when `append_zero()` is already used.

### `6fb77174` Use BFL Flux2 Automatic schedule for sigma generation

- Introduces Flux2-specific Automatic sigma generation based on BFL empirical schedule logic.

### `2867a1bf` Add short Flux2 Automatic trace for mu and sigma slices

- Adds debug print only.
- Preserve only if useful; safe to drop if upstream already provides equivalent observability or if noise is undesirable.

### `877ecf8e` Fix Flux2 TAESD compatibility and latent adapters

- Introduces Flux2 TAESD latent packing/unpacking adapters and Flux2-specific TAESD architecture path.

### `598f0e73` Fix LoRA behavior in fast-fp8 and fp8_scaled offline path

- Core local FP8/LoRA correctness patchset.

### `cd52d01a` Use no-BN path for Flux2 TAESD latent adapters

- Removes the temporary BN-based adaptation from the previous TAESD patch.
- This later state is the one to preserve.

### `f99b0498` Add Euler Negative RF sampler with strict s_tmax guard

- Introduces the new sampler and RF stepping logic.

### `660aef83` Remove Euler Negative RF diagnostics and keep smooth churn cap

- Refines the Euler Negative RF sampler.
- Preserve this later refined behavior, not the earlier raw/diagnostic form.

### `0accd1c1` CFG++: sigma-dependent CFG schedule

- Adds sigma-shaped CFG scheduling and user options for CFG++.

## Files That Should Trigger Manual Review

- `backend/operations.py`
- `backend/patcher/lora.py`
- `extensions-builtin/sd_forge_lora/networks.py`
- `modules/sd_vae_taesd.py`
- `modules/sd_samplers_kdiffusion.py`
- `modules_forge/packages/k_diffusion/external.py`
- `modules_forge/packages/k_diffusion/sampling.py`
- `modules/sd_samplers_cfg_denoiser.py`
- `modules/shared_options.py`
- `backend/diffusion_engine/base.py`
- `backend/diffusion_engine/flux2.py`

## Safe Upstream-First Areas

Anything outside the functional overlap listed above can be taken from upstream aggressively.

In particular, if a conflict does not touch:

- Flux2 TAESD / latent adapter behavior
- Flux2 Automatic sigma schedule behavior
- FP8 / mixed / quantized LoRA merge behavior
- Euler Negative RF sampler behavior
- CFG++ sigma-dependent CFG schedule

then it does not belong to the reviewed iqddd-owned local range and should default to upstream.
