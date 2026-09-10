# Weight lifetime and offline LoRA regression checks

## Krea INT8 autotune disk cache

Forge's two Triton INT8 GEMMs cache tuning results on disk, keyed by 64-row
buckets (exact M below 64), N, K, bias presence and argument dtypes. Real M is
unchanged; there is no padding. Triton also includes its version/backend target,
kernel source, relevant environment and candidate configs in the disk cache key.

Prewarm the main DiT INT8 shapes found in the current mixed Krea checkpoint:

```powershell
.\venv\Scripts\python.exe scripts/diagnostics/warmup_krea_int8.py
.\venv\Scripts\python.exe scripts/diagnostics/warmup_krea_int8.py --verify-cache-only
```

Defaults: BF16, batch=1 text-to-image, image width and height each 960..1216
(multiples of 16), 0..512 prompt tokens plus 5 remaining template positions.
This covers all combined row counts 3605..6293, 43 buckets and four unique INT8
shapes in `krea2_turbo_int4_tensorwise_mixed.safetensors` (172 combinations).
Prompt tokens mean the complete active prompt after schedule selection, not
only the inserted fragment; characters are not a token limit. Additional
emphasis/template segments, references, larger batches/CFG batching or different
dtypes can require additional entries. INT4 and attention are not prewarmed.

The script uses synthetic data and reads only checkpoint metadata/shapes. It
needs no TE, model load, API or Gradio. The first command resumes through existing
disk entries; the second uses a different M within each bucket in a fresh process
and raises if any autotune benchmark is attempted. It can still compile a new JIT
specialization: successful verification proves absence of autotuning, not zero
startup latency. Tuning near the bucket centre is a heuristic, not a guarantee
that every M in a bucket receives its individually fastest configuration.

The script prints the effective Triton cache directory. Run it with the same
venv, GPU and cache environment as the server. Rewarm after cache-invalidating
kernel/Triton changes. Out-of-range inputs remain supported and tune on demand.

## Weight lifetime checks

The production fix uses safetensors `pread` on Windows (and with
`--disable-mmap`) and retains parameter names, rather than Parameter objects,
in ModelPatcher loading/unloading lists. LoRA backups and merge order are
unchanged.

Run the CUDA regression checks from the repository root:

```powershell
.\venv\Scripts\python.exe scripts/diagnostics/test_weight_lifetime.py
```

Checks cover loader backend selection/metadata/device, release of replaced
payloads while the loading list is still alive, actual Forge patcher clones,
two different low-rank adapters with overlapping/disjoint layers, repeat and
strength changes, multiple adapters, unload/reload, and byte-exact restoration.
Formats tested: ConvRot INT4, INT8, FP8, FP16 and BF16.

For an event-driven API regression run without opening a browser:

```powershell
.\venv\Scripts\python.exe scripts/event_memory_probe.py --internal-events --lora-sequence --skip-full-size --output event-memory-lora.jsonl -- --nowebui --port 7860 --sage --xformers --uv --fast-fp8 --adv-samplers
```

This runs base → existing `diamel-v2.2-krea2-000012` → a temporary derived
three-layer adapter with different coefficients → base → diamel, with a fixed
seed. The temporary adapter is removed after the child stops. Drop
`--skip-full-size` to make the first request 1408×1024, 12 steps, Euler a/Krea2,
CFG 1. Subsequent switching requests use 512×512 and one step.
PNG is selected only in the disposable process because the API encoder does
not support every UI output format (notably AVIF). The config file is not saved.

`--internal-events` uses a separate launch wrapper, never imported by Forge's
top-level scripts discovery. Its runtime wrappers report model payload bytes,
original tensor objects still alive, LoRA backups and CUDA allocated/reserved
memory. Weak references do not retain source weights. Object lifetime and
storage lifetime differ for ordinary Parameters mutated by PyTorch; device
and deduplicated storage counts are reported alongside the object observations.

## Verified results

- `event-memory-fixed-verified.jsonl`: all five API calls succeeded. Base image
  SHA-256 before/after LoRA was identical; first/repeated diamel image SHA-256
  was identical; base, diamel and temporary adapter produced different images.
- Cold 512×512 baseline OS process-private peak: **18.805 GiB**, versus the
  earlier measured **26.240 GiB** with mmap and retained loading references.
  Reduction: **7.435 GiB**. During transfer, old CPU payloads shrink with each
  transferred layer, instead of staying near 8 GiB until load returns.
- The complete LoRA sequence reached **31.350 GiB process private**, with
  **44.850 GiB system commit**. Full diamel retains **7.998 GiB CPU backups**;
  the three-layer fixture retains **0.0616 GiB**; removing LoRA leaves no backups.
- `event-memory-fixed-lora.jsonl`: the 1408×1024 / 12-step generation completed
  and returned HTTP 200; that earlier run subsequently exhausted commit while
  enabling full diamel.
- Earlier full-diamel attempts failed with the old **39.878 GiB Commit Limit**.
  The successful sequence used the subsequently observed **51.878 GiB limit**.
  The implementation/test tools did not change pagefile settings. The fix does
  not guarantee that full offline LoRA backups fit the old limit.

The probe records errors as errors rather than zero memory, uses lightweight
`memory_info()`, and records both sampled maxima and the OS process peak.
Process-tree peak sums are upper bounds when multiple substantial children
have peaks at different times; system commit is always a separate metric.

## Krea 2 block benchmark

To compare a representative Krea 2 block at the token count corresponding to
a 1408x1024 output, without loading the complete model or requiring a BF16
checkpoint:

```powershell
.\venv\Scripts\python.exe scripts/diagnostics/benchmark_krea2_block.py --sage
```

The default block 1 has the checkpoint's modal projection mix (five INT4 and
three INT8 linears).  Its quantized forward is compared with the same block
after its checkpoint weights have been dequantized to BF16 outside the timed
region.  This measures runtime and memory, but cannot measure fidelity against
the unavailable original BF16 master weights.  The default prompt fixture has
9 text tokens after template stripping, for 5641 total tokens (5632 image + 9
text).  The image token grid is 88x64: the VAE downsamples spatially by 8 and
the transformer uses 2x2 latent patches.  The four BF16 text-fusion blocks are
also measured in their actual layerwise/refiner shapes and included in the
reported network estimate.  Environment variables documented at the top of
the script control the block, dimensions, prompt/text length, warmup and
repeat count.
