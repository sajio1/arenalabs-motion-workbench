# AlphaMotion acceleration measurements — 2026-10-18

This isolated experiment used one idle RTX A6000 48 GB, the installed Prime
checkpoint, six observed human landmarks, and the existing H2 descriptor.
It did not run finger-contact estimation or change model weights, precision,
guidance, projection, support refinement, or operation-refinement objectives.
Raw measurements are in `validation/alphamotion-speed-20261018.json`.

## Model inference and downstream refinement are different costs

Warm encoding plus decoding, with the existing 32-frame batch size, took:

| Input | Frames at 30 Hz | Encode | Decode | Total |
|---|---:|---:|---:|---:|
| EgoDex | 266 | 0.0585 s | 0.1307 s | 0.1892 s |
| Real robot | 539 | 0.1099 s | 0.2577 s | 0.3676 s |

These are medians of three synchronized CUDA measurements per operation after
warm-up, with model and descriptors resident. They exclude human reconstruction,
process startup, IO, projection and all refinement. Checkpoint loading alone
took 1.006 s in this experiment; existing full-job startup timers, including
descriptor/support setup, were 2.44–2.57 s.

The preceding full-generation cost study measured final operation refinement at
21.35 s for the EgoDex sample and 39.71 s for the real-robot sample. Both used a
CUDA-graph objective and reached the 1,600-iteration limit; neither reported
optimizer convergence. The additional support and body-coordination stages
were about 6 s and 4.5 s per sample. These are not neural forward-pass times.

## Exact-output cache fix

The old `decode_cached` repeatedly accessed compressed NPZ members inside its
per-frame loop. NumPy reinflates a member on each such access. It also copied
cached outputs to CUDA one frame at a time.

The fix materializes each member once per file and assembles results with
batched transfers and indexed writes. Model/config/target identity, exact
integer-code matching, frame IDs, timestamp checks and last-valid-file ordering
remain unchanged. Missing frames retain the original 32-frame decoder batches.

On the real-robot sample, with 299 reused and 240 newly decoded frames:

| Implementation | Repetitions | Median wall time |
|---|---:|---:|
| Original | 3 | 4.6046 s |
| Fixed | 5 | 0.3308 s |

This stage improved **13.92×**, saving 4.274 s. Rotation, position and contact-head
arrays were bit-for-bit equal against the original implementation; cache-hit
counts were also identical. The model contact head here is an existing decoder
output, not a run of the excluded finger-contact estimation stage.

Unit tests additionally cover no hits, full hits, shuffled partial hits, wrong
codes, wrong timestamps and a different target identity. Existing prior files
remain untouched. This measurement does not imply a 14× whole-pipeline speedup:
most generation time is spent in other stages, particularly SAM segmentation.

## Batch-size experiment: retained the existing default

Increasing batch size from 32 to 256 reduced warm encode+decode from 0.1892 to
0.1535 s for EgoDex, and from 0.3676 to 0.3040 s for the real sample. The tested
larger batches did not produce exactly the same integer encoding as batch 32.
Decoder outputs for fixed stored codes also had small floating-point differences.
The benchmark reports these comparisons rather than assuming exact equivalence.
The default batch size therefore remains 32; final motion quality for larger
batches was not validated.

## Further engineering opportunities, not implemented by this change

- Keep the model and descriptors resident across jobs, with scheduler ownership
  of that memory. The current admission policy rejects occupied GPUs, so a
  persistent worker cannot simply be added independently of the scheduler.
- Reduce CPU/GPU exchanges between SciPy L-BFGS-B and the CUDA objective. This
  requires separate quality and convergence validation; reducing iterations is
  not an equivalent acceleration.
- Improve committed-window reuse while preserving globally consistent source
  normalization and the exact-code matching rule. Do not relax identity checks
  merely to increase cache hits.

No claim of all-robot equivalence or one-hour end-to-end throughput follows from
these two samples. The cache assembly itself is generic across robot descriptors.
