# Point-cloud playback repair — 2026-10-19

The repair changes delivery and playback only. Existing geometry bundles, model
predictions, segmentation, registration, motion, FOV and other stored caches are
retained. It neither reruns inference nor reduces displayed point density.

## Cause

- A six-frame prefetch at 15 Hz provided only 0.4 seconds of runway. A missing
  observation paused the shared motion/video clock; the first late frame resumed
  playback immediately and could trigger another pause on the very next frame.
- JPEG decoding and dense calibrated unprojection ran on the UI thread.
- The running service predated the sequential RGB proxy reader present on disk.
  Its frame endpoint was too slow for sustained playback even with six requests.

## Changes

- Prefetch up to 45 frames (3 seconds), including the next loop's beginning.
  Start/resume continuous playback with 18 contiguous observations (1.2 seconds
  at 1x). Paused seeks require only the selected observation. Frame timestamps
  remain exact; buffering never substitutes another observation's geometry.
- Cap decoded browser buffers at 256 MiB and shrink the prefetch horizon for
  unusually dense frames. Keep seek cancellation, explicit retries and disposal.
- Decode JPEG and unproject depth in a module Worker, transferring typed arrays
  without copying their contents back to the UI thread.
- Cache wire packets in a disposable 128 MiB server LRU. Permission checks still
  precede cache reads. Revision-qualified requests may use a private browser cache
  for five minutes; a mismatched geometry revision is rejected.
- Restart the idle service to activate its existing sequential RGB reader.

## Verification

On the same 30 ice-sequence frame requests (indices 135–164), via local HTTP:

| Warm-service run | Before | After |
|---|---:|---:|
| Sequential, total | 66.697 s | 0.317 s |
| Sequential, mean response | 2.223 s | 0.0104 s |
| Six concurrent requests, total | 12.753 s | 0.658 s |
| Mean compressed payload | 140,347 bytes | 140,347 bytes |

These are delivery timings with warm dependencies, not reconstruction timings
or a guarantee for every network. Cached and uncached responses may differ.

API packets at frames 0, 137, 398 and 528 were byte-identical after decompression
to canonical packet construction, including depth, labels and the separate human
layer. Canonical geometry index hashes stayed unchanged. A stale revision returned
404. Both response-cache tests and 21 frontend tests passed; the isolated production
build passed. Other uncommitted work was excluded from that frontend build.

In-browser ice playback reached 30.54 seconds with zero recorded post-start stalls.
After adding loop prefetch, playback also crossed back to the beginning with zero
recorded stalls. The counters measure data starvation, not display refresh rate.
