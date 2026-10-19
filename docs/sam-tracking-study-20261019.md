# Detector cadence study - 2026-10-19

The current default is retained. This study tried periodic semantic detection
with native continuous multiplex tracking, then a conservative contour/confidence
fallback. All versions retain every prepared 15 Hz mask frame and the original
30 Hz motion. No finger-contact stage, robot solve or full FOV job was rerun.
All code and raw masks/logs were isolated outside the production package.

| Independent worker | EgoDex, 5 categories / 133 frames | Real robot, 4 categories / 270 frames |
|---|---:|---:|
| Current optimized default | 102.3 s | 302.0 s |
| Detection keyframes + continuous native tracker | 97.2 s | 115.1 s |
| Conservative contour/confidence fallback | 141.1 s | 130.9 s |
| Dense detection, batch-1 control | 119.0 s | 211.7 s |

These are one-run worker totals, including startup/model loading/serialization
and possible resource-lock waits, not four-GPU complete segmentation-stage time.
The periodic prototype checks every fifth prepared frame, after an initial
18-frame warm start; unconfirmed objects and low tracker confidence request
additional detection. The conservative version also checks contour overlap and
area changes, and revisits a frame with the ordinary detector when needed.
No previous mask is simply repeated as an observation.

The baseline detector batches 16 frames. The prototypes use batch 1 so that
skipped detection frames are not still computed by detector prefetch. A dense
batch-1 control also changes some masks: not all mask changes can be attributed
to cadence. These settings are experimental, not a universal industry frequency.

For EgoDex, mean foreground union IoU relative to baseline is 0.961 for cups and
0.984 for hands, but worst frames are 0.581 and 0.570. Visual inspection shows
an operated cup missing from the mask and local hand-mask expansion; instance
counts also change. The conservative mode does not resolve both quality and
throughput. The real-robot case is much faster mainly because three tested
categories are always empty in the baseline. This cannot be generalized to all
objects, datasets, full 12-category runs, or end-to-end generation.

A supplementary book recording was tested, but this baseline already detects
it from frame zero. The old catalog seed does not establish late appearance,
so this is not presented as validation of genuine late-object recovery.

An additional proposed optimization retained a native grounding batch cache
across same-text resets. It did not produce cache hits: inspection confirmed
that the upstream single-frame `add_prompt` path temporarily disables batched
grounding. That hypothesis is rejected; it is not shipped as an optimization.

Read the machine-readable measurements in
[validation/sam-tracking-study-20261019.json](../validation/sam-tracking-study-20261019.json).
The previous verified execution-only feature reuse remains unchanged. No new
speedup is added to the full-pipeline cost estimate from this study.

Design references: [FoundationPose's initialize-then-track demo](https://github.com/NVlabs/FoundationPose#run-model-based-demo)
and [SAM 3.1 multiplex release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md).
These support the architecture, not this pipeline's timing or quality claims.
