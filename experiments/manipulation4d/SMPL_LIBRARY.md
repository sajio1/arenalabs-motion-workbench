# Human motion assets

The human motion is a persistent source asset, independent of robot-result
caching. `smpl_library.py` builds the current manipulation catalog on physical
GPU 3, using existing source adapters and body inference. UMI is excluded until
the user provides its height. Duplicate source/window entries share one asset.
No robot-specific solver policy is introduced.

Each asset descriptor references the unchanged `smplx_motion.npz`,
`canonical.npz`, `egoallo_smplh.npz`, `observations.npz`, and observed source hand
trajectories where available. Hashes, source frame window, height/floor input,
head calibration, timestamps and quality diagnostics accompany the files.
The asset index lives under `human-motion-library/index.json`, separate from
the viewer catalog and robot caches. Existing valid priors remain in their
original directories, which the current robot-generation entry points reuse.

The same SMPL-X body can be converted to another robot. This still requires
AlphaMotion inference and robot refinement; it does not make those free.
Observed palms/wrists must remain the precision targets. Reconstructed human
hands must not replace the observed source fingers.

Changing source height/floor, the source window, observations or the body
inference policy requires another human asset. Height adjustments change the
declared floor input; source-world task trajectories stay unchanged. Preserve
the previous asset. UMI height variants and robot results are separate.

The interactive scheduling order should be:

1. Resolve the existing human asset, or reconstruct and save it.
2. Generate robot motion and joint states; allow playback as soon as ready.
3. Generate current-frame depth, segmentation and point clouds.
4. Render source-FOV robot arms and hands, after motion and segmentation exist.
5. Estimate per-digit contact, after the required observations exist.

Already-present outputs can be displayed immediately. Contact inference depends
on its actual inputs, not necessarily on the composited robot video. Per-stage
queued/running/ready/failed states should be independent. This isolated batch
does not modify the running viewer's scheduling or UI.

Generation is not a quality certificate. The audit records finite values, frame
counts, hashes, pelvis height, declared floor and mesh penetration. These are
reconstructions, not ground truth or verified physically executable motions.

Run `test_smpl_library.py` for provenance/invalidation checks. The CLI supports
`plan`, `run` and `infer`; a run is resumable and retains per-source failures.
The scheduler leases GPU 3 per inference, so other tasks are not terminated.
Use a frozen copy of the current pipeline modules as `--code-dir` for a batch.
