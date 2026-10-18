# Workbench validation — 2026-10-18

## Six-point rollout

The active AlphaMotion human input policy is
`human-six-pelvis-head-hands-feet-v1`: pelvis, head, left/right hand and
left/right foot. Frames remain independent model batch items; the original
windowed human reconstruction and robot refinement remain separate stages.

All 58 cached manipulation/locomanipulation source–robot combinations finished
motion and visual generation. All 29 existing locomotion conversions were
regenerated. The actual published robot/playback metadata was checked for all
87 outputs, with zero stale policy entries. Previous raw outputs and failed
attempts were retained separately from the active cache.

Four representative encoder cases matched the earlier offline six-point
results exactly at the discrete code level. Independent-frame and batched
encoding matched; changing masked joint channels did not change the codes.
This validates visibility and batching, not physical feasibility.

## Upload and download

A 90-frame real EgoDex observation package was imported through the normalized
recording contract and passed the full human → six-point H2/Wuji → DA3/SAM →
FOV → contact path. Mean hand reference error was 1.72 mm; this excludes scene
alignment, fingertip and contact error.

The official LeRobot 0.4.3 reader loaded the 90-frame uploaded result and a
450-frame Nymeria result, including episode subset selection, float32 states
and actions, image statistics and original/inpaint video samples. The included
extension reader decoded finite cloud coordinates and instance labels.

The deployed HTTPS export endpoint additionally packaged the 266-frame preset
H2 episode. Its ZIP download returned a valid HTTP range response, and another
browser workspace received 404 for the export. Secure workspace cookies and
the packaging-template endpoint were checked.

Twelve backend tests cover upload validation, path boundaries, workspace
ownership, empty SAM tracker recovery and missing recorded-camera distortion
metadata. Fourteen frontend tests and the production TypeScript build passed.

These are kinematic datasets. Generated states are not measured telemetry;
next-frame actions are references, and estimated contact is not pressure.
