# Output selection API and web validation

API version: **1.3.0**. 61 API/SDK/viewer tests and 43 Node tests passed. TypeScript and the Vite production build succeeded. Logs and source/build SHA256 hashes are in this directory.

The public request accepts a nonempty, unique subset of trajectory, point_cloud, contact, and inpaint_fov. Omission means all available outputs. Planning resolves source capabilities separately from internal dependencies. Explicit selection participates in idempotency and export variant identity; polling preserves the selected outputs and partial published results after a later stage fails.

The selector appears after robot selection, offers Select all available, and disables unsupported outputs with source-based reasons. Select all never requests unavailable outputs. Individual FOV selection remains FOV in the request; its trajectory and point-cloud dependencies are shown in the plan rather than being returned as selected outputs.

## Robot-free result snapshots

The public scene can contain `robots=[]` for point-cloud, contact, and FOV-only results. The existing scene loader clears native robot playback before loading source geometry. FOV video selection is now independent of `loadRobot`: it binds only an already-published patch URL for a registered robot and clears a stale patch when a new selected result has no FOV. Contact animation updates on its source clock before the robot-playback early return. Contact visibility is based on the contact artifact, not the presence of a Wuji robot trajectory. Native joint export stays disabled without native playback. API artifact enumeration follows public playback URLs and does not expose dependency-only trajectory files.

Functional Node tests exercise FOV-only scenes without robot data, stale/unregistered FOV rejection, and contact-only sample progression using the actual UI helper. They do not constitute a full browser rendering test.

## Limitations

No browser screenshot or interactive visual QA was possible: the available CUA surface reported no browsers. Existing build warnings concern static asset paths and a bundle larger than 500 kB. No numerical solver, source motion, or generated action quality changed in this UI/API work. The root agent performs deployment and cloud validation separately.
