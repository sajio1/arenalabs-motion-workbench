# ArenaLabs Motion Workbench

Turn synchronized egocentric recordings into robot motion and a downloadable
multimodal manipulation episode. The web workbench combines human-motion
reconstruction, cross-embodiment motion conversion, scene perception and
robot-view compositing. This repository contains inference/integration code;
it does not include pretrained weights, private recordings or generated caches.

**Current robot conditioning:** pelvis, head, left/right hand and left/right
foot — six observed human joints. AlphaMotion predicts frames independently;
body reconstruction and the robot refiner supply temporal structure. Changing
the target robot reuses the saved human prior and scene estimates.

## Start here

- [Complete workbench guide](docs/workbench-guide.md): what each stage does,
  supported inputs, outputs, cloud use, API examples, deployment and limitations.
- [Performance audit](docs/performance-audit-20261018.md): actual CPU/GPU paths,
  measured bottlenecks, applied fixes and remaining optimization work.
- [SAM acceleration measurements](docs/sam-speed-20261018.md): bounded feature
  reuse, exact-mask checks and measured improvements in GPU occupancy.
- [Recording upload / LeRobot export contract](docs/dataset-io.md): exact arrays,
  axes, units, clocks, packaging and browser workspace isolation.
- [Point-cloud direction](docs/point-cloud-direction.md): depth-only backend
  comparisons and compact depth/camera storage after retiring scene assets.
- [Weights and model release](docs/model-release.md): what belongs on Hugging
  Face, what must be obtained upstream and what a model release needs.
- [Validation](docs/release-validation.md): six-point migration and dataset
  reader checks. Generation does not certify physical execution or task success.

## End-to-end path

```text
Video + measured head/camera and bilateral hand/gripper trajectories
    ├── windowed EgoAllo (CUDA PyTorch + CUDA JAX) → SMPL-H → SMPL-X
    │       → six-point AlphaMotion → original robot refinement → joint states
    └── video frames + optical camera poses → MapAnything depth
            + Qwen object proposals → SAM 3.1 masks
            → compressed RGB-D + RLE / cameras → on-demand cloud and 3D boxes
Robot motion + camera + masks + depth → synthetic robot arm/hand FOV composite
Observed hands + video / object points → contact estimates / proximity indices
Saved modalities → LeRobot v3 episode + lossless ArenaLabs extension ZIP
```

EgoDex observed finger landmarks map separately to Wuji. Nymeria uses no
invented dexterous fingers. UMI and real-robot adapters use observed end-effector
references and gripper openings; recorded robot body, pelvis and feet are not
used as a human prior. A calibrated camera/head reference is still required.

## Use the hosted service

Open the workbench, choose a preset or **Upload data**, select a robot and end
effector, and download the generated episode using **Download dataset**. Upload
requires an explicit recording package, not a bare video. Users do not need
SSH access or local model weights to use a service whose owner supplies them.
The current browser workspace cookie is not a full account or quota system.

## Develop or self-host

Core Python code requires Python 3.10+ and CUDA for the full-body model. The
current complete cloud deployment uses separate environments for the API,
EgoAllo/JAX, MapAnything, optional DA3 and SAM/Qwen/rendering. Do not install all model dependencies
into one environment and assume they are compatible.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[runtime,viewer,dataset,geometry,test]"
python -m greenwich_pipelines list
cd web
npm ci
npm run build
```

These commands install/build the core package and UI. They do **not** download
weights or prepare a complete cloud installation. Follow the deployment section
in the [workbench guide](docs/workbench-guide.md); deployment paths are configured centrally in
`src/arenalab_workbench/settings.py` with environment overrides. This is not yet a portable
one-command installer or a published Docker image.

See the [cleanup and migration notes](docs/workbench-cleanup.md) for the package
move, retired implementations and deployment entry-point change.

## Repository map

| Path | Responsibility |
|---|---|
| `src/arenalab_workbench/` | Current workbench coordinator, adapters, human windows, perception, rendering, exports |
| `src/greenwich_pipelines/input_policy.py` | Shared six-point visibility mask |
| `src/greenwich_pipelines/frame_cache.py` | Exact-code reuse of streamed AlphaMotion outputs |
| `src/greenwich_pipelines/pipelines/fullbody/` | Native full-body decoder/projection and original refinement |
| `src/alphamotion/` | Model inference and mechanical descriptors |
| `src/greenwich_motion_sdk/` | Human motion formats and geometry |
| `web/src/manipulation4d/` | Interactive viewer, upload and dataset downloads |
| `configs/`, `examples/` | Core runtime/job and mechanical metadata examples |
| `tools/` | Explicit maintenance and read-only performance audit commands |
| `provenance/`, `validation/` | Migration sources and historical checks |

The original standalone Nymeria/Gento CLI adapters remain available for legacy
experiments; their historical endpoint policies are not the current web
workbench path. See [legacy runtime notes](docs/legacy-runtime-guide.md).
Retired reconstruction and video-finger experiments live in Git history; they
are not shipped as alternative production routes.

## Models, licensing and data

Model files stay outside Git. AlphaMotion currently loads `config.json` and
`model.safetensors` from the runtime configuration. Third-party body templates,
models, meshes and datasets retain their own terms; public repository visibility
does not grant permission to redistribute them. No project-wide license for
all first-party code/weights has been selected in this repository. See the
[release guide](docs/model-release.md) before redistributing model bundles.
