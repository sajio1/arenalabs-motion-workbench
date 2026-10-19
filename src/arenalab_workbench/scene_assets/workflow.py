"""Isolated stage DAG, failure receipts and content-addressed robot-free assets."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import subprocess
import sys
import time

from . import POLICY
from .formats import sha256, write_json


def run(work, output, model, converter, parameters=(), window=32, overlap=8, stride=1,
        sam_source=None,sam_config=None,foundation_source=None,object_python=None):
    from arenalab_workbench.gpu_scheduler import run as gpu_run
    work,output,model,converter=map(Path,[work,output,model,converter])
    if any(p is None for p in [sam_source,sam_config,foundation_source]):
        raise ValueError("SAM 3D and FoundationPose must be explicitly provisioned")
    if not parameters:
        raise ValueError("The full workflow requires immutable human/motion input artifacts")
    for path in [work/"manifest.json",model/"config.json",converter]:
        if not path.is_file():
            raise ValueError(f"Missing provisioned input: {path}")
    identity=dict(policy=POLICY,work_manifest_sha256=sha256(work/"manifest.json"),
                  frames_sha256=sha256(work/"frames.npz"),
                  mask_sha256={p.name:sha256(p) for p in sorted((work/"masks").glob("*.npz"))},
                  frames_rgb_sha256={p.name:sha256(p) for p in sorted((work/"frames").glob("*.jpg"))},
                  camera_sha256={name:sha256(work/name) for name in ["cameras.npz","camera-intrinsics.npz"] if (work/name).exists()},
                  model_sha256={p.name:sha256(p) for p in model.glob("*.safetensors")},
                  code_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob("*.py")},
                  converter_sha256=sha256(converter),window=window,overlap=overlap,stride=stride,
                  parameters_sha256=[sha256(p) for p in parameters])
    identity.update(sam_config_sha256=sha256(sam_config),sam_source=str(sam_source),foundation_source=str(foundation_source),
                    object_python=str(object_python) if object_python else None)
    sam_root=Path(sam_config).parent
    identity.update(sam_weights_sha256={p.name:sha256(p) for p in sorted(sam_root.iterdir())
        if p.suffix in {".ckpt",".safetensors",".yaml"}},
        foundation_weights_sha256={str(p.relative_to(foundation_source)):sha256(p)
            for p in sorted((Path(foundation_source)/"weights").rglob("*")) if p.is_file()},
        object_worker_sha256=sha256(object_python) if object_python else None)
    if output.exists():
        existing=output/"identity.json"
        if not existing.exists() or json.loads(existing.read_text())!=identity:
            raise ValueError("Job identity changed; choose a new isolated output directory")
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"identity.json",identity)
    base=[sys.executable,"-m","arenalab_workbench.scene_assets"]
    def stage(name,arguments,gpu=False):
        receipt=output/f"{name}-status.json"
        if receipt.exists() and json.loads(receipt.read_text()).get("status")=="completed":
            return
        # Keep interrupted attempts intact. A failed stage may be retried, but
        # must never overwrite its raw model prior, log, or partial assets.
        attempt=str(time.time_ns())
        if receipt.exists():
            receipt.rename(output/f"{name}-{attempt}-previous-status.json")
            directory=output/({"package":"assets"}.get(name,name))
            if directory.exists():
                directory.rename(output/f"{name}-{attempt}-previous-output")
            old_log=output/f"{name}.log"
            if old_log.exists():
                old_log.rename(output/f"{name}-{attempt}-previous.log")
        started=time.monotonic()
        write_json(receipt,dict(status="running",started_at=time.time()))
        with (output/f"{name}.log").open("w") as log:
            command=([str(object_python),"-m","arenalab_workbench.scene_assets"] if name=="objects" and object_python else base)
            result=(gpu_run if gpu else subprocess.run)(command+list(map(str,arguments)),
                        env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        status=dict(status="completed" if result.returncode==0 else "failed",seconds=time.monotonic()-started,
                    returncode=result.returncode,log=f"{name}.log")
        write_json(receipt,status)
        if result.returncode:
            raise RuntimeError(f"{name} failed; original log retained at {output/name}.log")
    stage("geometry",["reconstruct",work,output/"geometry","--model",model,"--window",window,
                      "--overlap",overlap,"--stride",stride],True)
    # Independent GPU workers share immutable geometry, not robot-specific data.
    with ThreadPoolExecutor(max_workers=2) as pool:
        geometry=output/"geometry/geometry.zarr"
        objects=pool.submit(stage,"objects",["objects",geometry,output/"objects","--sam-source",sam_source,
            "--sam-config",sam_config,"--foundation-source",foundation_source,"--work",work],True)
        stage("splats",["splats",geometry,output/"splats","--converter",converter],True)
        objects.result()
    stage("package",["pack",work,geometry,output/"splats",output/"objects",output/"assets",
                     "--parameters",*parameters])
    write_json(output/"status.json",dict(status="experimental_assets_ready",policy=POLICY,
               assets="assets/manifest.json",delivery_renderer_status="scene renderer available; full URDF/LeRobot join not integrated",
               online_page_modified=False,da3_baseline_modified=False))
