#!/usr/bin/env python3
"""Explicit cloud Docker worker; no network, model downloads or service changes.

Use as --object-python after explicitly building scene-assets-models.Dockerfile.
Docker permissions are established by the administrator, not escalated here.
"""
from pathlib import Path
import os
import subprocess
import sys


def command(arguments, env=None):
    env=os.environ if env is None else env
    if arguments[:3] != ["-m","arenalab_workbench.scene_assets","objects"]:
        raise ValueError("This worker accepts only the object-generation stage")
    def option(name):
        return Path(arguments[arguments.index(name)+1]).resolve(strict=True)
    geometry=Path(arguments[3]).resolve(strict=True)
    output=Path(arguments[4]).resolve()
    output.parent.resolve(strict=True)
    source=Path(env.get("ARENA_SCENE_SOURCE_ROOT",str(Path(__file__).resolve().parents[1]/"src"))).resolve(strict=True)
    cache=Path(env["ARENA_TORCH_CACHE_ROOT"]).resolve(strict=True)
    device=env.get("CUDA_VISIBLE_DEVICES")
    if device is None:
        raise ValueError("A GPU lease must set CUDA_VISIBLE_DEVICES")
    output_root=output.parent
    kernels=Path(env.get("ARENA_SCENE_CUDA_CACHE_ROOT",str(output_root/".model-kernels"))).resolve()
    kernels.mkdir(parents=True,exist_ok=True)
    readonly={option("--work"),option("--sam-source"),option("--sam-config").parent,
              option("--foundation-source"),source,cache}
    # The shared reconstruction job is the only writable bind. Prepared input,
    # model source/weights and the DINO cache stay read-only.
    mounts=[("rw",output_root)]
    if not kernels.is_relative_to(output_root):
        mounts.append(("rw",kernels))
    if not geometry.is_relative_to(output_root):
        raise ValueError("Geometry and object output must share the isolated job root")
    if any(p.is_relative_to(output_root) for p in readonly):
        raise ValueError("Keep prepared inputs, weights and code outside the writable job root")
    mounts += [("ro",p) for p in sorted(readonly)]
    argv=[env.get("ARENA_CONTAINER_ENGINE","docker"),"run","--rm","--network","none","--gpus","all","--shm-size","8g",
          "--label","arenalab.stage=scene-assets-objects"]
    if hasattr(os,"getuid"):
        argv += ["--user",f"{os.getuid()}:{os.getgid()}"]
    for mode,path in mounts:
        argv += ["--mount",f"type=bind,source={path},target={path}"+(",readonly" if mode=="ro" else "")]
    for key,value in dict(CUDA_VISIBLE_DEVICES=device,TORCH_HOME=str(cache),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
        HOME=str(kernels),TORCH_EXTENSIONS_DIR=str(kernels/"torch-extensions"),XDG_CACHE_HOME=str(kernels/"xdg"),
        LIDRA_SKIP_INIT="true",PYTHONPATH=str(source)+":/models/foundationpose:/models/foundation-build").items():
        argv += ["--env",key+"="+value]
    argv += [env.get("ARENA_SCENE_MODELS_IMAGE","arena-scene-assets-models:20261019"),"python",*arguments]
    return argv


if __name__=="__main__":
    raise SystemExit(subprocess.run(command(sys.argv[1:])).returncode)
