"""Separate research entry point. No changes to routes, UI, DA3 or live caches."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="stage",required=True)
    workflow=commands.add_parser("run")
    workflow.add_argument("work",type=Path)
    workflow.add_argument("output",type=Path)
    workflow.add_argument("--model",type=Path,required=True)
    workflow.add_argument("--converter",type=Path,required=True)
    workflow.add_argument("--parameters",type=Path,nargs="+",required=True,help="Immutable SMPL-X and observed source trajectories")
    workflow.add_argument("--window",type=int,default=32)
    workflow.add_argument("--overlap",type=int,default=8)
    workflow.add_argument("--stride",type=int,default=1)
    for field in ["sam-source","sam-config","foundation-source"]:
        workflow.add_argument("--"+field,type=Path,required=True)
    workflow.add_argument("--object-python",type=Path,help="Explicit isolated model-worker interpreter/wrapper")
    geometry=commands.add_parser("reconstruct")
    geometry.add_argument("work",type=Path,help="Existing prepared scene job with SAM masks")
    geometry.add_argument("output",type=Path)
    geometry.add_argument("--model",type=Path,required=True,help="Local MapAnything weights; never downloads")
    geometry.add_argument("--window",type=int,default=32)
    geometry.add_argument("--overlap",type=int,default=8)
    geometry.add_argument("--stride",type=int,default=1,help="Diagnostic subsampling, recorded in output frame ids")
    splats=commands.add_parser("splats")
    splats.add_argument("geometry",type=Path)
    splats.add_argument("output",type=Path)
    splats.add_argument("--converter",type=Path,required=True,help="Official SPZ ply_to_spz executable")
    splats.add_argument("--iterations",type=int,default=7000)
    objects=commands.add_parser("objects")
    objects.add_argument("geometry",type=Path)
    objects.add_argument("output",type=Path)
    objects.add_argument("--work",type=Path,required=True,help="Original-resolution prepared RGB/SAM observations")
    for field in ["sam-source","sam-config","foundation-source"]:
        objects.add_argument("--"+field,type=Path,required=True)
    objects.add_argument("--limit-tracks",type=int,help="Diagnostic only; default processes every track")
    tracking=commands.add_parser("objects-track",help="Retry tracking without rerunning object generation")
    for field in ["geometry","work","priors","output"]:
        tracking.add_argument(field,type=Path)
    tracking.add_argument("--foundation-source",type=Path,required=True)
    mesh_post=commands.add_parser("objects-postprocess",help="Official mesh cleanup preserving original tracked poses")
    mesh_post.add_argument("priors",type=Path);mesh_post.add_argument("output",type=Path)
    pack=commands.add_parser("pack")
    for field in ["work","geometry","splats","objects","output"]:
        pack.add_argument(field,type=Path)
    pack.add_argument("--parameters",type=Path,nargs="*",default=[])
    delivery=commands.add_parser("delivery-request")
    for field in ["assets","urdf","cameras","output"]:
        delivery.add_argument(field,type=Path)
    delivery.add_argument("--days",type=int,default=7)
    delivery.add_argument("--points",type=int,choices=[1024,4096,8192],default=4096)
    render=commands.add_parser("render")
    for field in ["assets","cameras","output"]:
        render.add_argument(field,type=Path)
    render.add_argument("--decoder",type=Path,required=True,help="Official SPZ spz_to_ply executable")
    for field in ["robot-model","playback","robot-to-world","urdf","robot-binding"]:
        render.add_argument("--"+field,type=Path)
    render.add_argument("--days",type=int,default=7)
    render.add_argument("--points",type=int,choices=[1024,4096,8192],default=4096)
    render.add_argument("--geometry-fps",type=int,choices=[15,30],default=15,help="RGB/actions keep their original clock; geometry carries explicit source frame ids")
    expiry=commands.add_parser("expire")
    expiry.add_argument("root",type=Path,help="Dedicated temporary delivery root")
    expiry.add_argument("--delete",action="store_true",help="Default is listing only")
    args=parser.parse_args()
    if args.stage=="run":
        from .workflow import run
        run(args.work,args.output,args.model,args.converter,args.parameters,args.window,args.overlap,args.stride,
            args.sam_source,args.sam_config,args.foundation_source,args.object_python)
    elif args.stage=="reconstruct":
        from .reconstruct import run
        run(args.work,args.output,args.model,args.window,args.overlap,args.stride)
    elif args.stage=="splats":
        from .splats import run
        run(args.geometry,args.output,args.converter,args.iterations)
    elif args.stage=="objects":
        from .objects import run
        run(args.geometry,args.output,args.sam_source,args.sam_config,args.foundation_source,args.work,args.limit_tracks)
    elif args.stage=="pack":
        from .package import pack
        pack(args.work,args.geometry,args.splats,args.objects,args.output,args.parameters)
    elif args.stage=="objects-track":
        from .objects import retrack
        retrack(args.geometry,args.work,args.priors,args.output,args.foundation_source)
    elif args.stage=="objects-postprocess":
        from .objects import postprocess_existing
        postprocess_existing(args.priors,args.output)
    elif args.stage=="delivery-request":
        from .package import delivery_request
        print(json.dumps(delivery_request(args.assets,args.urdf,args.cameras,args.output,args.days,args.points),indent=2))
    elif args.stage=="render":
        from .delivery import render
        render(args.assets,args.cameras,args.output,args.decoder,args.robot_model,args.playback,
               args.robot_to_world,args.urdf,args.days,args.points,args.robot_binding,args.geometry_fps)
    elif args.stage=="expire":
        from .retention import expire
        print(json.dumps(expire(args.root,delete=args.delete),indent=2))


if __name__=="__main__":
    main()
