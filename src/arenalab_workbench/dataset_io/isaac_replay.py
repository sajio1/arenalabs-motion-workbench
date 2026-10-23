"""Isaac Sim 5.1 standalone player/validator; uses only this downloaded package.

Start with Isaac's python.sh, not an arbitrary system Python interpreter.
"""
import argparse
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=Path, default=Path(__file__).with_name('robot.usda'))
parser.add_argument('--headless', action='store_true')
parser.add_argument('--validate', action='store_true')
parser.add_argument('--output', type=Path, default=Path('isaac-validation'))
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
from isaacsim import SimulationApp
app = SimulationApp({'headless': args.headless, 'active_gpu': args.gpu, 'physics_gpu': args.gpu,
                     'multi_gpu': False, 'width': 960, 'height': 720, 'renderer': 'RaytracedLighting'})
try:
    import numpy as np
    import omni.usd
    import omni.timeline
    import omni.replicator.core as rep
    from pxr import Usd, UsdGeom, UsdLux, Gf
    stage_path = args.stage.resolve()
    if not stage_path.is_file(): raise FileNotFoundError(stage_path)
    if not omni.usd.get_context().open_stage(str(stage_path)): raise RuntimeError('Isaac could not open USD')
    for _ in range(10): app.update()
    stage = omni.usd.get_context().get_stage()
    contract = json.loads(stage_path.with_name('replay.json').read_text(encoding='utf-8'))
    n = int(contract['frames'])
    if stage.GetTimeCodesPerSecond() != 30 or stage.GetEndTimeCode() != n-1:
        raise RuntimeError('USD clock differs from replay contract')
    if stage.GetUsedLayers() != [stage.GetRootLayer()]:
        # Session layer is anonymous and acceptable; external referenced assets are not.
        external = [layer.identifier for layer in stage.GetUsedLayers()
                    if not layer.anonymous and layer.identifier != stage.GetRootLayer().identifier]
        if external: raise RuntimeError('Replay depends on external USD layers: ' + str(external))
    light = UsdLux.DomeLight.Define(stage, '/World/ReplayLight')
    light.CreateIntensityAttr(350)
    light.CreateColorAttr(Gf.Vec3f(.32,.38,.45))
    bounds = UsdGeom.BBoxCache(Usd.TimeCode(0), ['default']).ComputeWorldBound(stage.GetPrimAtPath('/World/Robot')).ComputeAlignedRange()
    low, high = np.array(bounds.GetMin()), np.array(bounds.GetMax())
    center = (low+high)/2
    size = max(high-low)
    camera = UsdGeom.Camera.Define(stage, '/World/ReplayCamera')
    eye = center + np.array([1.5, 2.1, .7])*size
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*center), Gf.Vec3d(0,0,1))
    camera.AddTransformOp().Set(view.GetInverse())
    camera.CreateFocalLengthAttr(40.)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 1000))
    from omni.kit.viewport.utility import get_active_viewport
    viewport = get_active_viewport()
    if viewport is not None: viewport.set_active_camera(str(camera.GetPath()))
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_start_time(0)
    timeline.set_end_time(n/30)
    timeline.set_target_framerate(30)
    timeline.pause()
    if args.validate:
        args.output.mkdir(parents=True, exist_ok=True)
        scene_path = stage_path.parent.parent/'scene.json'
        scene = json.loads(scene_path.read_text(encoding='utf-8'))
        dataset = stage_path.parents[3]
        source = json.loads((dataset/scene['robots'][0]['playback_url']).read_text(encoding='utf-8'))
        positions = np.asarray(source['positions'])
        quaternions = np.asarray(source['quaternions'])
        if positions.shape != (n, contract['meshes'], 3) or quaternions.shape != (n, contract['meshes'], 4) or not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
            raise RuntimeError('Original playback has invalid or non-finite mesh poses')
        basis = np.asarray(contract['source_to_isaac_rotation'])
        max_position = max_rotation = 0.
        start = time.monotonic()
        # All mesh poses, every frame. Compare Isaac USD transforms against the
        # independent original playback, not against another exported pose copy.
        for frame in range(n):
            cache = UsdGeom.XformCache(Usd.TimeCode(frame))
            for index in range(contract['meshes']):
                prim = stage.GetPrimAtPath('/World/Robot/geom_%d' % index)
                if not prim: raise RuntimeError('Missing mesh prim')
                matrix = cache.GetLocalToWorldTransform(prim)
                actual = np.asarray(matrix).T
                if not np.isfinite(actual).all():
                    raise RuntimeError('Isaac returned a non-finite mesh transform')
                max_position = max(max_position, float(np.max(np.abs(actual[:3,3] - basis@positions[frame,index]))))
                x,y,z,w = quaternions[frame,index]
                rotation = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
                max_rotation = max(max_rotation, float(np.max(np.abs(actual[:3,:3] - basis@rotation))))
        if max_position > 1e-6 or max_rotation > 1e-6:
            raise RuntimeError('Isaac replay differs from source: '+str((max_position,max_rotation)))
        product = rep.create.render_product(str(camera.GetPath()), (960,720))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb')
        rgb.attach([product])
        from PIL import Image
        frames = [0,n//2,n-1,n//4,0,n-1]  # Forward and backward seek, including repeated endpoints.
        captures=[]
        for sample, frame in enumerate(frames):
            timeline.set_current_time(frame/30)
            for _ in range(4): app.update()
            rep.orchestrator.step(rt_subframes=4, delta_time=0., pause_timeline=True)
            image = np.asarray(rgb.get_data())
            if image.shape != (720,960,4) or float(image[...,:3].std(axis=(0,1)).max()) < 1:
                raise RuntimeError('Empty Isaac rendered frame')
            target = args.output/('frame-%02d-%04d.png' % (sample,frame))
            Image.fromarray(image).save(target)
            captures.append(dict(frame=frame,path=target.name,rgb_std=float(image[...,:3].std())))
        report = dict(engine='Isaac Sim 5.1.0', mode=contract['mode'], frames=n, meshes=contract['meshes'],
                      checked_mesh_poses=n*contract['meshes'], max_position_error_m=max_position,
                      max_rotation_matrix_error=max_rotation, elapsed_s=time.monotonic()-start,
                      renders=captures, physics_articulation=False, passed=True)
        (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('ISAAC_REPLAY_PASS '+json.dumps(report),flush=True)
    else:
        timeline.play()
        while app.is_running():
            app.update()
            if timeline.get_current_time() >= (n-1)/30:
                timeline.set_current_time(0)
except BaseException:
    import traceback
    import omni.kit.app
    traceback.print_exc()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
    # Kit fast shutdown exits from app.close(); preserve a failing process code.
    omni.kit.app.get_app().post_quit(1)
    raise
finally:
    app.close()
