"""Selected outputs for registered robot/gripper observations and supplied bodies."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import time

from arenalab_workbench.settings import paths
from arenalab_workbench.gripper_source import ROOT, save, digest
from arenalab_workbench.gpu_scheduler import run as gpu_run, resource_lock, merge_scene
from arenalab_workbench.progressive_job import Progress, initialize, merge_perception
from arenalab_workbench.scene_job import depth_current
from arenalab_workbench.depth_store import fov_current
from arenalab_workbench.gripper_registry import REGISTRY, sources
from arenalab_workbench.job_outputs import selection_for, publish_scene, apply_scene_scale, native_cache_valid, render_key, require_point_cloud, require_fov, publish_failure

POLICY = 'unified-gpu-hybrid-gripper-housing-v1'


def cached(scene_id, rid, scene, require_visual=True, destination=None):
    row = sources().get(scene_id)
    if not row:
        return False
    work = Path(row['work'])
    from arenalab_workbench.gripper_source import REAL_POLICY
    meta = json.loads((work / 'source.json').read_text())
    if meta['dataset'] == 'Real robot' and meta.get('adapter_policy') != REAL_POLICY:
        return False
    dest = Path(destination or row['destination'])
    from arenalab_workbench.gripper_body import body_path
    if not native_cache_valid(dest, rid, source_work=body_path(work), source_sha256=digest(work / 'source.npz')):
        return False
    report = json.loads((dest / ('robot-' + rid) / 'validation.json').read_text())
    if report.get('policy') != POLICY or report.get('head_calibration') != meta.get('head_calibration'):
        return False
    return not require_visual or (fov_current(scene, rid) and scene.get('geometry', {}).get('visual_policy') == 'gripper-visual-v4' and depth_current(scene))


def run(scene_id, rid, hand, status):
    if hand != 'robotiq-2f-140':
        raise ValueError('Gripper source requires a mapped gripper end effector')
    status = Path(status)
    selection, plan = selection_for(scene_id, rid)
    row = sources()[scene_id]
    work = Path(row['work'])
    mode = plan['trajectory_scale']
    from arenalab_workbench.unified_scene import scene_key
    public_dir = paths.store / 'manipulation4d/unified' / scene_key(scene_id, mode, rid, output_modalities=selection['effective_outputs'])
    dest = paths.store / 'manipulation4d/unified' / scene_key(scene_id, mode, rid) / 'artifacts'
    start = time.time()
    env = os.environ.copy()
    env.update(PYTHONPATH=paths.pythonpath(), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
    env.pop('ARENA_STREAM_ROBOT_ID', None)
    code = Path(__file__).parent
    from arenalab_workbench.robot_registry import roster
    robot = roster()[rid]
    mobile = robot['kind'] == 'Wheeled'
    from arenalab_workbench.gripper_body import body_path
    from arenalab_workbench.height_registry import public_entry, variant
    public = public_entry(scene_id)
    if public is None:
        public = variant(scene_id)['public']
    meta = json.loads((work / 'source.json').read_text())
    from arenalab_workbench.dataset_io.storage import mark_private
    mark_private(public_dir, public.get('owner'))
    mark_private(dest, public.get('owner'))
    initialize(dest / 'scene.json', public, meta['frames'])
    status.parent.mkdir(parents=True, exist_ok=True)
    url = '/assets/' + (public_dir / 'scene.json').relative_to(paths.store).as_posix()
    progress = Progress(status, url, rid, selection, plan)
    publish_scene(dest / 'scene.json', public_dir / 'scene.json', selection, progress.ready)
    progress.write('running', 'Preparing selected outputs')

    def publish(stage, phase):
        publish_scene(dest / 'scene.json', public_dir / 'scene.json', selection, dict(progress.ready, **{stage: True}))
        progress.publish(stage, phase)

    def recover_body():
        body = body_path(work)
        with resource_lock('body:' + str(body)):
            runner = subprocess.run if (body / 'prior-ready.json').exists() else gpu_run
            with (status.parent / (rid + '-body.log')).open('w') as log:
                runner([str(paths.scene_python), str(code / 'gripper_body.py'), str(work)], env=env.copy(),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        return body

    def recover_scene():
        with resource_lock('scene:' + scene_id):
            scene = json.loads((dest / 'scene.json').read_text())
            if scene.get('geometry', {}).get('source_sha256') != digest(work / 'source.npz') or scene.get('geometry', {}).get('visual_policy') != 'gripper-visual-v4' or not depth_current(scene):
                staging = dest / 'perception-stage'
                execute('gripper_scene.py', [str(work), str(staging)], paths.scene_python, 'Source-camera point cloud', cpu=True)
                merge_perception(dest / 'scene.json', json.loads((staging / 'scene.json').read_text()))
            require_point_cloud(json.loads((dest / 'scene.json').read_text()))

    def execute(script, args, python, phase, cpu=False):
        stage_started = time.time()
        save(status.with_name(status.stem + '-' + script + '.json'), dict(status='running', phase=phase, started_at=stage_started))
        with (status.parent / (rid + '-' + script + '.log')).open('w') as log:
            runner = subprocess.run if cpu else gpu_run
            runner([str(python), str(code / script), *args], env=env.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
        save(status.with_name(status.stem + '-' + script + '.json'), dict(status='completed', phase=phase, seconds=time.time()-stage_started))

    try:
        with ThreadPoolExecutor(max_workers=2) as branches:
            body_future = branches.submit(recover_body) if plan['body'] else None
            perception = branches.submit(recover_scene) if plan['scene'] else None
            body = body_future.result() if body_future else None
            if plan['retarget']:
                scene = json.loads((dest / 'scene.json').read_text())
                if mobile or not cached(scene_id, rid, scene, require_visual=False, destination=dest):
                    if mobile:
                        execute('wheeled_job.py', [rid, str(body), str(dest), hand, '--gripper-work', str(work)], paths.scene_python, 'Retargeting native mobile body and gripper')
                    else:
                        execute('gripper_retarget.py', [str(work), rid], paths.motion_python, 'Refining native gripper trajectory')
                        execute('gripper_export.py', [str(work), rid, str(dest)], paths.motion_python, 'Exporting native robot and gripper', cpu=True)
                    merge_scene(dest / 'scene.json', lambda scene: scene.get('patch_videos', {}).pop(rid, None))
                publish('motion', 'Trajectory ready')
            if perception:
                perception.result()
                if plan['source_body_for_scene_scale'] and not plan['retarget']:
                    apply_scene_scale(body, robot, dest / 'scene.json', gripper=True)
                publish('scene', 'Point cloud ready')
            if plan['render']:
                scene = json.loads((dest / 'scene.json').read_text())
                receipt = dest / ('robot-' + rid) / 'render-cache.json'
                key = render_key(dest, rid)
                hit = receipt.is_file() and json.loads(receipt.read_text()).get('key') == key and fov_current(scene, rid)
                if not hit:
                    execute('gripper_render.py', [str(work), rid, str(dest)], paths.scene_python, 'Rendering selected FOV')
                    save(receipt, dict(key=key))
                require_fov(json.loads((dest / 'scene.json').read_text()), rid)
                publish('fov', 'FOV ready')
        merge_scene(dest / 'scene.json', lambda scene: scene.update(generation_status=dict(status='completed', policy=POLICY,
                    robot_id=rid, hand_id=hand,
                    output_modalities=selection['effective_outputs'], trajectory_scale=mode, seconds=time.time()-start)))
        publish_scene(dest / 'scene.json', public_dir / 'scene.json', selection, progress.ready)
        progress.complete('Selected outputs ready')
        return url
    except Exception as error:
        publish_failure(public_dir / 'scene.json', selection, progress.ready, error)
        progress.write('failed', 'Generation stage failed: ' + str(error))
        raise
