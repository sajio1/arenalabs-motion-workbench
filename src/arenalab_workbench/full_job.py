"""Requested per-recording outputs and their necessary dependencies only."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import subprocess
import sys
import time

from arenalab_workbench.settings import paths
from arenalab_workbench.scene_job import resolve_clip, run as scene_run, save, depth_current
from arenalab_workbench.body_prior import folder as body_folder
from arenalab_workbench.gpu_scheduler import run as gpu_run, resource_lock, merge_scene, write
from arenalab_workbench.progressive_job import Progress, initialize, merge_perception
from arenalab_workbench.job_outputs import selection_for, publish_scene, apply_scene_scale, native_cache_valid, render_key, prepare_contact_source, require_point_cloud, require_fov, publish_failure

ROOT = Path(os.environ.get('ARENA_WORKBENCH_ROOT', str(paths.workbench)))
STORE = paths.store


def run(scene_id, robot_id, hand_id, status):
    with resource_lock('source:' + scene_id):
        pass  # Drain legacy whole-source jobs before stage-level scheduling.
    with resource_lock('robot:' + scene_id + ':' + robot_id + ':' + hand_id):
        return _run(scene_id, robot_id, hand_id, Path(status))


def _run(scene_id, robot_id, hand_id, status):
    from arenalab_workbench.gripper_job import sources, run as gripper_run
    if scene_id in sources():
        return gripper_run(scene_id, robot_id, hand_id, status)
    started = time.time()
    public, row = resolve_clip(scene_id)
    expected = public.get('end_effector', 'none' if public['dataset'] == 'Nymeria' else 'wuji')
    if hand_id != expected:
        raise ValueError('Source requires ' + expected)
    selection, plan = selection_for(scene_id, robot_id)
    mode = plan['trajectory_scale']
    from arenalab_workbench.unified_scene import scene_key
    version = scene_key(scene_id, mode, robot_id, output_modalities=selection['effective_outputs'])
    public_dir = STORE / 'manipulation4d/unified' / version
    worker_version = scene_key(scene_id, mode, robot_id)
    destination = STORE / 'manipulation4d/unified' / worker_version / 'artifacts'
    scene_url = '/assets/' + (public_dir / 'scene.json').relative_to(STORE).as_posix()
    from arenalab_workbench.dataset_io.storage import mark_private
    mark_private(public_dir, public.get('owner'))
    mark_private(destination, public.get('owner'))
    initialize(destination / 'scene.json', public, int(row['frames']))
    status.parent.mkdir(parents=True, exist_ok=True)
    progress = Progress(status, scene_url, robot_id, selection, plan)
    publish_scene(destination / 'scene.json', public_dir / 'scene.json', selection, progress.ready)
    progress.write('running', 'Preparing selected outputs')
    # A cloud-only request does not even prepare a body workspace.
    work = body_folder(scene_id) if plan['body'] else status.parent
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', PYTHONPATH=paths.pythonpath(),
               ARENA_SOURCE_WORK=str(work), ARENA_RETARGET_WORK=str(work / 'retarget' / robot_id),
               ARENA_EXPORT_DIR=str(destination), ARENA_SCENE_WORK=str(ROOT / 'scene-jobs' / version))
    env.pop('ARENA_STREAM_ROBOT_ID', None)
    from arenalab_workbench.robot_registry import roster
    robot = roster()[robot_id]
    mobile = robot['kind'] == 'Wheeled'

    def publish(stage, phase):
        # Scene publication precedes its ready flag: consumers never load half an artifact.
        ready = dict(progress.ready, **{stage: True})
        publish_scene(destination / 'scene.json', public_dir / 'scene.json', selection, ready)
        progress.publish(stage, phase)

    def recover_body():
        requested = time.time()
        with resource_lock('body:' + str(work)):
            acquired = time.time()
            cache_hit = (work / 'prior-ready.json').exists()
            with (status.parent / 'body-worker.log').open('w') as log:
                runner = subprocess.run if cache_hit else gpu_run
                runner([sys.executable, str(Path(__file__).with_name('body_prior.py')), scene_id,
                        '--status', str(status.with_name('body-stage.json'))], env=env.copy(),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        save(status.with_name('body-timing.json'), dict(cache_hit=cache_hit,
             lock_wait_seconds=acquired-requested, seconds=time.time()-acquired))
        return work

    def recover_scene():
        with resource_lock('scene:' + scene_id):
            current = json.loads((destination / 'scene.json').read_text())
            source_hash = current.get('geometry', {}).get('source_sha256')
            source = row.get('source')
            if source_hash and source and source_hash != hashlib.sha256(Path(source).read_bytes()).hexdigest():
                raise ValueError('Source changed; stale scene cannot be reused')
            if not depth_current(current):
                # scene_run owns its source/cache identity; it does not infer a body.
                fresh_url = scene_run(scene_id, status.with_name('scene-stage.json'))
                merge_perception(destination / 'scene.json', json.loads((STORE / fresh_url.removeprefix('/assets/')).read_text()))
        scene = json.loads((destination / 'scene.json').read_text())
        require_point_cloud(scene)
        return scene

    def execute(script, args, phase, cpu=False, source_work=None):
        stage_started = time.time()
        save(status.with_name(status.stem + '-' + script + '.json'), dict(status='running', phase=phase, started_at=stage_started))
        with (status.parent / (script + '-' + robot_id + '.log')).open('w') as log:
            runner = subprocess.run if cpu else gpu_run
            stage_env=env.copy()
            if source_work is not None:stage_env['ARENA_SOURCE_WORK']=str(source_work)
            runner([sys.executable, str(Path(__file__).with_name(script)), *args], env=stage_env,
                   cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
        save(status.with_name(status.stem + '-' + script + '.json'), dict(status='completed', phase=phase,
             seconds=time.time()-stage_started))

    def visual_contact():
        contact_work = prepare_contact_source(public, row, ROOT / 'contact-jobs')
        manifest = json.loads((contact_work / 'input_manifest.json').read_text())
        hands_sha = hashlib.sha256((contact_work / 'source_hands.npz').read_bytes()).hexdigest()
        staging = destination / 'contact-stage'
        initialize(staging / 'scene.json', public, int(row['frames']))
        contact_env = env.copy()
        contact_env.update(ARENA_EXPORT_DIR=str(staging), ARENA_DEFER_PROXIMITY='1', ARENA_SOURCE_WORK=str(contact_work))
        shared = contact_work / 'video-contact-result.json'
        with resource_lock('contact:' + scene_id):
            for candidate in (destination / 'video-contact.json', staging / 'video-contact.json', shared):
                if candidate.exists():
                    data = json.loads(candidate.read_text())
                    if data.get('source_sha256') == manifest['source_sha256'] and data.get('source_hands_sha256') == hands_sha:
                        return data, contact_work
            with (status.parent / 'contact-worker.log').open('w') as log:
                gpu_run([sys.executable, str(Path(__file__).with_name('contact_recording.py'))], env=contact_env,
                        stdout=log, stderr=subprocess.STDOUT, check=True)
            data = json.loads((staging / 'video-contact.json').read_text())
            write(shared, data)
            return data, contact_work

    branches = ThreadPoolExecutor(max_workers=3)
    try:
        body_future = branches.submit(recover_body) if plan['body'] else None
        scene_future = branches.submit(recover_scene) if plan['scene'] else None
        contact_future = branches.submit(visual_contact) if plan['contact'] else None
        if body_future:
            body_future.result()
        if plan['retarget']:
            validation = destination / ('robot-' + robot_id) / 'validation.json'
            if validation.exists():
                from arenalab_workbench.motion_backend import select_native_work_env
                env = select_native_work_env(env, json.loads(validation.read_text()), work, robot_id)
            if mobile or not native_cache_valid(destination, robot_id, source_work=work):
                if mobile:
                    execute('wheeled_job.py', [robot_id, str(work), str(destination), hand_id], 'Retargeting native mobile body')
                else:
                    execute('retarget_robot.py', [robot_id], 'Refining native robot trajectory')
                merge_scene(destination / 'scene.json', lambda scene: scene.get('patch_videos', {}).pop(robot_id, None))
            if validation.exists():
                from arenalab_workbench.motion_backend import select_native_work_env
                env = select_native_work_env(env, json.loads(validation.read_text()), work, robot_id)
            publish('motion', 'Trajectory ready')
        if scene_future:
            scene = scene_future.result()
            env['ARENA_SCENE_WORK'] = str(ROOT / 'scene-jobs' / scene.get('geometry', {}).get('scene_work', version))
            if plan['source_body_for_scene_scale'] and not plan['retarget']:
                apply_scene_scale(work, robot, destination / 'scene.json')
            publish('scene', 'Point cloud ready')
        if plan['render']:
            from arenalab_workbench.depth_store import fov_current
            receipt = destination / ('robot-' + robot_id) / 'render-cache.json'
            key = render_key(destination, robot_id)
            scene = json.loads((destination / 'scene.json').read_text())
            hit = receipt.is_file() and json.loads(receipt.read_text()).get('key') == key and fov_current(scene, robot_id)
            if not hit:
                if mobile:
                    execute('mobile_render_job.py', [str(work), str(destination), robot_id], 'Rendering selected FOV')
                else:
                    execute('render_recording.py', [robot_id], 'Rendering selected FOV')
                save(receipt, dict(key=key))
            require_fov(json.loads((destination / 'scene.json').read_text()), robot_id)
            publish('fov', 'FOV ready')
        if contact_future:
            contact_data, contact_work = contact_future.result()
            write(destination / 'video-contact.json', contact_data)
            url = '/assets/' + (destination / 'video-contact.json').relative_to(STORE).as_posix()
            merge_scene(destination / 'scene.json', lambda scene: scene.update(tactile=dict(status='video_estimate', url=url,
                        reason='Independent per-digit video estimate; not measured force')))
            if plan['proximity']:
                execute('proximity_recording.py', [], 'Computing geometry proximity', cpu=True, source_work=contact_work)
            publish('contact', 'Contact ready')
        merge_scene(destination / 'scene.json', lambda scene: scene.update(generation_status=dict(status='completed',
                    robot_id=robot_id, hand_id=hand_id, trajectory_scale=mode, output_modalities=selection['effective_outputs'],
                    pipeline_version='resident-skeleton-surface-wrist-v3', seconds=time.time()-started)))
        publish_scene(destination / 'scene.json', public_dir / 'scene.json', selection, progress.ready)
        progress.complete('Selected outputs ready')
        return scene_url
    except Exception as error:
        publish_failure(public_dir / 'scene.json', selection, progress.ready, error)
        progress.write('failed', 'Generation stage failed: ' + str(error))
        raise
    finally:
        branches.shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('scene_id'); parser.add_argument('robot_id'); parser.add_argument('hand_id')
    parser.add_argument('--status', type=Path, required=True)
    args = parser.parse_args()
    run(args.scene_id, args.robot_id, args.hand_id, args.status)
