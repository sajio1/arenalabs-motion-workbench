"""Prepare a new registry snapshot attaching complete states to an existing clip.

No active registry is replaced and no jobs or services are started. Original CV,
camera, hand and gripper observations stay registered separately.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .robot_motion_source import read_state, POLICY
from .motion_sources import sha256


def _gripper_revision(descriptor, spec):
    """Keep camera/tool observations, but never inherit a reconstructed body cache."""
    from .source_identity import robot_source_identity, identity_digest
    from .gpu_scheduler import resource_lock
    work = Path(descriptor['work']).resolve()
    original = work / 'source.json'
    metadata = json.loads(original.read_text())
    source_hash = sha256(work / 'source.npz')
    identity = dict(policy='attached-gripper-complete-source-v1', source_sha256=source_hash,
                    metadata_sha256=sha256(original), original_work=str(work),
                    complete_source=robot_source_identity(spec, dict(frames=metadata['frames'], fps=metadata.get('fps', 30))))
    destination = work.with_name(work.name + '-robot-source-' + identity_digest(identity)[:24])
    updated = copy.deepcopy(metadata)
    updated.update(motion_source=spec, original_source_work=str(work), original_source_sha256=source_hash)
    # Existing video bytes remain authoritative; relative references must not
    # become relative to the new body/cache workspace.
    if updated.get('head_video'):
        updated['head_video'] = str((work / updated['head_video']).resolve())
    for video in updated.get('wrist_videos', []):
        video['path'] = str((work / video['path']).resolve())
    updated.pop('render_scene_work', None)
    with resource_lock('gripper-source-revision:' + str(destination)):
        if destination.exists():
            if (json.loads((destination / 'source.json').read_text()) != updated
                    or sha256(destination / 'source.npz') != source_hash
                    or json.loads((destination / 'source-revision.json').read_text()) != identity):
                raise ValueError('Immutable attached gripper source changed')
        else:
            with tempfile.TemporaryDirectory(dir=work.parent, prefix=destination.name+'.prepare-') as temporary:
                temp = Path(temporary)
                shutil.copyfile(work / 'source.npz', temp / 'source.npz')
                (temp / 'source.json').write_text(json.dumps(updated, indent=2, allow_nan=False)+'\n')
                (temp / 'source-revision.json').write_text(json.dumps(identity, indent=2, allow_nan=False)+'\n')
                if sha256(temp / 'source.npz') != source_hash or sha256(original) != identity['metadata_sha256']:
                    raise ValueError('Original gripper observations changed during attachment')
                temp.rename(destination)
    return dict(descriptor, work=str(destination))


def attach(record, spec, *, preview=None, preview_url=None):
    from .source_identity import absolute_spec
    spec = absolute_spec(spec, Path.cwd())
    _, _, times, _ = read_state(spec)
    if len(times) != record['source_row']['frames']:
        raise ValueError('Complete robot motion must cover the registered clip exactly')
    observation_path = (str(Path(record['gripper']['work']) / 'source.npz') if record.get('gripper')
                        else record['source_row'].get('observations_source', record['source_row']['source']))
    with np.load(observation_path, allow_pickle=False) as observed:
        if 'timestamps_s' not in observed or observed['timestamps_s'].shape != times.shape or not np.allclose(observed['timestamps_s'], times, atol=1e-6, rtol=0):
            raise ValueError('Complete robot motion and registered observations must share the same clock')
    spec.update(family='robot', sha256=sha256(spec['path']))
    result = copy.deepcopy(record)
    result.update(motion_source=spec, body_adapter='supplied-robot')
    result['source_row']['motion_source'] = spec
    public=result['public']
    # A dataset's character/video preview is not evidence of the robot motion.
    # Preserve it as auxiliary metadata, never present it as the source robot.
    camera = record['source_row'].get('video') or {}
    recorded_video = bool(camera.get('path') and Path(camera['path']).is_file()
                          and camera.get('role') != 'preview-only'
                          and not record.get('manifest', {}).get('preview_video_only'))
    if record.get('gripper'):
        original_work = Path(record['gripper']['work'])
        metadata = json.loads((original_work/'source.json').read_text())
        recorded_video = bool(metadata.get('head_video') and (original_work/metadata['head_video']).is_file()
                              and metadata.get('video_role') != 'preview-only')
    if public.get('video') and not recorded_video:public['auxiliary_video']=public['video']
    public.update(source_policy=POLICY,motion_source_policy=POLICY,
                  source_robot_id=spec['robot_id'],source_body_family='robot',
                  has_supplied_body=True)
    if not recorded_video:public['video']=None
    spec['video_role'] = 'recorded' if recorded_video else spec.get('video_role', 'unspecified')
    public.pop('scene_url', None)
    if preview is not None or preview_url is not None:
        if not preview or not preview_url or not preview_url.startswith('/assets/'):
            raise ValueError('Provide a verified source robot preview and local asset URL together')
        if (preview.get('kind')!='robot_motion' or preview.get('robot_id')!=spec['robot_id']
                or preview.get('source_sha256')!=spec['sha256'] or preview.get('frames')!=len(times)
                or preview.get('retargeted') is not False or preview.get('source_trajectory_modified') is not False
                or not np.array_equal(preview.get('source_timestamps_s'),times)):
            raise ValueError('Preview must replay this exact original source robot motion')
        if sha256(preview['path'])!=preview.get('video_sha256'):
            raise ValueError('Source preview video hash changed')
        video = dict(url=preview_url,start_seconds=0,kind='robot_motion',robot_id=spec['robot_id'])
        if recorded_video:public['source_robot_preview_video']=video
        else:public['video']=video
    result['manifest'] = dict(result.get('manifest', {}), motion_source=spec,
                              previous_body_adapter=record.get('body_adapter'),
                              source_completion=False)
    if record.get('gripper'):
        result['gripper'] = _gripper_revision(record['gripper'], spec)
        result['source_row']['source'] = str(Path(result['gripper']['work']) / 'source.npz')
        result['source_row'].pop('observations_source', None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--motion-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview-receipt',type=Path)
    parser.add_argument('--preview-url')
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    spec = json.loads(args.motion_source.read_text())
    for key in ('path', 'objects_path'):
        if spec.get(key):spec[key] = str((args.motion_source.parent / spec[key]).resolve())
    preview=json.loads(args.preview_receipt.read_text()) if args.preview_receipt else None
    registry[args.scene] = attach(registry[args.scene], spec,preview=preview,preview_url=args.preview_url)
    # Exclusive creation deliberately refuses to replace a live/historical file.
    with args.output.open('x') as stream:
        json.dump(registry, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
