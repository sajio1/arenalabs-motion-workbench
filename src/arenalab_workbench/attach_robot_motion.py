"""Prepare a new registry snapshot attaching complete states to an existing clip.

No active registry is replaced and no jobs or services are started. Original CV,
camera, hand and gripper observations stay registered separately.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .robot_motion_source import read_state, POLICY
from .motion_sources import sha256


def attach(record, spec, *, preview=None, preview_url=None):
    spec = dict(spec)
    _, _, times, _ = read_state(spec)
    if len(times) != record['source_row']['frames']:
        raise ValueError('Complete robot motion must cover the registered clip exactly')
    observation_path = record['source_row'].get('observations_source', record['source_row']['source'])
    with np.load(observation_path, allow_pickle=False) as observed:
        if 'timestamps_s' not in observed or observed['timestamps_s'].shape != times.shape or not np.allclose(observed['timestamps_s'], times, atol=1e-6, rtol=0):
            raise ValueError('Complete robot motion and registered observations must share the same clock')
    spec.update(family='robot', sha256=sha256(spec['path']))
    result = copy.deepcopy(record)
    result.update(motion_source=spec, body_adapter='supplied-robot')
    public=result['public']
    # A dataset's character/video preview is not evidence of the robot motion.
    # Preserve it as auxiliary metadata, never present it as the source robot.
    if public.get('video'):public['auxiliary_video']=public['video']
    public.update(source_policy=POLICY,motion_source_policy=POLICY,
                  source_robot_id=spec['robot_id'],source_body_family='robot',
                  has_supplied_body=True,video=None)
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
        public['video']=dict(url=preview_url,start_seconds=0,kind='robot_motion',robot_id=spec['robot_id'])
    result['manifest'] = dict(result.get('manifest', {}), motion_source=spec,
                              previous_body_adapter=record.get('body_adapter'),
                              source_completion=False)
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
