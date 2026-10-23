"""Explicit, offline replacement of a supplied recording's source contract.

This prepares a new registry snapshot; it never changes the live registry.
Original observations and generated artifacts remain immutable. Changing a
complete source also changes its clock, object binding and cache identity.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path

def replace_supplied_source(record, package, manifest, *, preview=None):
    from .motion_sources import validate_body_attachment, _npz, _clock

    previous = record.get('motion_source') or record.get('source_row', {}).get('motion_source')
    if not previous or record.get('gripper'):
        raise ValueError('Source replacement requires an existing complete supplied source')
    # Observed camera/tool tracks require an explicit, separate synchronization
    # migration. A different source clock must never inherit them silently.
    row = record['source_row']
    if row.get('video') or row.get('observations_source'):
        raise ValueError('Recording has independent observations; migrate their clocks explicitly')
    spec = validate_body_attachment(Path(package), manifest)
    if spec is None:
        raise ValueError('Replacement must declare a complete supplied source')
    times, _, _ = _clock(_npz(spec['path']), spec)
    if len(times) < 2:
        raise ValueError('A replacement motion needs at least two timestamps')
    stage = spec.get('source_stage', manifest.get('source_stage'))
    if stage is not None and (not isinstance(stage, str) or not stage.strip()):
        raise ValueError('source_stage must be a nonempty declared string')
    result = deepcopy(record)
    result['motion_source'] = spec
    result['body_adapter'] = 'supplied-robot' if spec['family'] == 'robot' else 'supplied-body'
    result['source_row'] = dict(id=row['id'], source=spec['path'], frames=len(times),
                                fps=30., video=None, motion_source=spec,
                                source_metadata=str(Path(package).resolve() / 'provenance.json'))
    result['manifest'] = dict(deepcopy(manifest), motion_source=spec)
    result['source_revision'] = dict(policy='explicit-supplied-source-replacement-v1',
        previous_source=deepcopy(previous), previous_frames=row['frames'],
        selected_source_sha256=spec['sha256'],
        source_stage=stage,
        reason=manifest.get('source_selection_reason', 'Explicit source selection'))
    public = result['public']
    # Saved generated scenes describe their historical input, not this source.
    # Preserve the artifacts on disk, but do not advertise them as the current
    # recording's default preview or silently reuse their generation result.
    for key in ('scene_url', 'source_motion', 'source_body', 'source_robot_preview_video',
                'source_robot_id', 'generated_human_model'):
        public.pop(key, None)
    if public.get('video'):
        public['auxiliary_video'] = public['video']
    public.update(video=None, has_supplied_body=True, source_family=spec['family'],
                  source_format=spec['format'],
                  source_body_family=spec['family'], source_evidence=spec['evidence'],
                  source_stage=stage,
                  duration=float(times[-1] - times[0]), frames=len(times), fps=30.)
    if 'end_effector' in manifest:
        public['end_effector'] = manifest['end_effector']
    if spec['family'] == 'robot':
        from .robot_motion_source import POLICY
        public.update(source_robot_id=spec['robot_id'], generated_human_model=None,
                      source_policy=POLICY, motion_source_policy=POLICY)
        if preview is not None:
            if (preview.get('source_sha256') != spec['sha256']
                    or preview.get('source_robot_id') != spec['robot_id']
                    or preview.get('frames') != len(times)
                    or preview.get('fps') != 30.
                    or preview.get('world_up') != 'z'
                    or preview.get('retargeted') is not False
                    or preview.get('source_trajectory_modified') is not False):
                raise ValueError('Preview does not represent this exact native source')
            public['source_motion'] = deepcopy(preview)
    else:
        public.update(source_policy='supplied-body-direct-v1',
                      motion_source_policy='supplied-body-direct-v1')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    plan = json.loads(args.plan.read_text())
    seen = set()
    for item in plan['replacements']:
        key = item['recording_id']
        if key in seen:
            raise ValueError('Duplicate replacement: ' + key)
        seen.add(key)
        record = registry[key]
        previous = record.get('motion_source') or record['source_row'].get('motion_source', {})
        if previous.get('sha256') != item['expected_source_sha256']:
            raise ValueError('Registered source changed while preparing migration: ' + key)
        package = Path(item['package']).resolve()
        manifest = json.loads((package / 'recording.json').read_text())
        preview = json.loads(Path(item['preview_receipt']).read_text()) if item.get('preview_receipt') else None
        registry[key] = replace_supplied_source(record, package, manifest, preview=preview)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(registry, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(replaced=len(seen), output=str(args.output))))


if __name__ == '__main__':
    main()
