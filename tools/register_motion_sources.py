#!/usr/bin/env python3
"""Register explicitly staged, authorized motion-only examples; never downloads.

Registry update is an explicit CLI action. Data stays outside the Git checkout.
Body models stay at the decoding endpoint. No geometry/source bytes are served
by registration itself; the normal private source preparation builds 4D output.
"""
import argparse
import hashlib
import json
from pathlib import Path


def records(manifest_path, data_root):
    from arenalab_workbench.motion_sources import validate_body_attachment, _npz, _clock
    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(data_root).resolve(); result = {}
    for sample in manifest['samples']:
        folder = (root / sample['directory']).resolve()
        if not folder.is_relative_to(root):
            raise ValueError('Sample directory escapes explicit data root')
        recording = json.loads((folder / 'recording.json').read_text())
        supplied = validate_body_attachment(folder, recording)
        if supplied is None:
            raise ValueError('Expected a supplied body example')
        if sample['motion_sha256'] != supplied['sha256']:
            raise ValueError('Staged sample changed: ' + sample['id'])
        times, _, _ = _clock(_npz(supplied['path']), supplied)
        robot_source = supplied['format'] == 'robot_joint_state'
        stage = supplied.get('source_stage', recording.get('source_stage', sample.get('source_stage')))
        if stage is not None and (not isinstance(stage, str) or not stage.strip()):
            raise ValueError('source_stage must be a nonempty declared string')
        public = dict(id=sample['id'], title=sample['title'], dataset=sample['dataset'],
            end_effector='none', family=sample.get('family', 'Authorized motion examples'), duration=float(times[-1]-times[0]),
            uploaded=False, owner=None, video=None, has_supplied_body=True,
            source_family=supplied['family'], source_body_family=supplied['family'], source_evidence=supplied['evidence'],
            motion_source_policy='supplied-robot-fullbody-v1' if robot_source else 'supplied-body22-surface-v1')
        if stage is not None:public['source_stage'] = stage
        if robot_source:public['source_robot_id'] = supplied['robot_id']
        source_row = dict(id=sample['id'],source=supplied['path'],frames=len(times),fps=30.,video=None,
            motion_source=supplied,source_metadata=str(folder/'provenance.json'))
        result[sample['id']] = dict(public=public,source_row=source_row,motion_source=supplied,
            manifest=dict(recording,motion_source=supplied),body_adapter='supplied-robot' if robot_source else 'supplied-body')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--registry',type=Path,required=True,help='Explicit user-data/registry.json on the test deployment')
    parser.add_argument('--catalog',type=Path,help='Optional public catalog.json; source motion remains private storage')
    args=parser.parse_args()
    new=records(args.manifest,args.data_root)
    current=json.loads(args.registry.read_text()) if args.registry.exists() else {}
    for key,row in new.items():
        if key in current and current[key]!=row:
            raise ValueError('Refusing to replace a different registered recording: '+key)
    current.update(new)
    args.registry.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.registry.with_suffix('.json.tmp');temporary.write_text(json.dumps(current,indent=2)+'\n');temporary.replace(args.registry)
    if args.catalog:
        catalog=json.loads(args.catalog.read_text()) if args.catalog.exists() else []
        ids={row['id'] for row in catalog}
        catalog.extend(row['public'] for key,row in new.items() if key not in ids)
        args.catalog.parent.mkdir(parents=True,exist_ok=True)
        temporary=args.catalog.with_suffix('.json.tmp');temporary.write_text(json.dumps(catalog,indent=2)+'\n');temporary.replace(args.catalog)
    print(json.dumps({'registered':len(new),'ids':list(new)},indent=2))

if __name__=='__main__':main()
