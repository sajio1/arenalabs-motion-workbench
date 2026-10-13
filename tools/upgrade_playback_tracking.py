"""Add measured wrist telemetry to existing playback; no inference/refinement."""
import argparse
import json
import hashlib
from pathlib import Path
import mujoco
import numpy as np
from greenwich_pipelines.viewer.export import array_json
from greenwich_pipelines.viewer.tracking import tracking_payload
from greenwich_pipelines.viewer.server import conversion_signature, PAGES


def telemetry(row, path, pipeline):
    with np.load(path, allow_pickle=False) as archive:
        saved = {k: archive[k] for k in archive.files}
    return tracking_payload(row, saved, pipeline, mujoco.MjModel.from_xml_path(row['xml']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/viewer.local.json'))
    parser.add_argument('--before-signatures', type=Path, required=True)
    parser.add_argument('--historical-nymeria', type=Path)
    parser.add_argument('--historical-gento', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    storage = Path(config['storage'])
    registry_path = storage / 'viewer-conversions.json'
    registry = json.loads(registry_path.read_text())
    before = json.loads(args.before_signatures.read_text())
    robots = {r['id']: r for r in config['robots']}
    robots.update({r['id']: r for r in json.loads((storage / 'uploaded/registry.json').read_text())})
    clips = {c['id']: c for c in config['clips']}
    report = {'updated': [], 'skipped': []}
    for key, old_signature in before.items():
        entry = registry.get(key)
        if not entry or entry['signature'] != old_signature:
            report['skipped'].append({'key': key, 'reason': 'Cache changed since measurement snapshot'})
            continue
        clip, robot = clips[entry['clip_id']], robots[entry['robot_id']]
        path = storage / entry['url'].removeprefix('/assets/')
        try:
            roster = json.loads((path.parent / 'robots.json').read_text())[0]
            if roster != robot['pipeline_row']:
                raise ValueError('Physical metadata differs from saved conversion')
            results = json.loads((path.parent / 'output/result.json').read_text())
            motions = sorted((a for a in results['artifacts'] if a['kind'] == 'motion'), key=lambda a: a.get('clip', ''))
            pieces = [telemetry(roster, path.parent / 'output' / a['path'], PAGES[clip['page']]) for a in motions]
            payload = json.loads(path.read_text())
            if pieces and pieces[0]:
                tracking = pieces[0]
                for piece in pieces[1:]:
                    for field in ('actual_positions', 'target_positions', 'position_error_mm', 'orientation_error_deg'):
                        tracking[field].extend(piece[field])
                if len(tracking['position_error_mm']) != payload['frames']:
                    raise ValueError('Telemetry frame count differs from playback')
                payload['tracking'] = tracking
                array_json(path, payload)
            # Explicit export-only upgrade: pre-change verified cache and identical
            # saved physical roster, source/stat/model remain in the new signature.
            entry['signature'] = conversion_signature(clip, robot)
            report['updated'].append(key)
        except Exception as error:
            report['skipped'].append({'key': key, 'reason': str(error)})
    array_json(registry_path, registry)
    if args.historical_nymeria:
        for clip in clips.values():
            if clip['page'] != 'localmanipulation' or clip.get('source_window'):
                continue
            for rid, url in clip.get('cache', {}).items():
                if rid.startswith('custom_') or '/jobs/' in url:
                    continue
                try:
                    row = robots[rid]['pipeline_row']
                    result = args.historical_nymeria / clip['source_name'] / row['body'] / 'result.npz'
                    payload_path = storage / url.removeprefix('/assets/')
                    payload = json.loads(payload_path.read_text())
                    payload['tracking'] = telemetry(row, result, 'nymeria')
                    array_json(payload_path, payload)
                    report['updated'].append(f"{clip['id']}:{rid}")
                except Exception as error:
                    report['skipped'].append({'key': f"{clip['id']}:{rid}", 'reason': str(error)})
    if args.historical_gento:
        for clip in clips.values():
            if clip['page'] != 'manipulation' or not clip.get('cache'):
                continue
            for rid, url in clip['cache'].items():
                try:
                    # Match immutable prior SHA and exact target arrays, not labels.
                    roots = list(args.historical_gento.glob(f'*/{rid}/*/result.npz'))
                    measured = None
                    for result in roots:
                        with np.load(result) as d:
                            saved = {k: d[k] for k in d.files}
                        payload_path = storage / url.removeprefix('/assets/')
                        payload = json.loads(payload_path.read_text())
                        if len(saved['qpos']) != payload['frames']:
                            continue
                        basis = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])
                        if not np.allclose(payload['skeleton'], saved['body_positions'] @ basis.T, atol=1e-6, rtol=0):
                            continue  # Same input can have several different refined trajectories.
                        native_targets = np.asarray(payload['targets']) @ basis
                        if not np.allclose(native_targets, saved['targets'], atol=1e-6, rtol=0):
                            continue
                        original_report = json.loads((result.parent / 'report.json').read_text())
                        for prior in args.historical_gento.glob(f'*/{rid}/*/prime_original.npz'):
                            if hashlib.sha256(prior.read_bytes()).hexdigest() != original_report['prior_sha256']:
                                continue
                            with np.load(prior) as d:
                                raw = {k: d[k] for k in d.files}
                            if not np.array_equal(saved['targets'], raw['position'][:len(saved['qpos'])]):
                                raise ValueError('Historical prior target mismatch')
                            saved.update(wrist_ids=raw['wrist_ids'], semantic_local=raw['semantic_local'],
                                         ee_local_position=raw.get('ee_local_position', np.zeros((2, 3))))
                            row = robots[rid]['pipeline_row']
                            candidate = tracking_payload(row, saved, 'gento', mujoco.MjModel.from_xml_path(row['xml']))
                            if not (np.allclose(np.percentile(candidate['position_error_mm'], 95, axis=0), original_report['hand_error_mm_p95'], atol=1e-3, rtol=0)
                                    and np.allclose(np.percentile(candidate['orientation_error_deg'], 95, axis=0), original_report['orientation_deg_p95'], atol=1e-3, rtol=0)):
                                raise ValueError('Historical measurements do not reproduce recorded metrics')
                            candidate['reference'] = 'Historical wrist task targets vs final native FK; original immutable mounting frames'
                            measured = candidate
                            break
                        if measured:
                            break
                    if measured is None:
                        raise ValueError('No verified historical task contract found')
                    payload['tracking'] = measured
                    array_json(payload_path, payload)
                    report['updated'].append(f"{clip['id']}:{rid}")
                except Exception as error:
                    report['skipped'].append({'key': f"{clip['id']}:{rid}", 'reason': str(error)})
    array_json(storage / 'tracking-upgrade-report.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
