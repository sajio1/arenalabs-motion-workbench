"""Presentation metadata only: preserve sources, clip IDs, caches and robot history."""
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def package_identity(folder):
    """Match physical URDF and mesh bytes, independent of ZIP roots/absolute paths."""
    candidates = sorted(folder.rglob('*.urdf'))
    source = next((p for p in candidates if p.name != 'resolved.urdf'), None)
    if source is None:
        return None
    tree = ET.parse(source).getroot()
    for node in list(tree):
        if node.tag == 'mujoco':
            tree.remove(node)  # Importer path settings, not physical topology.
    meshes = {}
    for path in folder.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.stl', '.obj', '.dae'}:
            meshes.setdefault(path.name.lower(), []).append(path)
    for mesh in tree.iter('mesh'):
        name = Path(mesh.attrib['filename'].replace('\\', '/')).name.lower()
        variants = {hashlib.sha256(p.read_bytes()).hexdigest() for p in meshes.get(name, [])}
        if len(variants) != 1:
            return None  # Missing/ambiguous evidence must not collapse distinct models.
        mesh.set('filename', next(iter(variants)))
    canonical = ET.canonicalize(ET.tostring(tree, encoding='unicode'), strip_text=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


SCENARIOS = {
    'S19-Fresh_air': ('Home & outdoor activities', 'Fresh air & home routines'),
    'S7-Cooking': ('Cooking & food preparation', 'Cooking session excerpts'),
    'S12-Game_night': ('Games & play', 'Game night session excerpts'),
    'S2-Where_is_X': ('Object handling & storage', 'Finding & retrieving objects'),
    'S5-Workout': ('Exercise', 'Workout session excerpts'),
    'S11-Laundary': ('Clothing & textiles', 'Laundry session excerpts'),
    'S10-Housekeeping': ('Cleaning & housekeeping', 'Housekeeping session excerpts'),
}


def curated_category(action):
    if re.search(r'laundry|clothes|jacket|washer|blanket|bedding|pillow', action):
        return 'Clothing & textiles'
    if 'game' in action:
        return 'Games & play'
    if re.search(r'clean|wipe|wash|dispose', action):
        return 'Cleaning & housekeeping'
    return 'Object handling & storage'


def organize(config_path, annotations, curated):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    storage = Path(config['storage'])
    registry = storage / 'uploaded/registry.json'
    robots = json.loads(registry.read_text()) if registry.exists() else []
    for robot in robots:
        key = package_identity(registry.parent / robot['id'])
        if key:
            robot.setdefault('report', {})['selection_key'] = key
    if registry.exists():
        registry.write_text(json.dumps(robots, indent=2), encoding='utf-8')

    historical = {'ny_unseen_pan': ('Cooking & food preparation', 'Cookware handling'),
                  'ny_unseen_prep': ('Cooking & food preparation', 'Food preparation'),
                  'ny_unseen_switch': ('Posture & transitions', 'Task transitions'),
                  'ny_crouch_measure': ('Object handling & storage', 'Crouching & reaching')}
    syncs = [(p, json.loads(p.read_text())) for p in curated.glob('*/source_ego.sync.json')]
    for clip in config['clips']:
        if clip['page'] != 'localmanipulation':
            continue
        metadata = clip.setdefault('metadata', {})
        group = historical.get(clip['id'])
        origin = 'Historical excerpt label'
        sequence = metadata.get('Sequence')
        if sequence:
            path = annotations / (sequence + '_metadata_json.json')
            if path.exists():
                script = json.loads(path.read_text())['script']
                group = SCENARIOS.get(script, ('Other activities', script.split('-', 1)[-1].replace('_', ' ')))
                metadata['Recording scenario'] = script
                origin = 'Dataset recording scenario; not a per-frame action annotation'
            if clip.get('video'):
                video_path = Path(clip['video']['path']).resolve()
                match = next((p for p, sync in syncs if sync.get('source_id') == sequence and
                              sync.get('synchronization_verified') and (p.parent / 'source_ego.mp4').resolve() == video_path), None)
                if match:
                    action = match.parent.name.removeprefix('ego_nymeria_').replace('_', ' ')
                    group = (curated_category(action), action.capitalize())
                    origin = 'Existing curated action label with verified source-video mapping'
        if not group:
            group = ('Unclassified activities', 'Unlabelled session excerpts')
            origin = 'No activity annotation available'
        metadata.update({'Activity category': group[0], 'Activity subclass': group[1], 'Category origin': origin})
        clip['family'] = group[0]
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding='utf-8')
    groups = Counter(r.get('report', {}).get('selection_key', r['id']) for r in robots)
    report = {'uploaded_records': len(robots), 'unique_uploaded_models': len(groups),
              'nymeria_categories': dict(Counter(c['family'] for c in config['clips'] if c['page'] == 'localmanipulation'))}
    (storage / 'library-organization-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/viewer.local.json'))
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--curated', type=Path, required=True)
    args = parser.parse_args()
    organize(args.config, args.annotations, args.curated)
