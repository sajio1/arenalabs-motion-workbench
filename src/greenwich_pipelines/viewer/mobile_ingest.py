"""Reuse verified mobile descriptors only for identical physical URDF packages.

Names alone never establish compatibility. The fingerprint includes the entire
link/joint description and the bytes of every referenced mesh/texture. Solver
policy is shared with the registered mobile model; no per-upload tuning.
"""
from pathlib import Path
import copy
import hashlib
import json
import xml.etree.ElementTree as ET

from .export import array_json


def physical_fingerprint(urdf):
    urdf = Path(urdf).resolve()
    root = ET.parse(urdf).getroot()

    def resource(raw):
        if raw.startswith('package://'):
            package, tail = raw[len('package://'):].split('/', 1)
            parent = next((p for p in urdf.parents if p.name == package), None)
            if parent is None:
                raise ValueError('Cannot resolve registered package resource')
            path = parent / tail
        else:
            path = urdf.parent / raw
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def node(element):
        attrs = dict(element.attrib)
        if element is root:
            attrs.pop('name', None)  # Labels do not affect physical identity.
        if element.tag in {'mesh', 'texture'} and 'filename' in attrs:
            attrs['filename'] = 'sha256:' + resource(attrs['filename'])
        # unpack_robot injects compiler flags. They do not identify the original
        # robot. We load the trusted registered MJCF, never uploaded extensions.
        children = [node(child) for child in element if child.tag != 'mujoco']
        return [element.tag, sorted(attrs.items()), (element.text or '').strip(), children]

    return hashlib.sha256(json.dumps(node(root), separators=(',', ':')).encode()).hexdigest()


def ingest_registered_mobile(urdf, folder, body, report, registered):
    """Return a mobile pipeline row or None for an unrecognized physical model."""
    candidates = [r for r in registered if r.get('kind') == 'Wheeled'
                  and r.get('pipeline_row', {}).get('mobile_semantics')
                  and not r['id'].startswith('custom_')]
    if not candidates:
        return None
    fingerprint = physical_fingerprint(urdf)
    for robot in candidates:
        native = robot['pipeline_row']
        metadata_path = Path(native['metadata']).resolve()
        metadata = json.loads(metadata_path.read_text())
        source = (metadata_path.parent / metadata['source']).resolve()
        if physical_fingerprint(source) != fingerprint:
            continue
        folder = Path(folder).resolve()
        # Resolve all native MJCF resources before persisting the descriptor in
        # the uploaded package. Do not alter physical axes, limits or frames.
        tree = ET.parse(native['xml'])
        native_parent = Path(native['xml']).resolve().parent
        compiler = tree.getroot().find('compiler')
        for element in tree.getroot().findall('./asset/*'):
            if 'file' not in element.attrib:
                continue
            directory = ''
            if compiler is not None:
                key = 'texturedir' if element.tag == 'texture' else 'meshdir'
                directory = compiler.get(key, compiler.get('assetdir', ''))
            element.set('file', str((native_parent / directory / element.get('file')).resolve()))
        if compiler is not None:
            for key in ('meshdir', 'texturedir', 'assetdir'):
                compiler.attrib.pop(key, None)
        model = folder / 'model.xml'
        tree.write(model, encoding='utf-8', xml_declaration=True)
        metadata.update(model=str(model), source=str(Path(urdf).resolve()))
        target_metadata = folder / 'mobile-metadata.json'
        array_json(target_metadata, metadata)
        row = copy.deepcopy(native)
        row.update(name=body, body=body, xml=str(model), metadata=str(target_metadata))
        # Validate the required native frames and free planar-root representation.
        import mujoco as mj
        model_check = mj.MjModel.from_xml_path(str(model))
        if model_check.jnt_type[0] != mj.mjtJoint.mjJNT_FREE:
            raise ValueError('Registered mobile descriptor requires a free root')
        frames = [*metadata['wrists'], metadata['torso'], *row['mobile_semantics']['shoulders'],
                  *row['mobile_semantics']['elbows'], row['mobile_semantics']['head']]
        for name in frames:
            model_check.body(name)
        array_json(folder / 'descriptor.json', dict(schema='arenalab.mobile.v1', **row,
                   physical_fingerprint=fingerprint, matched_robot_id=robot['id']))
        report.update(status='ready', conversion_ready=True, matched_robot_id=robot['id'],
                      selection_key=robot['id'], physical_fingerprint=fingerprint,
                      mobile_semantics=row['mobile_semantics'],
                      frame_provenance='Exact URDF topology and asset-content match to verified native metadata',
                      message='Mobile robot ready. Verified native body and end-effector frames; uses the shared wheeled generation pipeline.')
        return row
    return None
