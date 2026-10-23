"""Scene identity includes every choice that changes its shared world geometry."""
import hashlib
import json


def scene_key(scene_id, scale_mode, robot_id, output_modalities=None):
    if scale_mode not in {'preserve', 'match_robot'}:
        raise ValueError('Unknown trajectory scale mode')
    identity = str(scene_id) + ':' + scale_mode + ':resident-v3'
    # A newly attached complete robot recording must never publish into its
    # historical sparse-human scene, including other targets in that scene.
    from .dataset_io.storage import entry
    recording = entry(scene_id) or {}
    source = recording.get('motion_source') or recording.get('source_row', {}).get('motion_source')
    if source and source.get('format') == 'robot_joint_state':
        identity += ':complete-robot-v1:' + json.dumps(source, sort_keys=True)
    if scale_mode == 'match_robot':
        if not robot_id:
            raise ValueError('Scaled scene requires a target robot')
        identity += ':' + str(robot_id)
    if output_modalities is not None:
        if not output_modalities or len(set(output_modalities))!=len(output_modalities):
            raise ValueError('Select one or more unique output modalities')
        identity += ':outputs:' + ','.join(sorted(output_modalities))
    return hashlib.sha256(identity.encode()).hexdigest()
