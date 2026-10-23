"""Scene identity includes every choice that changes its shared world geometry."""
import hashlib


def scene_key(scene_id, scale_mode, robot_id):
    if scale_mode not in {'preserve', 'match_robot'}:
        raise ValueError('Unknown trajectory scale mode')
    identity = str(scene_id) + ':' + scale_mode + ':resident-v3'
    if scale_mode == 'match_robot':
        if not robot_id:
            raise ValueError('Scaled scene requires a target robot')
        identity += ':' + str(robot_id)
    return hashlib.sha256(identity.encode()).hexdigest()
