"""Source semantics shared by API preflight and the generation queue."""

def input_requirements(scene_id):
    from arenalab_workbench.height_registry import default_id, variant, public_entry
    from arenalab_workbench.gripper_job import sources
    resolved = default_id(scene_id) or scene_id
    entry = public_entry(resolved)
    if entry is None: raise ValueError('Recording is not connected')
    hand = ('robotiq-2f-140' if resolved in sources() else
            entry.get('end_effector', 'none' if entry['dataset'] == 'Nymeria' else 'wuji'))
    return {'recording_id': resolved, 'hand_id': hand,
            'requires_initial_height': entry['dataset'] == 'UMI' and not variant(resolved)}
