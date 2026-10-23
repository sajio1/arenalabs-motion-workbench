"""Pure output capability and dependency policy for the unified motion pipeline.

Callers provide registered source metadata, never just public preview media.
This module performs no I/O, inference, dispatch, caching or artifact mutation.
"""
from collections.abc import Mapping

POLICY = 'unified-output-modalities-v1'
OUTPUTS = ('trajectory', 'point_cloud', 'contact', 'inpaint_fov')


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def source_capabilities(public, row, gripper=False):
    """Describe supported outputs from trusted source registration metadata.

    ``row.video.path`` must describe a registered source camera input, not a
    third-person preview. A known missing calibration can explicitly disable it
    with ``camera_input_registered=False``. A Wuji label alone is insufficient:
    the input adapter must attest ``has_observed_hands=True`` or
    ``hand_landmark_count=21`` after verifying the source landmark stream.
    Robot topology/action/dataset names never determine this policy.
    """
    if not isinstance(public, Mapping) or not isinstance(row, Mapping):
        raise ValueError('Source capabilities require registered source metadata')
    metadata = row.get('metadata') or {}
    video = row.get('video') or {}
    if not isinstance(metadata, Mapping) or not isinstance(video, Mapping):
        raise ValueError('Invalid source camera metadata')
    camera = bool(_nonempty(video.get('path'))
                  and row.get('camera_input_registered') is not False
                  and metadata.get('preview_video_only') is not True)
    source = bool(_nonempty(row.get('source')) or _nonempty(row.get('observations_source'))
                  or isinstance(row.get('motion_source'), Mapping)
                  and _nonempty(row['motion_source'].get('path')))
    # A registered gripper worker reads its own source.npz inside work.
    source = source or bool(gripper and _nonempty(row.get('work')))
    hand = 'robotiq-2f-140' if gripper else public.get('end_effector', 'none')
    observed_hands = bool(hand == 'wuji' and source and (
        row.get('has_observed_hands') is True or row.get('hand_landmark_count') == 21))
    fov = bool(source and camera and (gripper or observed_hands))
    contact = bool(camera and observed_hands and not gripper)
    camera_reason = 'Registered source camera input is available' if camera else 'No registered source camera input; public preview video does not enable CV'
    if gripper:
        contact_reason = 'No gripper contact estimator is connected'
    elif not observed_hands:
        contact_reason = 'Contact requires an observed 21-landmark hand stream'
    else:
        contact_reason = camera_reason
    if not camera:
        fov_reason = camera_reason
    elif not source:
        fov_reason = 'Source motion observations are unavailable'
    elif not (gripper or observed_hands):
        fov_reason = 'FOV rendering supports observed Wuji hands or the registered gripper pipeline'
    else:
        fov_reason = 'Registered camera and supported end-effector renderer are available'
    outputs = {
        'trajectory': dict(available=source, reason='Registered motion source is available' if source else 'Source motion observations are unavailable'),
        'point_cloud': dict(available=camera, reason=camera_reason),
        'contact': dict(available=contact, reason=contact_reason),
        'inpaint_fov': dict(available=fov, reason=fov_reason),
    }
    return dict(policy=POLICY, available_outputs=[name for name in OUTPUTS if outputs[name]['available']],
                outputs=outputs, registered_camera=camera, observed_hand_landmarks=observed_hands,
                end_effector=hand, pipeline_kind='gripper' if gripper else 'body')


def _validate_selection(requested):
    if not isinstance(requested, (list, tuple)) or not requested:
        raise ValueError('OUTPUTS_INVALID: outputs must be a nonempty list')
    if any(not isinstance(name, str) or name not in OUTPUTS for name in requested):
        raise ValueError('OUTPUTS_INVALID: supported outputs are '+', '.join(OUTPUTS))
    if len(set(requested)) != len(requested):
        raise ValueError('OUTPUTS_INVALID: duplicate outputs are not allowed')
    # The same set has the same execution/cache identity, irrespective of UI order.
    return [name for name in OUTPUTS if name in requested]


def resolve_outputs(requested, capabilities):
    """Resolve omission to all applicable outputs; reject unavailable choices."""
    if not isinstance(capabilities, Mapping) or not isinstance(capabilities.get('outputs'), Mapping):
        raise ValueError('Output capabilities are required')
    available = [name for name in OUTPUTS if capabilities['outputs'].get(name, {}).get('available') is True]
    if requested is None:
        if not available:
            raise ValueError('OUTPUTS_UNAVAILABLE: this source has no supported outputs')
        effective = available
    else:
        effective = _validate_selection(requested)
        unavailable = [name for name in effective if name not in available]
        if unavailable:
            details = '; '.join(name+': '+str(capabilities['outputs'].get(name, {}).get('reason', 'Unavailable')) for name in unavailable)
            raise ValueError('OUTPUTS_UNAVAILABLE: '+details)
    return dict(requested_outputs=None if requested is None else list(requested),
                effective_outputs=effective, defaulted=requested is None, policy=POLICY)


def plan_outputs(effective, trajectory_scale='preserve', point_cloud_available=False):
    """Return required computation, separate from explicitly requested artifacts.

    FOV needs an internal robot trajectory and scene even when they were not
    selected for export. Contact alone needs source observations and video, not
    an inferred body or a robot rollout. Proximity is optional when geometry is
    already present or also requested. Source normalization for match_robot can
    require a body for cloud-only output; it must not trigger robot retargeting.
    """
    selected = _validate_selection(effective)
    if trajectory_scale not in {'preserve', 'match_robot'}:
        raise ValueError('Unknown trajectory scale mode')
    if not isinstance(point_cloud_available, bool):
        raise ValueError('point_cloud_available must be a boolean')
    render = 'inpaint_fov' in selected
    retarget = 'trajectory' in selected or render
    scene = 'point_cloud' in selected or render
    contact = 'contact' in selected
    normalize_for_scene = trajectory_scale == 'match_robot' and scene
    body = retarget or normalize_for_scene
    proximity = contact and (scene or point_cloud_available)
    reasons = {
        'body': (['trajectory or FOV requires a complete body source'] if retarget else [])
                + (['matched source/scene scale requires body morphology'] if normalize_for_scene else []),
        'retarget': ['trajectory or FOV requires target robot motion'] if retarget else [],
        'scene': ['point cloud or FOV requires registered-camera perception'] if scene else [],
        'contact': ['requested source-video hand contact estimate'] if contact else [],
        'render': ['requested FOV robot rendering'] if render else [],
        'proximity': ['optional geometry proximity uses available point cloud'] if proximity else [],
    }
    return dict(policy=POLICY, effective_outputs=selected, body=body, retarget=retarget,
                scene=scene, contact=contact, render=render, proximity=proximity,
                source_observations=True, source_body_for_scene_scale=normalize_for_scene,
                trajectory_scale=trajectory_scale, dependency_reasons=reasons)
