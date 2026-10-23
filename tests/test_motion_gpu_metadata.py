from copy import deepcopy
import json

import pytest
pytest.importorskip('mujoco')
pytest.importorskip('torch')

from arenalab_motion.robot_metadata import ensure_robot_metadata, validate_robot_metadata, RobotMetadataError


def rig(tmp_path, *, labels=True, extra=False):
    limbs=[]
    for side, sign in [('left',1),('right',-1)]:
        for end, first, middle, z in [('wrist','shoulder','elbow',.4), ('ankle','hip','knee',-.2)]:
            names=[side+'_'+first,side+'_'+middle,side+'_'+end]
            joint=lambda i: (names[i]+'_pitch') if labels else ('axis_'+side+'_'+str(z)+'_'+str(i))
            inserted='<body name="extra_'+side+'_'+end+'" pos="0 0 -.1"><joint name="extra_axis_'+side+'_'+end+'"/><geom size=".01"/>' if extra else ''
            limbs.append(f'<body name="{names[0]}" pos="0 {sign*.2} {z}"><joint name="{joint(0)}"/><geom size=".01"/>'
                         f'<body name="{names[1]}" pos="0 0 -.3"><joint name="{joint(1)}"/><geom size=".01"/>'
                         f'{inserted}<body name="{names[2]}" pos="0 0 -.3"><joint name="{joint(2)}"/><geom size=".01"/></body>'
                         + ('</body>' if extra else '')+'</body></body>')
    path=tmp_path/'robot.xml'
    path.write_text('<mujoco><worldbody><body name="base" pos="0 0 1"><freejoint/><geom size=".1"/>'+''.join(limbs)+'</body></worldbody></mujoco>')
    return dict(name='arbitrary_customer_identifier',xml=str(path),body='base',endpoints={role:role for role in ('left_wrist','right_wrist','left_ankle','right_ankle')})


def test_semantics_resolve_physical_labels_and_preserve_explicit_metadata(tmp_path):
    row=rig(tmp_path)
    resolved=ensure_robot_metadata(row)
    assert 'semantic_metadata' not in row
    assert resolved['semantic_metadata']['semantics']['left_elbow']=='left_elbow'
    assert resolved['semantic_metadata']['semantics']['chest']=='base'
    assert ensure_robot_metadata(resolved)==resolved
    assert validate_robot_metadata(resolved)['root']==1


def test_anonymous_geometry_only_accepts_unambiguous_three_centers(tmp_path):
    row=rig(tmp_path,labels=False)
    resolved=ensure_robot_metadata(row)
    assert resolved['semantic_metadata']['semantics']['right_knee']=='right_knee'
    assert all('three' in x for x in resolved['semantic_metadata']['resolution_evidence'].values())
    ambiguous=rig(tmp_path,labels=False,extra=True)
    with pytest.raises(RobotMetadataError,match='ambiguous'):
        ensure_robot_metadata(ambiguous)


def test_invalid_cross_limb_metadata_rejected(tmp_path):
    resolved=ensure_robot_metadata(rig(tmp_path))
    broken=deepcopy(resolved)
    broken['semantic_metadata']['semantics']['left_elbow']='right_elbow'
    with pytest.raises(RobotMetadataError,match='ancestry'):
        ensure_robot_metadata(broken)
    broken=deepcopy(resolved)
    broken['endpoints']['left_wrist']='right_wrist'
    with pytest.raises(RobotMetadataError,match='distinct'):
        ensure_robot_metadata(broken)


def test_minimal_row_resolves_missing_endpoints_from_physical_tree(tmp_path):
    row=rig(tmp_path)
    del row['endpoints']
    resolved=ensure_robot_metadata(row)
    assert resolved['endpoints']=={role:role for role in ('left_wrist','right_wrist','left_ankle','right_ankle')}
    assert resolved['semantic_metadata']['semantics']['left_elbow']=='left_elbow'
    assert 'endpoints' not in row
