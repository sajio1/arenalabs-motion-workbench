"""Source-only rigs cannot silently become target robots or accept upload paths."""
import json
import pytest
from arenalab_workbench import source_robot_registry as registry


@pytest.fixture
def local(tmp_path,monkeypatch):
    from arenalab_workbench import robot_registry
    config=tmp_path/'source-robots.json';xml=tmp_path/'source.xml';xml.write_text('<mujoco/>')
    monkeypatch.setenv('ARENA_SOURCE_ROBOT_REGISTRY',str(config))
    targets={'g1':dict(id='g1',pipeline_row=dict(xml='/target/g1.xml',body='g1'))}
    monkeypatch.setattr(robot_registry,'roster',lambda:targets.copy())
    config.write_text(json.dumps(dict(robots=[dict(id='hoi_unitree_h2',pipeline_row=dict(xml='source.xml',body='h2'))])))
    return config,xml,targets


def test_source_only_fixed_admin_model_does_not_enter_target_roster(local):
    from arenalab_workbench.robot_registry import roster
    config,xml,targets=local
    source=registry.resolve_source_robot('hoi_unitree_h2')
    assert source['xml']==str(xml) and source['body']=='h2'
    assert set(roster())=={'g1'}
    assert registry.resolve_source_robot('g1')==targets['g1']['pipeline_row']
    with pytest.raises(ValueError,match='unavailable'):registry.resolve_source_robot('/customer/arbitrary.xml')


def test_source_model_validation_reuses_metadata_but_tracks_asset_bytes(local,monkeypatch):
    from arenalab_motion import robot_metadata
    _,xml,_=local;calls=[]
    def validate(row):calls.append(row);return dict(row,semantic_metadata={'fixed':True})
    monkeypatch.setattr(robot_metadata,'ensure_robot_metadata',validate);registry._validated.cache_clear()
    first=registry.validated_source_robot('hoi_unitree_h2');first['semantic_metadata']['fixed']=False
    assert registry.validated_source_robot('hoi_unitree_h2')['semantic_metadata']['fixed'] is True
    assert len(calls)==1
    xml.write_text('<mujoco><!--new revision--></mujoco>')
    registry.validated_source_robot('hoi_unitree_h2')
    assert len(calls)==2


@pytest.mark.parametrize('case',['duplicate','conflicting_model','missing_xml'])
def test_bad_admin_registry_fails_explicitly(local,case):
    config,_,_=local;value=json.loads(config.read_text());row=value['robots'][0]
    if case=='duplicate':value['robots'].append(row.copy())
    elif case=='conflicting_model':row['pipeline_row']['model']='different.xml'
    else:row['pipeline_row'].pop('xml')
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError):registry.resolve_source_robot('hoi_unitree_h2')
