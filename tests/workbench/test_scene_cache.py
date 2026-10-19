"""Source content, policy and source clock must all invalidate scene reuse."""
from copy import deepcopy
from arenalab_workbench.scene_job import policy_signature


def test_scene_cache():
    row=dict(frames=450,fps=30,source_window=dict(start_frame=120,end_frame=570),video=dict(start_seconds=1.,frame_times=[1.,1.033,1.067]))
    key=policy_signature(row,'source-sha','video-sha')
    assert key==policy_signature(deepcopy(row),'source-sha','video-sha')
    assert key!=policy_signature(row,'changed-source','video-sha')
    assert key!=policy_signature(row,'source-sha','changed-video')
    assert key!=policy_signature(row,'source-sha','video-sha',policy='another-policy')
    for category,field,value in [('source_window','start_frame',121),('video','start_seconds',2.),('video','frame_times',[1.,1.034,1.067])]:
        changed=deepcopy(row);changed[category][field]=value
        assert key!=policy_signature(changed,'source-sha','video-sha')
    for field,value in [('frames',449),('fps',60)]:
        changed=deepcopy(row);changed[field]=value
        assert key!=policy_signature(changed,'source-sha','video-sha')
    print('PASS: deterministic source/policy cache; changed source, video, window, frame clock or rate cannot reuse it')
