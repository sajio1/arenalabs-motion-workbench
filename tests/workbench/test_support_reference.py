import numpy as np
from arenalab_workbench.support_reference import support_reference

def test_swing_is_not_locked_and_source_is_immutable():
    neutral=np.zeros((20,3));neutral[:,0]=np.r_[np.full(10,-.1),np.full(10,.1)]
    weights=np.zeros((20,55));weights[:10,7]=1;weights[10:,8]=1
    vertices=np.repeat(neutral[None],120,0);vertices[:,:,2]=-.1
    vertices[40:80,10:,2]+=.2
    human=dict(vertices=vertices,betas=np.zeros(1))
    model=dict(v_template=neutral,shapedirs=np.zeros((20,3,1)),weights=weights)
    before=vertices.copy();floor,confidence,clearance,report=support_reference(human,model)
    assert abs(floor+.1)<1e-10
    assert confidence[60,1]<.01 and confidence[60,0]>.99
    assert clearance[60,1]>.15 and clearance[60,0]==0
    assert np.array_equal(before,vertices) and report['source_trajectory_edited'] is False

if __name__=='__main__':test_swing_is_not_locked_and_source_is_immutable();print('support reference regression passed')
