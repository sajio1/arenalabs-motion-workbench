from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments/manipulation4d'))
from tracking_recovery import propagate_resilient


class Predictor:
    def __init__(self, visible, fail=None):
        self.visible=visible;self.fail=fail;self.propagations=[]
    def handle_request(self, request):
        if request['type']=='reset_session':return {}
        return {'outputs':{'visible':request['frame_index'] in self.visible}}
    def handle_stream_request(self, request):
        start=request['start_frame_index'];self.propagations.append(start)
        for frame in range(start,6):
            if frame==self.fail:raise RuntimeError('No points are provided; please add points first')
            yield {'frame_index':frame,'outputs':{'visible':frame in self.visible}}


@pytest.mark.parametrize('prompt_resets_state', [False, True])
def test_late_detection_and_empty_tracker_preserve_all_frames(prompt_resets_state):
    predictor=Predictor({1,2,4,5},fail=3);frames={}
    def record(frame,data,offset):
        frames[frame]=offset+1 if data['visible'] else 0
        return data['visible'],frames[frame]
    recoveries=propagate_resilient(predictor,'s','object',6,record,prompt_resets_state=prompt_resets_state)
    assert sorted(frames)==list(range(6))
    assert frames[0]==frames[3]==0
    assert frames[1]==frames[2]==1 and frames[4]==frames[5]==2
    assert predictor.propagations==[1,4] and recoveries[0]['resume_frame']==3


@pytest.mark.parametrize('prompt_resets_state', [False, True])
def test_absence_is_observed_per_frame_without_fake_seeds(prompt_resets_state):
    predictor=Predictor(set());frames=[]
    def record(frame,data,offset):
        frames.append(frame);return False,0
    assert propagate_resilient(predictor,'s','object',6,record,prompt_resets_state=prompt_resets_state)==[]
    assert frames==list(range(6)) and predictor.propagations==[]
