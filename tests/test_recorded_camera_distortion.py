from pathlib import Path
import json
import sys
import numpy as np
from arenalab_workbench.recorded_camera_distortion import inverse_brown_render_map,recorded_render_map


def test_zero_distortion_is_identity():
    k=np.array([[50,0,32],[0,50,24],[0,0,1]])
    x,y=inverse_brown_render_map(k,[0]*5,64,48)
    yy,xx=np.mgrid[:48,:64]
    np.testing.assert_allclose(x,xx,atol=1e-5)
    np.testing.assert_allclose(y,yy,atol=1e-5)


def test_missing_recorded_calibration_does_not_invent_distortion(tmp_path):
    (tmp_path/'meta').mkdir()
    (tmp_path/'meta/info.json').write_text(json.dumps({'recording_metadata':{}}))
    assert recorded_render_map({'dataset':'Real robot','source':str(tmp_path)},np.eye(3),64,48) is None
