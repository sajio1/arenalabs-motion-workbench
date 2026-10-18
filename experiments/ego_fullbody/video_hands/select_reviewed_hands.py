"""Apply a separate human-reviewed data annotation mask, never robot policy."""
from pathlib import Path
import os,json,numpy as np
root=Path(os.environ['ARENA_EGO_WORKDIR']);out=root/'video-hands';d=dict(np.load(out/'raw-mano.npz'))
annotation=json.loads((out/'reviewed-detections.json').read_text(encoding='utf-8-sig'))
status=np.zeros(d['confidence'].shape,dtype=int)
for item in annotation['accepted']:
 f,side=item['frame'],item['side'];assert d['confidence'][f,side]>0
 status[f,side]=2
d['status']=status
np.savez_compressed(out/'reviewed-mano.npz',**d)
report=dict(accepted_per_hand=(status>0).sum(0).tolist(),accepted_total=int((status>0).sum()),total_hand_frames=int(status.size),review='Source-image ownership and visible hand checked; hidden fingers still inferred; manual sample review required',rejected_methods=['arbitrary guided-crop regression hallucinates cloth','zoom association still detects background hands / cloth'])
(out/'reviewed-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
