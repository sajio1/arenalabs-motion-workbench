"""Fill short segmentation gaps with bidirectional optical-flow consistency."""
from pathlib import Path
import json,sys,numpy as np,cv2
work=Path(sys.argv[1]);audit=json.loads((work/'hardware-classification.json').read_text());images=[cv2.imread(str(p),0) for p in sorted((work/'frames').glob('*.jpg'))];h,w=images[0].shape;yy,xx=np.mgrid[:h,:w].astype('f4');dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM);saved=np.load(work/'masks/human-capture-hardware-masks.npz');result=saved['masks'].copy();rep=[]
for item in audit['instances']:
 if not item.get('excluded'):continue
 data=np.load(work/'masks'/f"{item['key']}-masks.npz");m=(data['masks']==item['instance']).astype('u1');valid=np.where(m.sum((1,2))>0)[0]
 for t in np.where(m.sum((1,2))==0)[0]:
  source=int(valid[np.argmin(abs(valid-t))])
  if abs(source-t)>7:continue
  current=m[source].copy();step=1 if t>source else -1
  for j in range(source+step,t+step,step):
   previous=j-step;flow=dis.calc(images[j],images[previous],None);reverse=dis.calc(images[previous],images[j],None);mx=xx+flow[:,:,0];my=yy+flow[:,:,1];back=cv2.remap(reverse,mx,my,cv2.INTER_LINEAR);consistent=np.linalg.norm(flow+back,axis=2)<2;current=(cv2.remap(current,mx,my,cv2.INTER_NEAREST)>0)&consistent;current=current.astype('u1')
  result[t]|=current;rep.append(dict(key=item['key'],instance=item['instance'],frame=int(t),from_frame=source,pixels=int(current.sum())))
np.savez_compressed(work/'masks/human-capture-hardware-masks.npz',masks=result,frame_indices=saved['frame_indices'],source_frame_indices=saved['source_frame_indices']);(work/'hardware-flow-audit.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep))
