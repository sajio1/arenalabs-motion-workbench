"""SAM 3.1 foreground tracking; preserves source frame IDs and instance masks."""
from pathlib import Path
import json, os, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')
import cv2
import numpy as np
import torch
from sam3.model_builder import build_sam3_multiplex_video_predictor

ROOT = Path('/home/ubuntu/arenalab-experiments/manipulation4d')
FRAMES = Path(os.environ['ARENA_SEGMENT_FRAMES'])
OUT = Path(os.environ['ARENA_SEGMENT_OUTPUT'])
OUT.mkdir(exist_ok=True)
start = time.time()
predictor = build_sam3_multiplex_video_predictor(
    checkpoint_path=str(ROOT / 'sam31-weights/sam3.1_multiplex.pt'),
    bpe_path=str(ROOT / 'sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'),
    use_fa3=False, compile=False, warm_up=False,
)
# Upstream base predictor passes this default to multiplex init_state, whose
# released signature does not accept it. No state offloading is requested.
original_init_state = predictor.model.init_state
def init_state_compat(*, offload_state_to_cpu=False, **kwargs):
    if offload_state_to_cpu:
        raise ValueError('State offloading is unsupported by this multiplex release')
    return original_init_state(**kwargs)
predictor.model.init_state = init_state_compat

audit=json.loads((OUT.parent/'hardware-classification.json').read_text());result=None
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for item in audit['instances']:
  if not item.get('excluded'):continue
  saved=np.load(OUT/f"{item['key']}-masks.npz");mask=(saved['masks'][item['frame']]==item['instance']);result=np.zeros_like(saved['masks'],dtype='u2') if result is None else result;result[saved['masks']==item['instance']]=1
  session=predictor.handle_request({'type':'start_session','resource_path':str(FRAMES)})['session_id']
  y,x=np.where(mask);box=[float(x.min()/960),float(y.min()/540),float((x.max()-x.min()+1)/960),float((y.max()-y.min()+1)/540)]
  distance=cv2.distanceTransform(mask.astype('u1'),cv2.DIST_L2,5);py,px=np.unravel_index(distance.argmax(),distance.shape)
  predictor.handle_request({'type':'add_prompt','session_id':session,'frame_index':item['frame'],'points':[[float(px/960),float(py/540)]],'point_labels':[1],'obj_id':1})
  for response in predictor.handle_stream_request({'type':'propagate_in_video','session_id':session,'start_frame_index':item['frame'],'propagation_direction':'both','force_tracker_propagation':True}):
   masks=np.asarray(response['outputs']['out_binary_masks']);union=masks.reshape(-1,*masks.shape[-2:]).any(0).astype('u1') if masks.size else np.zeros((540,960),'u1');union=cv2.resize(union,(960,540),interpolation=cv2.INTER_NEAREST);result[response['frame_index']]|=union
  predictor.handle_request({'type':'close_session','session_id':session});print('TRACKED',item['key'],item['instance'],flush=True)
if result is not None:np.savez_compressed(OUT/'human-capture-hardware-masks.npz',masks=result,frame_indices=saved['frame_indices'],source_frame_indices=saved['source_frame_indices'])

(OUT.parent/'hardware-tracking.json').write_text(json.dumps(dict(policy='source-hardware-v4',bidirectional=True,instances=sum(bool(i.get('excluded')) for i in audit['instances']))))
