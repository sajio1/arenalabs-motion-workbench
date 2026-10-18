"""SAM 3.1 foreground tracking; preserves source frame IDs and instance masks."""
from pathlib import Path
import json, os, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')
import cv2
import numpy as np
import torch
from sam3.model_builder import build_sam3_multiplex_video_predictor
from tracking_recovery import propagate_resilient

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
report = json.loads((OUT/'segmentation.json').read_text()) if (OUT/'segmentation.json').exists() else {'model': 'SAM 3.1 Object Multiplex', 'source_fps': 30, 'mask_fps': 15, 'prompts': {}}
prompts=json.loads(os.environ['ARENA_SEGMENT_PROMPTS'])
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    for item in prompts:
        key, prompt = item[:2]
        discovery_frame = int(item[2]) if len(item)>2 else 0
        # Text VG discovers newly visible instances throughout propagation. Seed
        # at its start: a future seed prepopulates tracker IDs before frame zero
        # and can desynchronize multiplex masks from IDs.
        seed_frame = 0
        category_start=time.monotonic()
        response = predictor.handle_request({'type': 'start_session', 'resource_path': str(FRAMES)})
        session = response['session_id']
        frame_masks, audit = {}, {}
        def record(index, data, offset):
            masks=np.asarray(data['out_binary_masks']);ids=np.asarray(data['out_obj_ids'])
            labelmap=np.zeros((540,960),dtype=np.uint16)
            for object_id,mask in zip(ids,masks):
                mask=np.asarray(mask).squeeze().astype(np.uint8)
                if mask.shape!=labelmap.shape:mask=cv2.resize(mask,(960,540),interpolation=cv2.INTER_NEAREST)
                label=offset+int(object_id)+1
                if label>65535:raise ValueError('Too many tracker identities for uint16 masks')
                labelmap[mask>0]=label
            frame_masks[index]=labelmap
            audit[index]={'frame':index,'instances':(ids+offset).tolist(),'pixels':int((labelmap>0).sum())}
            return bool(np.any(labelmap)),int(labelmap.max())
        recoveries=propagate_resilient(predictor,session,prompt,len(list(FRAMES.glob('*.jpg'))),record)
        indices = sorted(frame_masks)
        expected=len(list(FRAMES.glob('*.jpg')))
        if indices!=list(range(expected)):raise ValueError('SAM output does not cover all prepared frames')
        np.savez_compressed(OUT / f'{key}-masks.npz', masks=np.stack([frame_masks[i] for i in indices]), frame_indices=indices, source_frame_indices=np.asarray(indices) * 2)
        report['prompts'][key] = {'text': prompt, 'seed_frame':next((i for i in indices if audit[i]['pixels']>0),None), 'no_instances_detected':not any(audit[i]['pixels'] for i in indices), 'discovery_frame':discovery_frame, 'frames': len(indices), 'seconds':time.monotonic()-category_start, 'audit': [audit[i] for i in indices], 'tracker_recoveries': recoveries, 'tracking_policy': 'Contiguous observed text-seeded propagation; new IDs after an empty-tracker restart'}
        (OUT / 'segmentation.json').write_text(json.dumps(report, indent=2))
        predictor.handle_request({'type': 'close_session', 'session_id': session})
        print('SEGMENTED', key, len(indices), flush=True)
report['seconds'] = time.time() - start
(OUT / 'segmentation.json').write_text(json.dumps(report, indent=2))
print('SEGMENTATION_COMPLETE', flush=True)
