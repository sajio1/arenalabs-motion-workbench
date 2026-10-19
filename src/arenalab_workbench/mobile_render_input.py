"""Adapt calibrated EgoDex FOV to the existing native-arm compositor."""
from pathlib import Path
import json,os,sys,hashlib
import h5py,numpy as np
from arenalab_workbench.scene_job import ROOT,save

def prepare(bodywork,dest):
    bodywork=Path(bodywork);dest=Path(dest)
    source=json.loads((bodywork/'input_manifest.json').read_text())
    if source['dataset']!='EgoDex':raise ValueError('Calibrated source camera is required for this compositor')
    scene=json.loads((dest/'scene.json').read_text());perception=ROOT/'scene-jobs'/scene['geometry']['scene_work']
    key=hashlib.sha256(str(perception.resolve()).encode()+(perception/'depth.json').read_bytes()).hexdigest()[:20]
    out=bodywork/'mobile-render-input'/key;(out/'scene').mkdir(parents=True,exist_ok=True)
    for name in ['depth.npz','depth-store.json','depth-runs','depth.json','frames','prompts.json','masks']:
        path=out/'scene'/name
        if (perception/name).exists() and not path.exists():path.symlink_to(perception/name,target_is_directory=(perception/name).is_dir())
    start=int((source.get('source_window') or {}).get('start_frame',0));n=source['frames']
    with h5py.File(source['source']) as f:
        camera=f['transforms/camera'][start:start+n];intrinsic=f['camera/intrinsic'][()]
    # Compositor and viewer both use the unmodified EgoDex Y-up world here.
    # Intrinsics are resized from the source image dimensions in the pinned env.
    import cv2
    video=source['video'];cap=cv2.VideoCapture(video['path']);scale=np.diag([960/cap.get(cv2.CAP_PROP_FRAME_WIDTH),540/cap.get(cv2.CAP_PROP_FRAME_HEIGHT),1]);cap.release()
    np.savez_compressed(out/'scene/cameras.npz',poses=camera,intrinsic=scale@intrinsic)
    save(out/'source.json',dict(render_scene_work=str(out/'scene'),head_video=video['path'],start_seconds=video.get('start_seconds',0),frame_times=video.get('frame_times')))
    return out

if __name__=='__main__':
    from arenalab_workbench.gripper_render import run
    body,dest,robot=sys.argv[1:];run(prepare(body,dest),robot,dest)
