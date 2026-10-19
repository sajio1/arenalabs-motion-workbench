"""Independent GPU stages for a scene manifest; one shared policy for all clips."""
from arenalab_workbench.settings import paths
from pathlib import Path
import hashlib,json,os,re,sys,time
os.environ.setdefault('CUDA_VISIBLE_DEVICES','3')
# Depth Anything has its existing, pinned environment. Do not blend its Torch /
# Transformers dependencies with the SAM/Qwen environment.
if sys.argv[2]=='depth':
    depth_python=paths.da3_python
    if Path(sys.executable).absolute()!=depth_python:
        os.execv(str(depth_python),[str(depth_python),*sys.argv])
import cv2,h5py,numpy as np
from arenalab_workbench.video_io import read_at

work=Path(sys.argv[1]);stage=sys.argv[2];manifest=json.loads((work/'manifest.json').read_text());row=manifest['source']
def save(name,value):(work/name).write_text(json.dumps(value,indent=2,allow_nan=False))

if stage=='prepare':
    frames=work/'frames';frames.mkdir(exist_ok=True);(work/'masks').mkdir(exist_ok=True)
    cap=cv2.VideoCapture(row['video']['path']);fps=cap.get(cv2.CAP_PROP_FPS)
    if fps<=0:raise ValueError('Video has no valid frame rate')
    n=int(row['frames']);ids=np.arange(0,n,2);start=float(row['video'].get('start_seconds',0));clock=row['video'].get('frame_times')
    rectification=None
    if manifest['public']['dataset']=='Nymeria':
        sequence=row.get('metadata',{}).get('Sequence',Path(row['source']).parent.name)
        profile=paths.workbench / 'nymeria-calibration'/(sequence+'.npz')
        if not profile.exists():raise ValueError('Nymeria lens calibration is missing; refusing uncalibrated fisheye depth')
        rectification=np.load(profile)
        np.savez_compressed(work/'lens-rectification.npz',**dict(rectification))
        np.savez_compressed(work/'camera-intrinsics.npz',intrinsic=rectification['intrinsic'])
        save('lens-calibration.json',json.loads(profile.with_suffix('.json').read_text()))
    if rectification is not None:maps=rectification['normalized_map']*np.array([cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT)],dtype='f4')
    used=[]
    for i,frame in enumerate(ids):
        seconds=float(clock[frame]) if clock else start+frame/float(row['fps'])
        ok,image=read_at(cap,seconds)
        if not ok:raise ValueError(f'Source video frame unavailable at {seconds:.3f}s')
        if i==0:height,width=image.shape[:2]
        if rectification is not None:
            if width!=height:raise ValueError('Nymeria calibration requires the uncropped square source view')
            image=cv2.remap(image,maps[:,:,0],maps[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        else:image=cv2.resize(image,(960,540))
        cv2.imwrite(str(frames/f'{i:05d}.jpg'),image);used.append(seconds)
    cap.release()
    calibrated=False
    if rectification is not None:
        from arenalab_workbench.nymeria_camera import prepare_camera
        prepare_camera(row,work,paths.workbench)
        calibrated=True
    # EgoDex provides optical camera intrinsics + source-world camera transforms.
    if Path(row['source']).suffix.lower() in {'.hdf5','.h5'}:
        with h5py.File(row['source']) as f:
            if 'camera/intrinsic' in f and 'transforms/camera' in f:
                offset=int(row.get('source_window',{}).get('start_frame',0));poses=f['transforms/camera'][offset:offset+n]
                if len(poses)!=n:raise ValueError('Camera poses do not cover source window')
                K=f['camera/intrinsic'][()].copy();K[0]*=960/width;K[1]*=540/height
                if not np.isfinite(poses).all():raise ValueError('Nonfinite source camera poses')
                np.savez_compressed(work/'cameras.npz',poses=poses,intrinsic=K);calibrated=True
    if Path(row['source']).suffix.lower()=='.npz' and manifest['public'].get('uploaded'):
        with np.load(row['source']) as data:
            if 'camera' in data and 'intrinsic' in data:
                k=data['intrinsic'].copy();k[0]*=960/width;k[1]*=540/height
                np.savez_compressed(work/'cameras.npz',poses=data['camera'],intrinsic=k);calibrated=True
    np.savez_compressed(work/'frames.npz',ids=ids,times=np.asarray(used),source_times=np.arange(n)/float(row['fps']))
    save('preparation.json',dict(frames=len(ids),source_frames=n,source_fps=row['fps'],mask_fps=15,calibrated_camera=calibrated,video_sampling_seconds=used,width=960,height=540,source_trajectory_edited=False))

elif stage=='discover':
    from arenalab_workbench.discover_objects import run
    run(work)

elif stage=='depth':
    import torch
    from depth_anything_3.api import DepthAnything3
    names=sorted((work/'frames').glob('*.jpg'));images=[cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB) for p in names]
    ids=np.load(work/'frames.npz')['ids'];model_id='depth-anything/DA3NESTED-GIANT-LARGE-1.1';model=DepthAnything3.from_pretrained(model_id).to('cuda');camera=np.load(work/'cameras.npz') if (work/'cameras.npz').exists() else None
    start=time.monotonic();out={k:[] for k in ['depth','conf','images','intrinsics','extrinsics']};anchors=np.linspace(0,len(ids)-1,min(9,len(ids))).astype(int)
    # With no measured camera poses use one global prediction, never independently
    # scaled batches, and never register the guessed scene to source robot targets.
    cores=[np.arange(len(ids))] if camera is None else [np.arange(i,min(i+16,len(ids))) for i in range(0,len(ids),16)]
    for core in cores:
        batch=np.unique(np.r_[core,anchors]);kwargs={}
        if camera is None and (work/'camera-intrinsics.npz').exists():kwargs['intrinsics']=np.repeat(np.load(work/'camera-intrinsics.npz')['intrinsic'][None],len(batch),0).astype('f4')
        if camera is not None:kwargs=dict(extrinsics=np.linalg.inv(camera['poses'][ids[batch]]).astype('f4'),intrinsics=np.repeat(camera['intrinsic'][None],len(batch),0).astype('f4'),align_to_input_ext_scale=True)
        p=model.inference([images[i] for i in batch],process_res=504,ref_view_strategy='saddle_balanced',**kwargs);sel=np.searchsorted(batch,core)
        for key in out:out[key].append(getattr(p,'processed_images' if key=='images' else key)[sel])
        print('DEPTH',int(core[0]),len(core),time.monotonic()-start,flush=True)
    np.savez_compressed(work/'depth.npz',**{k:np.concatenate(v) for k,v in out.items()},ids=ids)
    error=float(np.max(abs(np.concatenate(out['extrinsics'])-np.linalg.inv(camera['poses'][ids])[:,:3]))) if camera is not None else None
    if error is not None and error>1e-4:raise ValueError('Depth inference altered the supplied source camera poses')
    save('depth.json',dict(model=model_id,inference_seconds=time.monotonic()-start,frames=len(ids),fps=15,measured_camera=camera is not None,camera_pose_max_difference=error,post_scale_applied=False,source_trajectory_edited=False,physical_gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES','3')),measured_depth=False,lens_rectified=(work/'lens-calibration.json').exists()))
else:raise ValueError('Unknown inference stage')
