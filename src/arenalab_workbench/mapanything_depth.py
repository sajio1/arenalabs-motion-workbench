"""MapAnything depth-only worker. No meshes, splats or robot-policy changes.

Camera-conditioned, overlapping bounded windows. Supplied optical camera poses
and intrinsics remain unchanged in the assembled contract; native predictions
are retained separately. No arbitrary SAM-mask input is claimed for the model.
"""
from pathlib import Path
import hashlib
import json
import os
import time
import uuid
import numpy as np
from arenalab_workbench.depth_store import MODEL, POLICY, create_array
from arenalab_workbench.settings import paths


def windows(n, size=32, overlap=8):
    if n < 1 or size < 2 or not 0 < overlap < size:
        raise ValueError('Require frames>0 and 0<overlap<window')
    result = []
    start = 0
    while start < n:
        stop = min(start+size, n)
        result.append(np.arange(start, stop))
        if stop == n:
            break
        start = stop-overlap
    return result


def selection(chunks, n):
    # Prefer greatest temporal context; never average independently predicted depth.
    return [max((min(i-int(c[0]), int(c[-1])-i), j)
                for j, c in enumerate(chunks) if c[0] <= i <= c[-1])[1] for i in range(n)]


def file_hash(path):
    path=Path(path)
    if path.is_dir():
        h=hashlib.sha256()
        for item in sorted(p for p in path.rglob('*') if p.is_file()):
            h.update(item.relative_to(path).as_posix().encode());h.update(bytes.fromhex(file_hash(item)))
        return h.hexdigest()
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def run(work):
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    import torch
    import zarr
    from PIL import Image
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs
    from arenalab_workbench.gpu_scheduler import write

    work = Path(work)
    if not paths.mapanything_weights.is_dir() or not torch.cuda.is_available():
        raise RuntimeError('Provision local MapAnything weights and a CUDA runtime first')
    ids = np.load(work/'frames.npz')['ids']
    names = sorted((work/'frames').glob('*.jpg'))
    if len(names) != len(ids):
        raise ValueError('Prepared RGB/depth frame coverage mismatch')
    camera_file = next((work/name for name in ['cameras.npz','camera-poses.npz'] if (work/name).exists()),None)
    camera = np.load(camera_file) if camera_file is not None else None
    calibration = camera if camera is not None and 'intrinsic' in camera else (
        np.load(work/'camera-intrinsics.npz') if (work/'camera-intrinsics.npz').exists() else None)
    if camera is not None:
        poses = camera['poses'][ids]
        if poses.shape != (len(ids), 4, 4) or not np.isfinite(poses).all():
            raise ValueError('Optical source poses do not cover the prepared source clock')
    size, overlap = paths.depth_window, paths.depth_overlap
    # An unknown camera gauge cannot be independently reset in successive windows.
    if camera is None and len(ids) > size:
        raise ValueError('Long MapAnything input requires synchronized optical camera poses')
    chunks = windows(len(ids), size, overlap)
    owners = selection(chunks, len(ids))
    run_dir = work/'depth-runs'/uuid.uuid4().hex
    raw = run_dir/'native';raw.mkdir(parents=True)
    input_paths = [work/'frames.npz', *names]
    input_paths += [work/name for name in ['cameras.npz','camera-poses.npz','camera-intrinsics.npz'] if (work/name).exists()]
    inputs = {str(p.relative_to(work)): file_hash(p) for p in input_paths}
    write(run_dir/'input.json', dict(policy=POLICY, model=MODEL, input_sha256=inputs,
        weights_sha256={p.name:file_hash(p) for p in paths.mapanything_weights.glob('*.safetensors')},
        window=size, overlap=overlap))
    started = time.monotonic()
    model = MapAnything.from_pretrained(str(paths.mapanything_weights), local_files_only=True).to('cuda').eval()
    store = None
    receipts = []
    for number, indices in enumerate(chunks):
        values, known_k = [], []
        for index in indices:
            image = Image.open(names[index]).convert('RGB')
            # Placeholder K is used only to record the exact image preprocessing
            # affine when calibration is absent; never fed as measured intrinsics.
            k = calibration['intrinsic'].astype('f4') if calibration is not None else np.array(
                [[image.width,0,image.width/2],[0,image.width,image.height/2],[0,0,1]], dtype='f4')
            value = dict(img=image, intrinsics=torch.tensor(k))
            if camera is not None:
                value.update(camera_poses=torch.tensor(poses[index].astype('f4')), is_metric_scale=torch.tensor([True]))
            values.append(value);known_k.append(k)
        views = preprocess_inputs(values)
        affines, intrinsics = [], []
        for view, k in zip(views, known_k):
            intrinsic = view['intrinsics'].squeeze().numpy().copy()
            affines.append(intrinsic@np.linalg.inv(k));intrinsics.append(intrinsic)
            if calibration is None:
                view.pop('intrinsics')
        with torch.inference_mode():
            predictions = model.infer(views, memory_efficient_inference=True,
                minibatch_size=1, use_amp=True, amp_dtype='bf16', apply_mask=True, mask_edges=True)
        def array(value, key):
            return value[key].detach().float().cpu().numpy()[0]
        native = dict(ids=ids[indices], depth=np.stack([array(p,'depth_z').squeeze(-1) for p in predictions]),
            valid=np.stack([array(p,'mask').squeeze(-1).astype(bool) for p in predictions]),
            intrinsics=np.stack([array(p,'intrinsics') for p in predictions]),
            poses=np.stack([array(p,'camera_poses') for p in predictions]))
        if 'conf' in predictions[0]:
            native['conf'] = np.stack([array(p,'conf').reshape(native['depth'].shape[1:]) for p in predictions])
        raw_file = raw/f'{number:05d}.zarr'
        # Retain native depth/camera/validity predictions before any source anchoring.
        # No derived XYZ sequence or input RGB is redundantly persisted here.
        native_store=zarr.open_group(str(raw_file),mode='w')
        for key,value in native.items():
            create_array(native_store,key,value.shape,value.dtype,chunks=value.shape)[:]=value
        h, w = native['depth'].shape[1:]
        if store is None:
            store = zarr.open_group(str(run_dir/'depth.zarr'), mode='w')
            for name, shape, dtype in [('valid',(len(ids),h,w),'bool'),
                ('intrinsics',(len(ids),3,3),'f8'),('extrinsics',(len(ids),3,4),'f8'),
                ('preprocessing_affine',(len(ids),3,3),'f8'),('ids',(len(ids),),'i8')]:
                create_array(store,name,shape,dtype)
            store.attrs.update(native_shape=[len(ids),h,w],native_confidence='conf' in native,
                native_chunks=[f'native/{j:05d}.zarr' for j in range(len(chunks))],
                native_owners=owners,native_locals=[int(i-chunks[j][0]) for i,j in enumerate(owners)])
        for local, index in enumerate(indices):
            if owners[index] != number:
                continue
            pose = poses[index] if camera is not None else native['poses'][local]
            k = intrinsics[local] if calibration is not None else native['intrinsics'][local]
            for name, value in dict(valid=native['valid'][local],
                intrinsics=k, extrinsics=np.linalg.inv(pose)[:3],
                preprocessing_affine=affines[local], ids=ids[index]).items():
                store[name][index] = value
        receipts.append(dict(path=str(raw_file.relative_to(work)), sha256=file_hash(raw_file)))
        print('MAPANYTHING',number,len(indices),time.monotonic()-started,flush=True)
        del predictions, views, values, native
    if any(file_hash(work/name) != checksum for name,checksum in inputs.items()):
        raise ValueError('Source camera or images changed during depth inference')
    metadata = dict(policy=POLICY, model=MODEL, inference_seconds=time.monotonic()-started,
        frames=len(ids), fps=15, camera_conditioned=camera is not None,
        measured_camera=calibration is not None and camera is not None and not (work/'camera-calibration.json').exists(),
        camera_pose_max_difference=0. if camera is not None else None,
        measured_depth=False, source_trajectory_edited=False, post_scale_applied=False,
        window=size, overlap=overlap, native_resolution=[w,h], native_predictions=receipts,
        window_assembly='Greatest temporal context; source optical camera anchors fixed',
        masks='SAM masks remain downstream; native model has no external mask input',
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), weights_finetuned=False)
    write(work/'depth.json',metadata)
    write(work/'depth-store.json',dict(schema='depth-store-v2', path=str((run_dir/'depth.zarr').relative_to(work))))


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
