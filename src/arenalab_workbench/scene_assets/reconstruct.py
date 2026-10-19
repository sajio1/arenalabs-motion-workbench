"""Bounded MapAnything chunks, masked global registration and disk-backed geometry."""
from pathlib import Path
from functools import lru_cache
import json
import time
import numpy as np

from .formats import sha256, write_json
from .geometry import windows, robust_similarity, apply_transform, pose_graph, backproject
from .store import NpzArrays, create_geometry
from .loops import describe, verified_loop


class MaskFrames:
    def __init__(self, work, ids, scratch):
        self.layers,self.instances = [],[]
        next_label = 1
        for key,category,*_ in json.loads((work/"prompts.json").read_text()):
            saved = NpzArrays(work/"masks"/f"{key}-masks.npz",scratch/key)
            if not np.array_equal(saved["frame_indices"],np.arange(len(ids))):
                raise ValueError(f"Mask frame coverage mismatch: {key}")
            if "source_frame_indices" in saved and not np.array_equal(saved["source_frame_indices"],ids):
                raise ValueError(f"Mask timestamps disagree: {key}")
            values = set()
            for frame in saved["masks"]:
                values.update(np.unique(frame).tolist())
            mapping = {}
            for instance in sorted(values-{0}):
                if key.startswith("human-"):
                    mapping[instance] = 65535
                else:
                    if next_label>=65535:
                        raise ValueError("Too many object instances")
                    mapping[instance] = next_label
                    self.instances.append(dict(id=next_label,category=category,mask_key=key,source_instance=int(instance)))
                    next_label += 1
            self.layers.append((saved,mapping))
        if not self.layers:
            raise ValueError("SAM human/hardware/object masks are required")

    def frame(self,index):
        labels = None
        for saved,mapping in self.layers:
            mask = saved["masks"][index]
            if labels is None:
                labels = np.zeros(mask.shape,dtype="uint16")
            for source,target in mapping.items():
                labels[(mask==source)&(labels!=65535)] = target
        return labels


def run(work, output, model_path, size=32, overlap=8, stride=1):
    import cv2
    import torch
    from PIL import Image
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    work,output,model_path = map(Path,[work,output,model_path])
    if not model_path.is_dir() or not torch.cuda.is_available():
        raise RuntimeError("Explicitly provisioned local MapAnything weights and cloud CUDA are required")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve raw priors; choose a fresh reconstruction directory")
    output.mkdir(parents=True,exist_ok=True)
    inputs = [work/"manifest.json",work/"frames.npz",work/"prompts.json",*sorted((work/"masks").glob("*.npz")),
              *sorted((work/"frames").glob("*.jpg"))]
    inputs += [work/name for name in ["cameras.npz","camera-intrinsics.npz"] if (work/name).exists()]
    hashes = {str(p):sha256(p) for p in inputs}
    write_json(output/"input-receipt.json",dict(input_sha256=hashes,model_sha256={p.name:sha256(p) for p in model_path.glob("*.safetensors")},
        window=size,overlap=overlap,stride=stride))
    all_ids = np.load(work/"frames.npz")["ids"]
    selected = np.arange(0,len(all_ids),stride)
    ids = all_ids[selected]
    chunks = windows(len(ids),size,overlap)
    masks = MaskFrames(work,all_ids,output/"input-arrays")
    images = sorted((work/"frames").glob("*.jpg"))
    if len(images)!=len(all_ids):
        raise ValueError("Prepared image coverage mismatch")
    camera = NpzArrays(work/"cameras.npz",output/"input-arrays/cameras") if (work/"cameras.npz").exists() else None
    calibration = camera if camera is not None else (NpzArrays(work/"camera-intrinsics.npz",output/"input-arrays/intrinsics")
        if (work/"camera-intrinsics.npz").exists() else None)
    model = MapAnything.from_pretrained(str(model_path),local_files_only=True).to("cuda").eval()
    raw = output/"raw-chunks";raw.mkdir()
    auxiliary = output/"chunk-context";auxiliary.mkdir()
    started = time.monotonic()
    camera_audits = []
    store = None
    for number,indices in enumerate(chunks):
        values,original_ks = [],[]
        for index in indices:
            image = Image.open(images[selected[index]]).convert("RGB")
            k = np.asarray(calibration["intrinsic"],dtype="float32") if calibration is not None else np.array(
                [[image.width,0,image.width/2],[0,image.width,image.height/2],[0,0,1]],dtype="float32")
            value = dict(img=image,intrinsics=torch.tensor(k))
            if camera is not None:
                value.update(camera_poses=torch.tensor(camera["poses"][ids[index]].astype("float32")),is_metric_scale=torch.tensor([True]))
            values.append(value);original_ks.append(k)
        views = preprocess_inputs(values)
        labels,affines,known_ks = [],[],[]
        for index,view,k in zip(indices,views,original_ks):
            h,w = view["img"].shape[-2:]
            affine = view["intrinsics"].squeeze().numpy() @ np.linalg.inv(k)
            label = cv2.warpAffine(masks.frame(int(selected[index])),affine[:2],(w,h),flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT,borderValue=65535)
            label[cv2.dilate((label==65535).astype("uint8"),np.ones((7,7),"uint8"))>0] = 65535
            labels.append(label);affines.append(affine)
            known_ks.append(view["intrinsics"].squeeze().numpy().copy())
            if calibration is None:
                view.pop("intrinsics")
        with torch.inference_mode():
            predictions = model.infer(views,memory_efficient_inference=True,minibatch_size=1,
                use_amp=True,amp_dtype="bf16",apply_mask=True,mask_edges=True)
        def array(prediction,key):
            return prediction[key].detach().float().cpu().numpy()[0]
        chunk = dict(indices=indices,points=[],poses=[],depth=[],valid=[],k=[])
        rgbs = []
        for prediction in predictions:
            for key,source in [("points","pts3d"),("poses","camera_poses"),("k","intrinsics")]:
                chunk[key].append(array(prediction,source))
            chunk["depth"].append(array(prediction,"depth_z").squeeze(-1))
            chunk["valid"].append(array(prediction,"mask").squeeze(-1).astype(bool))
            rgbs.append(np.rint(np.clip(array(prediction,"img_no_norm"),0,1)*255).astype("uint8"))
        chunk = {k:np.asarray(v) for k,v in chunk.items()}
        # Raw outputs are never replaced by source anchors or pose graph results.
        np.savez_compressed(raw/f"{number:05}.npz",**chunk)
        np.savez_compressed(auxiliary/f"{number:05}.npz",labels=labels,images=rgbs,affines=affines,known_ks=known_ks)
        if camera is not None:
            expected = camera["poses"][ids[indices]]
            gauge = expected[0] @ np.linalg.inv(chunk["poses"][0])
            camera_audits.append(dict(chunk=number,reference_aligned_raw_pose_max_difference=float(np.max(np.abs(gauge@chunk["poses"]-expected)))))
        if store is None:
            store = create_geometry(output/"geometry.zarr",len(ids),h,w)
        del predictions,views,values,chunk,rgbs,labels
        print(json.dumps(dict(chunk=number,frames=indices.tolist(),seconds=time.monotonic()-started)),flush=True)
    del model
    torch.cuda.empty_cache()

    @lru_cache(maxsize=2)
    def load_chunk(number):
        saved = NpzArrays(raw/f"{number:05}.npz")
        context = NpzArrays(auxiliary/f"{number:05}.npz")
        return saved,context

    def frame_at(number,local):
        saved,context = load_chunk(number)
        index = chunks[number][local]
        depth = np.asarray(saved["depth"][local])
        pose = camera["poses"][ids[index]].copy() if camera is not None else saved["poses"][local].copy()
        k = context["known_ks"][local].copy() if calibration is not None else saved["k"][local].copy()
        return dict(depth=depth,poses=pose,k=k,points=backproject(depth,k,pose),valid=saved["valid"][local],
            labels=context["labels"][local],images=context["images"][local],affine=context["affines"][local])

    transforms,edges,audits = [np.eye(4)],[],[]
    for b in range(1,len(chunks)):
        for a in range(b):
            common = np.intersect1d(chunks[a],chunks[b])
            if not len(common):
                continue
            pa,pb = [],[]
            for index in common:
                left = frame_at(a,int(np.flatnonzero(chunks[a]==index)[0]))
                right = frame_at(b,int(np.flatnonzero(chunks[b]==index)[0]))
                good = left["valid"] & right["valid"] & (left["labels"]==0) & (right["labels"]==0)
                ya,xa = np.where(good)
                sample = np.arange(0,len(ya),max(1,len(ya)//800))[:800]
                pa.append(left["points"][ya[sample],xa[sample]])
                pb.append(right["points"][ya[sample],xa[sample]])
            pa,pb = np.concatenate(pa),np.concatenate(pb)
            if camera is not None:
                audit = dict(transform_applied=False,reason="Recorded optical camera anchors fixed",points=len(pa))
                transform = np.eye(4)
            else:
                transform,audit = robust_similarity(pb,pa,metric=False)
                keep = np.linalg.norm(apply_transform(transform,pb)-pa,axis=-1)<.05
                edges.append((a,b,pa[keep][::4],pb[keep][::4]))
            audits.append(dict(a=a,b=b,kind="overlap",**audit))
            if a==b-1:
                transforms.append(transforms[a]@transform)
    # Bounded keyframe descriptors propose only non-overlapping chunk pairs.
    descriptors = []
    for j in range(len(chunks)):
        f = frame_at(j,len(chunks[j])//2)
        descriptors.append(describe(f["images"],f["valid"]&(f["labels"]==0))[0])
    loop_count = 0
    for b in range(2,len(chunks)):
        candidates = [a for a in range(b) if not len(np.intersect1d(chunks[a],chunks[b]))]
        candidates = sorted(candidates,key=lambda a:float(descriptors[a]@descriptors[b]),reverse=True)[:3]
        for a in candidates:
            match,audit = verified_loop(frame_at(a,len(chunks[a])//2),frame_at(b,len(chunks[b])//2),metric=False)
            audits.append(dict(a=a,b=b,kind="nonoverlap_loop",**audit,transform_applied=camera is None and match is not None))
            if match is not None:
                loop_count += 1
                if camera is None:
                    edges.append((a,b,*match))
    if len(transforms)!=len(chunks):
        raise ValueError("Disconnected chunks; source trajectories must not be moved to hide this")
    if camera is None:
        transforms,graph_audit = pose_graph(transforms,edges,metric=False)
    else:
        graph_audit = dict(success=True,mode="Recorded optical camera anchors fixed",edges=0)
    for index in range(len(ids)):
        _,j = max((min(int(index-c[0]),int(c[-1]-index)),j) for j,c in enumerate(chunks) if c[0]<=index<=c[-1])
        frame = frame_at(j,int(np.flatnonzero(chunks[j]==index)[0]))
        transform = transforms[j]
        scale = np.cbrt(np.linalg.det(transform[:3,:3]))
        pose = frame["poses"].copy()
        pose[:3,:3] = transform[:3,:3]/scale @ pose[:3,:3]
        pose[:3,3] = apply_transform(transform,pose[None,:3,3])[0]
        for key,value in dict(ids=ids[index],poses=pose,depth=frame["depth"]*scale,k=frame["k"],valid=frame["valid"],
            images=frame["images"],labels=frame["labels"],preprocessing_affine=frame["affine"]).items():
            store[key][index] = value
    if any(sha256(p)!=digest for p,digest in hashes.items()):
        raise ValueError("Input changed during reconstruction")
    write_json(output/"geometry.json",dict(model="MapAnything",model_path=str(model_path),frames=len(ids),instances=masks.instances,
        input_camera_poses=camera is not None,world="source" if camera is not None else "estimated",
        scale="input metric" if camera is not None else "model-estimated metric + global Sim3 stitching",
        raw_camera_audits=camera_audits,pose_graph=graph_audit,registration=audits,verified_nonoverlap_loops=loop_count,
        mask_policy="SAM exclusions in static registration, loop closure and GS losses; native backbone has no external mask key",
        memory_policy="One model chunk; two disk-mapped registration chunks; frame-chunked Zarr, no retained dense sequence",
        seconds=time.monotonic()-started,raw_chunks=[dict(path=p.name,sha256=sha256(p)) for p in sorted(raw.glob("*.npz"))]))
