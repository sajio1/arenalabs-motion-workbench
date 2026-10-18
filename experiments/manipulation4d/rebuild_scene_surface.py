"""Re-fuse existing scene observations without rerunning or altering any model.

Works with legacy category masks and current job masks. A separate candidate
manifest references unchanged robot, foreground cloud, video and camera assets.
"""
from pathlib import Path
import argparse,hashlib,json,time
import cv2,numpy as np
from scene_surface import write_background

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(scene_path,depth_path,mask_dir,destination,url):
    scene=json.loads(scene_path.read_text());destination.mkdir(parents=True,exist_ok=True)
    with np.load(depth_path) as saved:depth={k:saved[k] for k in saved.files}
    raw_hash=sha(depth_path);T,H,W=depth['depth'].shape;source_camera=scene.get('source_camera')
    all_masks=np.zeros((T,H,W),bool)
    for path in sorted(mask_dir.glob('*-masks.npz')):
        with np.load(path) as saved:
            rows={int(i):mask for i,mask in zip(saved['frame_indices'],saved['masks'])}
        for i in range(T):
            raw=rows.get(i)
            if raw is None:continue
            if source_camera and not source_camera.get('estimated'):
                k=np.asarray(source_camera['intrinsic'],float).copy();k[0]*=raw.shape[1]/source_camera['width'];k[1]*=raw.shape[0]/source_camera['height']
                mask=cv2.warpPerspective(raw,depth['intrinsics'][i]@np.linalg.inv(k),(W,H),flags=cv2.INTER_NEAREST)
            else:mask=cv2.resize(raw,(W,H),interpolation=cv2.INTER_NEAREST)
            all_masks[i]|=mask>0
    valid=np.isfinite(depth['depth'])&(depth['depth']>.05)&~all_masks
    def block(stream,array):
        stream.write(b'\0'*((-stream.tell())%4));offset=stream.tell();stream.write(array.tobytes());return offset
    memory,report=write_background(destination/'background.bin',url+'background.bin',depth,valid,block)
    if memory is None:raise ValueError('No consistent static surface; original scene retained')
    original=scene.get('spatial_memory',{})
    report.update(original_points=original.get('count'),raw_depth_sha256=raw_hash,raw_depth_unchanged=sha(depth_path)==raw_hash)
    scene['spatial_memory']=memory;scene['geometry']['surface_fusion']=report
    scene['geometry']['background']='One TSDF zero-crossing surface. No frame-background overlay when fusion is enabled. Whole-clip offline; unseen surfaces remain unknown.'
    (destination/'scene.json').write_text(json.dumps(scene,separators=(',',':'),allow_nan=False));(destination/'surface-validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--depth',type=Path,required=True);p.add_argument('--masks',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--url',required=True);a=p.parse_args();run(a.scene,a.depth,a.masks,a.output,a.url)
