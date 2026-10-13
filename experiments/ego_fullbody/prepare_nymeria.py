"""Produce a strictly sparse head/wrist observation file, plus separate held-out truth."""
from pathlib import Path
import importlib.util,json,hashlib,argparse
import numpy as np
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--start-frame',type=int,required=True);p.add_argument('--frames',type=int,default=450);p.add_argument('--clip',default='unspecified');p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
from greenwich_pipelines.pipelines.nymeria import xsens_clock as clock
src=args.source
d=np.load(src);times,report=clock.recording_clock(d['timestamps_us'])
start=args.start_frame;count=args.frames
if start<1 or count<3:raise ValueError('Need one predecessor and at least three output frames')
query=times[0]+np.arange(start-1,start+count)*1e6/30
if query[-1]>times[-1]:raise ValueError('Clip exceeds source recording')
rows=clock.nearest_source_rows(times,query)
p=d['segment_tXYZ'][rows].reshape(-1,23,3)
r=Rotation.from_quat(d['segment_qWXYZ'][rows].reshape(-1,4)[:,[1,2,3,0]]).as_matrix().reshape(-1,23,3,3)
rest=Rotation.from_quat(d['tpose_segment_qWXYZ'].reshape(23,4)[:,[1,2,3,0]]).as_matrix()
relative=r@rest.transpose(0,2,1)
tp=d['tpose_segment_tXYZ'].reshape(23,3)
# SMPL anatomical x=left, y=up, z=forward, expressed in Xsens world.
B=np.array([[0,0,1],[1,0,0],[0,1,0.]])
head_r=relative[:,6]@B
forward=[];normal=[]
for wrist,elbow in [(14,13),(10,9)]:
    direction=tp[wrist]-tp[elbow];direction/=np.linalg.norm(direction)
    forward.append(np.einsum('tij,j->ti',relative[:,wrist],direction))
    normal.append(np.einsum('tij,j->ti',relative[:,wrist],[0,0,-1.]))
out=args.output;out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/'observations.npz',head_position=p[:,6],head_rotation=head_r,wrist_position=p[1:,[14,10]],palm_forward=np.stack(forward,1)[1:],palm_normal=np.stack(normal,1)[1:],timestamps_s=np.arange(count)/30,floor_z=np.array(0.))
np.savez_compressed(out/'heldout_xsens.npz',positions=p[1:],rotations=r[1:],raw_rows=rows[1:])
(out/'input_manifest.json').write_text(json.dumps(dict(dataset='Nymeria',source=str(src),clip=args.clip,start_frame_30hz=start,frames=count,fps=30,model_inputs=['head pose','left/right wrist pose'],withheld=['pelvis','feet','other body joints'],finger_observations=False,cpf_policy='anatomical head plus SMPL-H neutral eye offset; not calibrated Aria camera',floor_policy='Xsens world z=0',clock=report),indent=2))
print('Saved',count,'frames. Head range:',p[:,6].min(0),p[:,6].max(0))
