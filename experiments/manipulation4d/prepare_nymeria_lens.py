from pathlib import Path
import json,numpy as np
from projectaria_tools.core import data_provider
root=Path('/home/ubuntu/arenalab-experiments/manipulation4d');out=root/'nymeria-calibration';out.mkdir(exist_ok=True)
import argparse
parser=argparse.ArgumentParser(description='Build the optical remap for an upright Nymeria RGB video from its own VRS calibration.')
parser.add_argument('sequence');parser.add_argument('vrs',type=Path);args=parser.parse_args()
entries=[(args.sequence,args.vrs)]
for sid,path in entries:
 p=data_provider.create_vrs_data_provider(str(path));cal=p.get_device_calibration().get_camera_calib('camera-rgb');size=cal.get_image_size();H,W=540,960;f=W/2;cx=(W-1)/2;cy=(H-1)/2
 maps=np.empty((H,W,2),'f4');valid=np.zeros((H,W),'u1')
 for y in range(H):
  for x in range(W):
   # Source MP4 is the upright (90 degree clockwise) RGB view.
   uv=cal.project(np.array([(y-cy)/f,-(x-cx)/f,1.]))
   if uv is None:maps[y,x]=-1;continue
   maps[y,x]=[(size[1]-1-uv[1])/size[1],uv[0]/size[0]];valid[y,x]=1
 K=np.array([[f,0,cx],[0,f,cy],[0,0,1.]])
 np.savez_compressed(out/(sid+'.npz'),normalized_map=maps,valid=valid,intrinsic=K)
 (out/(sid+'.json')).write_text(json.dumps(dict(source=str(path),model=str(cal.get_model_name()),projection_params=cal.get_projection_params().tolist(),source_image_size=size.tolist(),policy='aria-fisheye624-upright-pinhole-v1',output_size=[W,H],horizontal_fov_deg=90,world_registration='unavailable; lens calibration is not an Xsens-to-Aria world transform'),indent=2))
 print('CALIBRATION',sid,int(valid.sum()),flush=True)
