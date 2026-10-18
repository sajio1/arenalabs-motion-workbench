"""Display-only native arm preview, explicitly separate from a source FOV composite."""
from pathlib import Path
import json
import numpy as np,trimesh,pyrender,imageio.v2 as imageio
from scipy.spatial.transform import Rotation as R
from gripper_source import save,digest

def run(work,rid,dest):
 dest=Path(dest);folder=dest/('robot-'+rid);motion=json.loads((folder/'playback.json').read_text());selected=json.loads((folder/'render-layout.json').read_text())['arm_and_gripper_geoms'];model=trimesh.load(folder/'model.glb',force='scene')
 positions=np.asarray(motion['positions']);rotations=R.from_quat(np.asarray(motion['quaternions']).reshape(-1,4)).as_matrix().reshape(motion['frames'],-1,3,3)
 points=positions[:,selected].reshape(-1,3);center=(points.min(0)+points.max(0))/2;radius=max(.4,np.linalg.norm(points.max(0)-points.min(0))/2+.2)
 scene=pyrender.Scene(bg_color=[.04,.09,.065,1],ambient_light=[.65]*3);nodes=[(i,scene.add(pyrender.Mesh.from_trimesh(model.geometry[f'geom_{i}']))) for i in selected]
 eye=center+np.array([-.6,.5,-1.4])*radius;z=(eye-center)/np.linalg.norm(eye-center);x=np.cross([0,1,0],z);x/=np.linalg.norm(x);y=np.cross(z,x);pose=np.eye(4);pose[:3,:3]=np.stack([x,y,z],1);pose[:3,3]=eye
 scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(55)),pose=pose);scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2),pose=pose)
 renderer=pyrender.OffscreenRenderer(960,540);path=dest/(rid+'-arm-preview-v2.mp4');writer=imageio.get_writer(path,fps=30,codec='libx264',quality=8,macro_block_size=1,output_params=['-movflags','+faststart'])
 for t in range(motion['frames']):
  for i,node in nodes:
   transform=np.eye(4);transform[:3,:3]=rotations[t,i];transform[:3,3]=positions[t,i];scene.set_pose(node,transform)
  rgba,_=renderer.render(scene);writer.append_data(rgba[:,:,:3])
 writer.close();renderer.delete();from gpu_scheduler import merge_scene
 def update(asset):
  asset.setdefault('patch_videos',{})[rid]='/assets/'+path.relative_to('/opt/arenalab/store').as_posix()+'?v='+digest(path)[:16];asset.setdefault('patch_kinds',{})[rid]='native_preview';asset.get('patch_unavailable',{}).pop(rid,None)
 merge_scene(dest/'scene.json',update)
