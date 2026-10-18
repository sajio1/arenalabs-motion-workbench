import os
os.environ['PYOPENGL_PLATFORM']='egl'
from pathlib import Path
import numpy as np,pyrender,trimesh,cv2,imageio.v2 as imageio
np.infty=np.inf
r=Path(os.environ['ARENA_OUTPUTDIR']);d=np.load(r/'h2/h2-render.npz');p=d['positions'];rot=d['rotations']
scenes=[]
for a,b in d['hand_ranges']:
 s=pyrender.Scene(bg_color=[.06,.08,.07,1],ambient_light=[.65]*3);nodes=[]
 for k in range(a,b):
  mesh=trimesh.Trimesh(d[f'v{k}'],d[f'f{k}'],process=False)
  nodes.append(s.add(pyrender.Mesh.from_trimesh(mesh,material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.65,.8,.7,1],roughnessFactor=.6))))
 # Hand root frame is fixed in this close-up; no effect on robot playback.
 eye=np.array([.27,-.32,.28]);target=np.array([0.,0.,.09]);z=eye-target;z/=np.linalg.norm(z);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
 m=np.eye(4);m[:3,:3]=np.stack([x,y,z],1);m[:3,3]=eye
 s.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(40)),pose=m);s.add(pyrender.DirectionalLight(color=[1]*3,intensity=2),pose=m)
 scenes.append((s,nodes,int(a),int(b)))
renderer=pyrender.OffscreenRenderer(640,720);writer=imageio.get_writer(r/'wuji-closeup.mp4',fps=30,codec='libx264',quality=7,macro_block_size=1)
for f in range(60,len(p)):
 ims=[]
 for side,(s,nodes,a,b) in enumerate(scenes):
  R=rot[f,a];origin=p[f,a]
  for k,node in zip(range(a,b),nodes):
   m=np.eye(4);m[:3,:3]=R.T@rot[f,k];m[:3,3]=R.T@(p[f,k]-origin);s.set_pose(node,m)
  im=renderer.render(s)[0].copy();weight=float(d['finger_weight'][f,side])
  cv2.putText(im,['LEFT WUJI','RIGHT WUJI'][side],(24,40),0,.8,(230,240,235),2)
  cv2.putText(im,'RGB update + prior' if weight>.05 else 'Prior fallback',(24,675),0,.7,(210,230,215),2)
  ims.append(im)
 writer.append_data(np.concatenate(ims,axis=1))
 if f==180:imageio.imwrite(r/'wuji-closeup-preview.jpg',np.concatenate(ims,axis=1))
writer.close();renderer.delete()
