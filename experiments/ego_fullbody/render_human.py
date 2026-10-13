import os
os.environ['PYOPENGL_PLATFORM']='egl'
from pathlib import Path
import numpy as np,trimesh,pyrender,cv2,imageio.v2 as imageio
np.infty=np.inf  # pyrender 0.1.45 compatibility with NumPy 2.
from scipy.ndimage import gaussian_filter1d

root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)));d=np.load(root/'smplx_motion.npz')
v=d['vertices'];faces=d['faces'];n=len(v)
scene=pyrender.Scene(bg_color=[.055,.075,.07,1],ambient_light=[.55,.55,.55])
floor=trimesh.creation.box(extents=[30,30,.04]);floor.apply_translation([v[:,:,0].mean(),v[:,:,1].mean(),-.025]);floor.visual.vertex_colors=[72,87,79,255]
scene.add(pyrender.Mesh.from_trimesh(floor))
grid=[]
for axis in [0,1]:
 for tick in range(-30,31):
  size=[30,.006,.002] if axis==0 else [.006,30,.002];g=trimesh.creation.box(extents=size);xyz=[0,0,.001];xyz[1-axis]=tick*.5;g.apply_translation(xyz);grid.append(g)
g=trimesh.util.concatenate(grid);g.visual.vertex_colors=[88,105,94,255];scene.add(pyrender.Mesh.from_trimesh(g))
camera=pyrender.PerspectiveCamera(yfov=np.deg2rad(42));cn=scene.add(camera)
ln=scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2.5))
renderer=pyrender.OffscreenRenderer(1280,720)
material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.74,.80,.77,1],metallicFactor=.05,roughnessFactor=.7)
centre=gaussian_filter1d(d['joints'][:,0].copy(),5,axis=0);centre[:,2]=.9
def camera_pose(eye,target):
 z=(eye-target);z/=np.linalg.norm(z);x=np.cross([0,0,1.],z);x/=np.linalg.norm(x);y=np.cross(z,x);m=np.eye(4);m[:3,:3]=np.stack([x,y,z],1);m[:3,3]=eye;return m
writer=imageio.get_writer(root/'01-smplx.mp4',fps=30,codec='libx264',quality=8,macro_block_size=1)
node=None
for i in range(n):
 if node is not None:scene.remove_node(node)
 node=scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(v[i],faces,process=False),material=material,smooth=True))
 pose=camera_pose(centre[i]+np.array([2.8,-3.2,1.3]),centre[i]);scene.set_pose(cn,pose);scene.set_pose(ln,pose)
 im=renderer.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)[0].copy()
 cv2.putText(im,'SMPL-X | Nymeria head + wrists -> EgoAllo',(26,38),cv2.FONT_HERSHEY_SIMPLEX,.85,(220,235,225),2)
 cv2.putText(im,f'{i/30:.2f} s | 30 Hz | Fingers: inferred prior, not source observations',(26,690),cv2.FONT_HERSHEY_SIMPLEX,.6,(210,225,215),1)
 writer.append_data(im)
 if i in [0,120,240,360]:imageio.imwrite(root/f'smplx-{i:03d}.png',im)
writer.close();renderer.delete();print('SMPLX_VIDEO_COMPLETE')
