import os
os.environ['PYOPENGL_PLATFORM']='egl'
from pathlib import Path
import sys,numpy as np,trimesh,pyrender,cv2,imageio.v2 as imageio
np.infty=np.inf  # pyrender 0.1.45 compatibility with NumPy 2.
from scipy.ndimage import gaussian_filter1d
root=Path(os.environ.get('ARENA_OUTPUTDIR',Path(__file__).resolve().parent));label=sys.argv[1];d=np.load(root/(label+'-render.npz'))
motion=np.load(os.environ['ARENA_MOTION_PATH']) if 'ARENA_MOTION_PATH' in os.environ else None
p=d['positions'];rot=d['rotations'];n=len(p)
scene=pyrender.Scene(bg_color=[.055,.075,.07,1],ambient_light=[.65,.65,.65])
floor=trimesh.creation.box(extents=[30,30,.04]);floor.apply_translation([p[:,:,0].mean(),p[:,:,1].mean(),-.025]);floor.visual.vertex_colors=[72,87,79,255];scene.add(pyrender.Mesh.from_trimesh(floor))
grid=[]
for axis in [0,1]:
 for tick in range(-30,31):
  size=[30,.006,.002] if axis==0 else [.006,30,.002];g=trimesh.creation.box(extents=size);xyz=[0,0,.001];xyz[1-axis]=tick*.5;g.apply_translation(xyz);grid.append(g)
g=trimesh.util.concatenate(grid);g.visual.vertex_colors=[88,105,94,255];scene.add(pyrender.Mesh.from_trimesh(g))
nodes=[]
for k in range(p.shape[1]):
 c=np.asarray(d[f'c{k}'],dtype=float).ravel();c=c/255 if c.max()>1 else c;c=np.r_[c[:3],1.]
 material=pyrender.MetallicRoughnessMaterial(baseColorFactor=c,metallicFactor=.12,roughnessFactor=.65)
 nodes.append(scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(d[f'v{k}'],d[f'f{k}'],process=False),material=material)))
cn=scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(42)));ln=scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2.2));renderer=pyrender.OffscreenRenderer(1280,720)
centre=gaussian_filter1d(p[:,0].copy(),5,axis=0);centre[:,2]=.9
def cp(eye,target):
 z=eye-target;z/=np.linalg.norm(z);x=np.cross([0,0,1.],z);x/=np.linalg.norm(x);y=np.cross(z,x);m=np.eye(4);m[:3,:3]=np.stack([x,y,z],1);m[:3,3]=eye;return m
writer=imageio.get_writer(root/('02-h2-wuji.mp4' if label=='h2' else '03-a3-wuji.mp4'),fps=30,codec='libx264',quality=8,macro_block_size=1)
for f in range(n):
 for k,node in enumerate(nodes):
  m=np.eye(4);m[:3,:3]=rot[f,k];m[:3,3]=p[f,k];scene.set_pose(node,m)
 pose=cp(centre[f]+[2.8,-3.2,1.3],centre[f]);scene.set_pose(cn,pose);scene.set_pose(ln,pose)
 im=renderer.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)[0].copy()
 cv2.putText(im,label.upper()+' + WUJI | Original wrists + SMPL-X body prior',(26,38),cv2.FONT_HERSHEY_SIMPLEX,.82,(220,235,225),2)
 if motion is not None:
  err=motion['wrist_error_m'][f]*100;angle=motion['wrist_error_deg'][f]
  cv2.putText(im,f'Wrist error  L {err[0]:.2f} cm / {angle[0]:.1f} deg    R {err[1]:.2f} cm / {angle[1]:.1f} deg',(26,70),cv2.FONT_HERSHEY_SIMPLEX,.58,(180,230,190),1)
 cv2.putText(im,f'{f/30:.2f} s | 30 Hz | Kinematic playback | Fingers: RGB estimate + prior fallback',(26,690),cv2.FONT_HERSHEY_SIMPLEX,.6,(210,225,215),1)
 writer.append_data(im)
 if f in [0,120,240,360]:imageio.imwrite(root/f'{label}-{f:03d}.png',im)
writer.close();renderer.delete();print('ROBOT_VIDEO_COMPLETE',label)
