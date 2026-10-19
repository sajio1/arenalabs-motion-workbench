"""Render native robot arms + the selected gripper through the source camera."""
from arenalab_workbench.settings import paths
from pathlib import Path
import os,sys,json
os.environ['PYOPENGL_PLATFORM']='egl';os.environ.setdefault('CUDA_VISIBLE_DEVICES','3');os.environ.setdefault('EGL_DEVICE_ID','3')
import numpy as np,cv2,trimesh,pyrender,imageio.v2 as imageio
from scipy.spatial.transform import Rotation as R
from arenalab_workbench.gripper_source import save
from arenalab_workbench.fov_cache import read_at,inpaint
from arenalab_workbench.recorded_camera_distortion import recorded_render_map
np.infty=np.inf

def run(work,rid,dest):
 work=Path(work);dest=Path(dest);folder=dest/('robot-'+rid);scene_path=work/'scene';scene=json.loads((dest/'scene.json').read_text());meta=json.loads((work/'source.json').read_text());scene_path=Path(meta.get('render_scene_work') or scene.get('geometry',{}).get('scene_work',str(scene_path)))
 if not (scene_path/'cameras.npz').exists():
  from arenalab_workbench.robot_preview import run as preview
  preview(work,rid,dest);return
 playback=json.loads((folder/'playback.json').read_text());selected=json.loads((folder/'render-layout.json').read_text())['arm_and_gripper_geoms'];model=trimesh.load(folder/'model.glb',force='scene')
 # NpzFile does not cache decompressed members: archive['depth'][frame] would
 # inflate the entire sequence on every rendered frame. Keep only needed arrays.
 from arenalab_workbench.depth_store import load as load_depth
 depth=load_depth(scene_path)
 with np.load(scene_path/'cameras.npz') as saved:K=saved['intrinsic'];poses=saved['poses']
 n=playback['frames'];p=np.array(playback['positions']);rot=R.from_quat(np.array(playback['quaternions']).reshape(-1,4)).as_matrix().reshape(n,-1,3,3)
 render=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.65]*3);nodes=[]
 for i in selected:
  mesh=model.geometry[f'geom_{i}'];nodes.append((i,render.add(pyrender.Mesh.from_trimesh(mesh))))
 camera=render.add(pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.015,zfar=10));light=render.add(pyrender.DirectionalLight(color=np.ones(3),intensity=1.7));renderer=pyrender.OffscreenRenderer(960,540);cap=cv2.VideoCapture(meta['head_video']);writer=imageio.get_writer(dest/(rid+'-arm-view-v1.tmp.mp4'),fps=30,codec='libx264',quality=8,macro_block_size=1)
 prompts=json.loads((scene_path/'prompts.json').read_text());human=None;objects=None
 for item in prompts:
  key=item[0]
  mask=np.load(scene_path/'masks'/f'{key}-masks.npz')['masks']>0
  if key.startswith('human-'):human=mask if human is None else human|mask
  else:objects=mask if objects is None else objects|mask
 if human is None:human=np.zeros((len(depth['ids']),540,960),bool)
 # Foreground hardware masks win over generic tool/object proposals.
 distortion_map=recorded_render_map(meta,K,960,540)
 pixels=[]
 for t in range(n):
  seconds=meta['frame_times'][t] if meta.get('frame_times') else meta.get('start_seconds',0)+t/30
  ok,img=read_at(cap,seconds) if meta.get('frame_times') or meta.get('start_seconds',0) else cap.read()
  if not ok:raise ValueError('Source video ends before trajectory')
  img=cv2.resize(img,(960,540));di=min(t//2,len(human)-1)
  for i,node in nodes:
   T=np.eye(4);T[:3,:3]=rot[t,i];T[:3,3]=p[t,i];render.set_pose(node,T)
  view=poses[t]@np.diag([1,-1,-1,1]);render.set_pose(camera,view);render.set_pose(light,view);rgba,z=renderer.render(render,flags=pyrender.RenderFlags.RGBA)
  if distortion_map is not None:
   rgba=cv2.remap(rgba,*distortion_map,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
   z=cv2.remap(z,*distortion_map,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
  observed=cv2.warpPerspective(depth['depth'][di],K@np.linalg.inv(depth['intrinsics'][di]),(960,540));mask=cv2.dilate(human[di].astype('u1'),np.ones((5,5),'u1'));keep=(z>0)&~((observed>.03)&(z>observed+.012)&(mask==0));composite=cv2.cvtColor(inpaint(work,img,mask*255),cv2.COLOR_BGR2RGB);composite[keep]=rgba[keep,:3];writer.append_data(composite);pixels.append(int(keep.sum()))
 writer.close();cap.release();renderer.delete();(dest/(rid+'-arm-view-v1.tmp.mp4')).replace(dest/(rid+'-arm-view-v1.mp4'));from arenalab_workbench.gpu_scheduler import merge_scene
 def update_scene(scene):
  info=json.loads((scene_path/'depth.json').read_text())
  scene.setdefault('patch_depth_policy',{})[rid]=info.get('policy',info.get('model','unknown'))
  scene.setdefault('patch_kinds',{})[rid]='estimated_camera_composite' if (scene_path/'camera-calibration.json').exists() else 'source_camera_composite'
  scene.setdefault('patch_videos',{})[rid]='/assets/'+(dest/(rid+'-arm-view-v1.mp4')).relative_to(str(paths.store)).as_posix()
  scene.get('patch_unavailable',{}).pop(rid,None)
 merge_scene(dest/'scene.json',update_scene);save(folder/'arm-view-validation.json',dict(frames=n,native_arms_and_grippers=True,source_camera_poses_unchanged=True,synthetic_inpainting=True,recorded_color_distortion_applied=distortion_map is not None,minimum_robot_pixels=min(pixels),mean_robot_pixels=float(np.mean(pixels))))
if __name__=='__main__':run(*sys.argv[1:])
