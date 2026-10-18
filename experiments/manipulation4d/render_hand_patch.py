"""Source-camera robot-hand composite, with explicit estimated-depth/inpaint provenance."""
from pathlib import Path
import os, json
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ.setdefault('CUDA_VISIBLE_DEVICES','3');os.environ.setdefault('EGL_DEVICE_ID','3')
import cv2, h5py, numpy as np, trimesh, pyrender, imageio.v2 as imageio, mujoco as mj
np.infty=np.inf
from alphamotion.viz.kinematics import visual_mesh_geom_ids

ROOT=Path('/home/ubuntu/arenalab-experiments/manipulation4d')
PILOT=Path('/opt/arenalab/store/manipulation4d/pilot')
EXP=Path('/home/ubuntu/arenalab-experiments/egodex-threepoint')
S=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)
source=json.loads((EXP/'input_manifest.json').read_text())['source']
with h5py.File(source) as f:camera=f['transforms/camera'][()];intrinsic=f['camera/intrinsic'][()]*.5;intrinsic[2,2]=1
from prepare_objects import depth
mask=np.load(ROOT/'cup-lids/hand-masks.npz')['masks']>0
hand_geoms=sum(len(visual_mesh_geom_ids(mj.MjModel.from_xml_path(str(EXP/f'wuji/urdf/{side}.urdf')))) for side in ['left','right'])
patches={};report={}
for robot in ['h2','a3']:
    with np.load(EXP/f'operation-accuracy/palm/{robot}/{robot}-render.npz') as f:data={k:f[k] for k in f.files}
    positions=data['positions']@S;rotations=S.T@data['rotations'];n=len(positions);begin=positions.shape[1]-hand_geoms
    scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.65,.65,.65]);nodes=[]
    material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.60,.74,.66,1],metallicFactor=.08,roughnessFactor=.65)
    for g in range(begin,positions.shape[1]):nodes.append((g,scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(data[f'v{g}'],data[f'f{g}'],process=False),material=material))))
    node_camera=scene.add(pyrender.IntrinsicsCamera(intrinsic[0,0],intrinsic[1,1],intrinsic[0,2],intrinsic[1,2],znear=.015,zfar=8))
    node_light=scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=1.7));renderer=pyrender.OffscreenRenderer(960,540)
    cap=cv2.VideoCapture(str(PILOT/'original.mp4'));writer=imageio.get_writer(PILOT/f'{robot}-hand-patch.mp4',fps=30,codec='libx264',quality=8,macro_block_size=1);visible=[];filled=[]
    for frame in range(n):
        ok,image=cap.read()
        if not ok:raise ValueError('Source video shorter than trajectory')
        image=cv2.resize(image,(960,540));i=min(frame//2,len(mask)-1);human=mask[i].astype(np.uint8)
        for g,node in nodes:
            transform=np.eye(4);transform[:3,:3]=rotations[frame,g];transform[:3,3]=positions[frame,g];scene.set_pose(node,transform)
        view=camera[frame]@np.diag([1,-1,-1,1]);scene.set_pose(node_camera,view);scene.set_pose(node_light,view)
        rgb,robot_depth=renderer.render(scene,flags=pyrender.RenderFlags.RGBA)
        observed_depth=cv2.warpPerspective(depth['depth'][i],intrinsic@np.linalg.inv(depth['intrinsics'][i]),(960,540))
        inpaint_mask=cv2.dilate(human,np.ones((5,5),np.uint8))*255
        background=cv2.inpaint(image,inpaint_mask,4,cv2.INPAINT_TELEA)
        occluded=(observed_depth>.03)&(robot_depth>observed_depth+.012)&(inpaint_mask==0)
        keep=(robot_depth>0)&~occluded
        composed=cv2.cvtColor(background,cv2.COLOR_BGR2RGB);composed[keep]=rgb[keep,:3]
        cv2.rectangle(composed,(0,0),(960,29),(10,23,16),-1)
        cv2.putText(composed,f'{robot.upper()} + Wuji | synthetic robot-hand view',(12,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(205,232,190),1)
        writer.append_data(composed);visible.append(int(keep.sum()));filled.append(int(((inpaint_mask>0)&~keep).sum()))
        if frame in [0,60,120,200]:imageio.imwrite(PILOT/f'{robot}-patch-{frame:03d}.jpg',composed)
    writer.close();cap.release();renderer.delete()
    patches[robot]=f'/assets/manipulation4d/pilot/{robot}-hand-patch.mp4'
    report[robot]=dict(frames=n,fps=30,camera='original EgoDex intrinsics and camera poses; no invented camera-to-eye translation',segmentation='SAM 3.1 hand masks at source-matched 15Hz',depth='DA3 estimated depth; human mask excluded from occluders',gap_fill='OpenCV Telea inpainting; synthetic content, not measured background',visible_robot_pixels=dict(mean=float(np.mean(visible)),minimum=min(visible)),uncovered_inpaint_pixels_mean=float(np.mean(filled)),physical_contact_validated=False)
    print('HAND_PATCH_COMPLETE',robot,flush=True)
scene=json.loads((PILOT/'scene.json').read_text());scene['patch_videos']=patches;(PILOT/'scene.json').write_text(json.dumps(scene,separators=(',',':')))
(PILOT/'patch-validation.json').write_text(json.dumps(report,indent=2))
