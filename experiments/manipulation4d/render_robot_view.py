"""Composite the selected robot's native arms and Wuji hands into source FOV.

Arm membership follows the two wrist branches of the mechanical tree. No robot
identity selects a rendering offset, motion parameter or camera calibration.
"""
from pathlib import Path
import json,os
os.environ['PYOPENGL_PLATFORM']='egl';os.environ.setdefault('CUDA_VISIBLE_DEVICES','3');os.environ.setdefault('EGL_DEVICE_ID','3')
import cv2,h5py,numpy as np,trimesh,pyrender,mujoco as mj,imageio.v2 as imageio
np.infty=np.inf
from alphamotion.viz.kinematics import visual_mesh_geom_ids
from prepare_objects import depth,OUT

ROOT=Path('/home/ubuntu/arenalab-experiments/manipulation4d');EXP=ROOT.parent/'egodex-threepoint';DEST=Path(os.environ.get('ARENA_EXPORT_DIR','/opt/arenalab/store/manipulation4d/pilot'))
S=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)
manifest=json.loads((DEST/'scene.json').read_text());source=json.loads((EXP/'input_manifest.json').read_text())['source']
with h5py.File(source) as f:camera=f['transforms/camera'][()];intrinsic=f['camera/intrinsic'][()].copy();intrinsic[:2]*=.5
masks={key:np.load(OUT/f'{key}-masks.npz')['masks']>0 for key in ['hand','arm','cup','lid']}
def ancestors(model,body):
    path=[]
    while body>0:path.append(body);body=int(model.body_parentid[body])
    return path+[0]
report={};patches={}
for robot in manifest['robots']:
    identifier=robot['id'];rowfile=next((p for p in [EXP/f'{identifier}-robot.json',ROOT/f'retarget/{identifier}-robot.json'] if p.exists()),None)
    if rowfile is None:raise ValueError(f'Mechanical metadata unavailable for {identifier}')
    row=json.loads(rowfile.read_text())[0];model=mj.MjModel.from_xml_path(row['xml']);ids=list(visual_mesh_geom_ids(model))
    native=[g for g in ids if not any(word in (model.body(int(model.geom_bodyid[g])).name+' '+model.mesh(int(model.geom_dataid[g])).name).lower() for word in ['hand','palm','finger','thumb','dummy'])]
    paths=[ancestors(model,model.body(row['endpoints'][side+'_wrist']).id) for side in ['left','right']]
    common=next(body for body in paths[0] if body in paths[1]);branches=[path[path.index(common)-1] for path in paths]
    arm_bodies={body for body in range(model.nbody) if any(branch in ancestors(model,body) for branch in branches)}
    datafile=next((p for p in [EXP/f'operation-accuracy/palm/{identifier}/{identifier}-render.npz',ROOT/f'retarget/operation-accuracy/palm/{identifier}/robot-render.npz'] if p.exists()),None)
    if datafile is None:raise ValueError(f'Native visual result unavailable for {identifier}')
    with np.load(datafile) as saved:data={key:saved[key] for key in saved.files}
    for i,g in enumerate(native):
        mid=model.geom_dataid[g];a=model.mesh_vertadr[mid];n=model.mesh_vertnum[mid]
        if data[f'v{i}'].shape!=model.mesh_vert[a:a+n].shape or not np.allclose(data[f'v{i}'],model.mesh_vert[a:a+n],atol=1e-6):raise ValueError('Native mesh/trajectory binding differs; refusing an approximate arm binding')
    selected=[i for i,g in enumerate(native) if int(model.geom_bodyid[g]) in arm_bodies]+list(range(len(native),data['positions'].shape[1]))
    if not any(i<len(native) for i in selected):raise ValueError('No native robot arm meshes found')
    positions=data['positions']@S;rotations=S.T@data['rotations'];scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.65,.65,.65]);nodes=[]
    for i in selected:
        color=np.asarray(data[f'c{i}']).ravel();color=color/255 if color.max()>1 else color
        material=pyrender.MetallicRoughnessMaterial(baseColorFactor=color.tolist(),metallicFactor=.08,roughnessFactor=.65)
        nodes.append((i,scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(data[f'v{i}'],data[f'f{i}'],process=False),material=material))))
    viewnode=scene.add(pyrender.IntrinsicsCamera(intrinsic[0,0],intrinsic[1,1],intrinsic[0,2],intrinsic[1,2],znear=.015,zfar=8));lightnode=scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=1.7));renderer=pyrender.OffscreenRenderer(960,540)
    cap=cv2.VideoCapture(str(Path('/opt/arenalab/store')/manifest['video']['url'].removeprefix('/assets/')));filename=f'{identifier}-arm-view-{os.environ.get("ARENA_RENDER_VERSION","v1")}.mp4';temporary=DEST/(filename+'.tmp.mp4');writer=imageio.get_writer(temporary,fps=30,codec='libx264',quality=8,macro_block_size=1);pixels=[]
    for frame in range(len(positions)):
        ok,image=cap.read()
        if not ok:raise ValueError('Source video ended before the trajectory')
        image=cv2.resize(image,(960,540));i=min(frame//2,len(masks['hand'])-1)
        human=(masks['hand'][i]|masks['arm'][i]).astype('u1');human=cv2.dilate(human,np.ones((5,5),np.uint8));human[masks['cup'][i]|masks['lid'][i]]=0
        for g,node in nodes:
            matrix=np.eye(4);matrix[:3,:3]=rotations[frame,g];matrix[:3,3]=positions[frame,g];scene.set_pose(node,matrix)
        view=camera[frame]@np.diag([1,-1,-1,1]);scene.set_pose(viewnode,view);scene.set_pose(lightnode,view)
        rgba,z=renderer.render(scene,flags=pyrender.RenderFlags.RGBA);observed=cv2.warpPerspective(depth['depth'][i],intrinsic@np.linalg.inv(depth['intrinsics'][i]),(960,540))
        keep=(z>0)&~((observed>.03)&(z>observed+.012)&(human==0))
        composed=cv2.cvtColor(cv2.inpaint(image,human*255,4,cv2.INPAINT_TELEA),cv2.COLOR_BGR2RGB);composed[keep]=rgba[keep,:3];writer.append_data(composed);pixels.append(int(keep.sum()))
        if frame in [0,60,120,200]:imageio.imwrite(DEST/f'{identifier}-arm-view-{frame:03d}.jpg',composed)
    writer.close();cap.release();renderer.delete();temporary.replace(DEST/filename)
    patches[identifier]='/assets/manipulation4d/'+DEST.name+'/'+filename
    report[identifier]=dict(frames=len(positions),fps=30,native_arm_meshes=sum(i<len(native) for i in selected),hand_meshes=sum(i>=len(native) for i in selected),arm_branches=[model.body(b).name for b in branches],native_mesh_binding_verified=True,robot_pixels_min=min(pixels),robot_pixels_mean=float(np.mean(pixels)),camera='Unchanged source camera poses and intrinsics',human_removal='SAM 3.1 hands plus arms; foreground object masks retained',depth='Recorded DA3 scene estimate',synthetic_inpainting=True,physical_contact_validated=False)
    print('ARM_VIEW_READY',identifier,json.dumps(report[identifier]),flush=True)
manifest=json.loads((DEST/'scene.json').read_text());manifest['patch_videos']=patches;(DEST/'scene.json').write_text(json.dumps(manifest,separators=(',',':')))
(DEST/'arm-view-validation.json').write_text(json.dumps(report,indent=2))
