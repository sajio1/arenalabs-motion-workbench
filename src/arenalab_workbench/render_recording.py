"""Composite the selected robot's native arms and Wuji hands into source FOV.

Arm membership follows the two wrist branches of the mechanical tree. No robot
identity selects a rendering offset, motion parameter or camera calibration.
"""
from arenalab_workbench.settings import paths
from pathlib import Path
import json,os,sys,subprocess
pinned_python=paths.workbench / 'venv/bin/python'
if Path(sys.executable).absolute()!=pinned_python:os.execv(str(pinned_python),[str(pinned_python),*sys.argv])
os.environ['PYOPENGL_PLATFORM']='egl';os.environ.setdefault('CUDA_VISIBLE_DEVICES','3');os.environ.setdefault('EGL_DEVICE_ID','3')
import cv2,h5py,numpy as np,trimesh,pyrender,mujoco as mj,imageio.v2 as imageio
np.infty=np.inf
from alphamotion.viz.kinematics import visual_mesh_geom_ids
subprocess.run([sys.executable,str(Path(__file__).with_name('audit_recording.py')),sys.argv[1]],check=True)
from arenalab_workbench.fov_cache import read_at, inpaint
from arenalab_workbench.perception_complete import complete_scene
os.environ['ARENA_SCENE_WORK']=str(complete_scene())
with np.load(Path(os.environ['ARENA_SCENE_WORK'])/'depth.npz') as saved:depth={k:saved[k] for k in saved.files}

ROOT=paths.workbench;EXP=Path(os.environ['ARENA_SOURCE_WORK']);DEST=Path(os.environ.get('ARENA_EXPORT_DIR',str(paths.store / 'manipulation4d/pilot')))
S=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)
manifest=json.loads((DEST/'scene.json').read_text());input_manifest=json.loads((EXP/'input_manifest.json').read_text());source=input_manifest['source'];video=input_manifest['video'];start=int((input_manifest.get('source_window') or {}).get('start_frame',0));n=input_manifest['frames']
cap=cv2.VideoCapture(video['path']);intrinsic_scale=np.diag([960/cap.get(cv2.CAP_PROP_FRAME_WIDTH),540/cap.get(cv2.CAP_PROP_FRAME_HEIGHT),1]);cap.release()
with h5py.File(source) as f:camera=f['transforms/camera'][start:start+n];intrinsic=intrinsic_scale@f['camera/intrinsic'][()]
scene_work=Path(os.environ['ARENA_SCENE_WORK']);prompts=json.loads((scene_work/'prompts.json').read_text());human=None;object_mask=None
for item in prompts:
 key,category=item[:2]
 mask=np.load(scene_work/'masks'/f'{key}-masks.npz')['masks']>0
 if key.startswith('human-'):human=mask if human is None else human|mask
 else:object_mask=mask if object_mask is None else object_mask|mask
if object_mask is None:object_mask=np.zeros_like(human)

def ancestors(model,body):
    path=[]
    while body>0:path.append(body);body=int(model.body_parentid[body])
    return path+[0]
report={};patches={}
for robot in [r for r in manifest['robots'] if r['id']==sys.argv[1]]:
    identifier=robot['id'];rowfile=next((p for p in [Path(os.environ.get('ARENA_RETARGET_WORK',str(EXP/'retarget')))/f'{identifier}-robot.json',EXP/f'{identifier}-robot.json',EXP/f'retarget/{identifier}-robot.json'] if p.exists()),None)
    if rowfile is None:raise ValueError(f'Mechanical metadata unavailable for {identifier}')
    row=json.loads(rowfile.read_text())[0];model=mj.MjModel.from_xml_path(row['xml']);ids=list(visual_mesh_geom_ids(model))
    native=[g for g in ids if not any(word in (model.body(int(model.geom_bodyid[g])).name+' '+model.mesh(int(model.geom_dataid[g])).name).lower() for word in ['hand','palm','finger','thumb','dummy'])]
    paths=[ancestors(model,model.body(row['endpoints'][side+'_wrist']).id) for side in ['left','right']]
    common=next(body for body in paths[0] if body in paths[1]);branches=[path[path.index(common)-1] for path in paths]
    arm_bodies={body for body in range(model.nbody) if any(branch in ancestors(model,body) for branch in branches)}
    datafile=Path(os.environ['ARENA_REFINED_OUTPUT'])/'robot-render.npz' if os.environ.get('ARENA_REFINED_OUTPUT') else next((p for p in [Path(os.environ.get('ARENA_RETARGET_WORK',str(EXP/'retarget')))/f'operation-accuracy/palm/{identifier}/robot-render.npz',EXP/f'operation-accuracy/palm/{identifier}/{identifier}-render.npz',EXP/f'retarget/operation-accuracy/palm/{identifier}/robot-render.npz'] if p.exists()),None)
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
    cap=cv2.VideoCapture(video['path']);filename=f'{identifier}-arm-view-{os.environ.get("ARENA_RENDER_VERSION","v1")}.mp4';temporary=DEST/(filename+'.tmp.mp4');writer=imageio.get_writer(temporary,fps=30,codec='libx264',quality=8,macro_block_size=1);pixels=[]
    for frame in range(len(positions)):
        seconds=video['frame_times'][frame] if video.get('frame_times') else video.get('start_seconds',0)+frame/30
        ok,image=read_at(cap,seconds)
        if not ok:raise ValueError('Source video ended before the trajectory')
        image=cv2.resize(image,(960,540));i=min(frame//2,len(human)-1)
        human_frame=cv2.dilate(human[i].astype('u1'),np.ones((5,5),np.uint8));human_frame[object_mask[i]]=0
        for g,node in nodes:
            matrix=np.eye(4);matrix[:3,:3]=rotations[frame,g];matrix[:3,3]=positions[frame,g];scene.set_pose(node,matrix)
        view=camera[frame]@np.diag([1,-1,-1,1]);scene.set_pose(viewnode,view);scene.set_pose(lightnode,view)
        rgba,z=renderer.render(scene,flags=pyrender.RenderFlags.RGBA);observed=cv2.warpPerspective(depth['depth'][i],intrinsic@np.linalg.inv(depth['intrinsics'][i]),(960,540))
        keep=(z>0)&~((observed>.03)&(z>observed+.012)&(human_frame==0))
        composed=cv2.cvtColor(inpaint(EXP,image,human_frame*255),cv2.COLOR_BGR2RGB);composed[keep]=rgba[keep,:3];writer.append_data(composed);pixels.append(int(keep.sum()))
        if frame in [0,60,120,200]:imageio.imwrite(DEST/f'{identifier}-arm-view-{frame:03d}.jpg',composed)
    writer.close();cap.release();renderer.delete();temporary.replace(DEST/filename)
    patches[identifier]='/assets/'+(DEST/filename).relative_to(str(paths.store)).as_posix()
    report[identifier]=dict(frames=len(positions),fps=30,native_arm_meshes=sum(i<len(native) for i in selected),hand_meshes=sum(i>=len(native) for i in selected),arm_branches=[model.body(b).name for b in branches],native_mesh_binding_verified=True,robot_pixels_min=min(pixels),robot_pixels_mean=float(np.mean(pixels)),camera='Unchanged source camera poses and intrinsics',human_removal='SAM 3.1 hands plus arms; foreground object masks retained',depth='Recorded DA3 scene estimate',synthetic_inpainting=True,physical_contact_validated=False)
    print('ARM_VIEW_READY',identifier,json.dumps(report[identifier]),flush=True)
from arenalab_workbench.gpu_scheduler import merge_scene
merge_scene(DEST/'scene.json',lambda manifest:manifest.setdefault('patch_videos',{}).update(patches))
(DEST/'arm-view-validation.json').write_text(json.dumps(report,indent=2))
