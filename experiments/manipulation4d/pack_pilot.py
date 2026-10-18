"""Export observed 4D scene and palm-referenced robot results in source world metres."""
from pathlib import Path
import json, shutil
import cv2, numpy as np, trimesh
from scipy.spatial.transform import Rotation
from alphamotion.engine.descriptor import build_from_mjcf
from prepare_objects import mask_at, world_points, depth, frame_ids, OUT as INPUT, depth_calibration

ROOT=Path('/home/ubuntu/arenalab-experiments/manipulation4d')
EXPERIMENT=Path('/home/ubuntu/arenalab-experiments/egodex-threepoint')
DEST=Path('/opt/arenalab/store/manipulation4d/pilot');DEST.mkdir(parents=True,exist_ok=True)
URL='/assets/manipulation4d/pilot/'
S=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)  # original Y-up -> intermediate Z-up
write=lambda p,d:Path(p).write_text(json.dumps(d,separators=(',',':'),allow_nan=False))
selected=json.loads((INPUT/'selected-objects.json').read_text())
existing=json.loads((DEST/'scene.json').read_text()) if (DEST/'scene.json').exists() else {}
tracks={}
for row in selected:
    path=DEST/f'{row["kind"]}-{row["instance_id"]}-track.json'
    if path.exists():tracks[row['kind']]=json.loads(path.read_text())
labels={k:np.load(INPUT/f'{k}-masks.npz')['masks'] for k in ['cup','lid','hand']}
if (INPUT/'arm-masks.npz').exists():labels['arm']=np.load(INPUT/'arm-masks.npz')['masks']
cloud_frames=[];bounds=[]
with (DEST/'cloud.bin').open('wb') as stream:
    def block(array):
        stream.write(b'\0'*((-stream.tell())%4));offset=stream.tell();stream.write(array.tobytes());return offset
    for i,source_id in enumerate(frame_ids):
        cloud=world_points(i);rgb=depth['images'][i]
        masks={k:mask_at(v,i) for k,v in labels.items()}
        semantic=np.zeros(cloud.shape[:2],np.uint8)
        semantic[masks['cup']>0]=3;semantic[masks['lid']>0]=4
        for slot,row in enumerate(selected,1):
            object_id=tracks[row['kind']][i]['segmentation_instance_id'] if row['kind'] in tracks else row['instance_id']
            semantic[masks[row['kind']]==object_id]=slot
        human=masks['hand']>0
        if 'arm' in masks:human|=masks['arm']>0
        human&=~((masks['cup']>0)|(masks['lid']>0))
        visible=(depth['depth'][i]>.05)&(depth['depth'][i]<5)&np.isfinite(cloud).all(-1)&~human
        sample=np.zeros_like(visible);sample[::2,::2]=True;visible&=sample
        xyz=cloud[visible].astype('<f4');colors=rgb[visible].astype('u1');semantic=semantic[visible]
        cloud_frames.append(dict(source_frame=int(source_id),time=float(source_id/30),count=len(xyz),positions=block(xyz),colors=block(colors),labels=block(semantic)))
        bounds.extend([xyz.min(0).tolist(),xyz.max(0).tolist()])
shutil.copyfile('/home/ubuntu/arenalab-experiments/da3/no-hands/original.mp4',DEST/'original.mp4')
source=np.load(EXPERIMENT/'source_hands.npz');source_points=source['points']@S
write(DEST/'source-hands.json',dict(points=source_points.astype('float32').tolist(),confidence=source['confidence'].astype('float32').tolist(),fps=30))
robots=[]
for label in ['h2','a3']:
    folder=EXPERIMENT/f'operation-accuracy/palm/{label}';saved=np.load(folder/f'{label}-render.npz')
    with np.load(folder/'motion.npz') as m:motion={k:m[k] for k in m.files}
    scene=trimesh.Scene();geoms=saved['positions'].shape[1]
    for g in range(geoms):
        mesh=trimesh.Trimesh(saved[f'v{g}'],saved[f'f{g}'],process=False)
        color=np.asarray(saved[f'c{g}']).ravel();color=color*255 if color.max()<=1 else color
        mesh.visual.vertex_colors=np.tile(color.astype(np.uint8),(len(mesh.vertices),1))
        scene.add_geometry(mesh,node_name=f'geom_{g}',geom_name=f'geom_{g}')
    scene.export(DEST/f'{label}-wuji.glb')
    positions=saved['positions']@S
    rotations=S.T@saved['rotations']
    quaternions=Rotation.from_matrix(rotations.reshape(-1,3,3)).as_quat().reshape(len(positions),geoms,4)
    op=np.load(folder/'operation-points.npz');actual=op['actual'][:,:,0]@S;targets=op['source'][:,:,0]@S
    error=np.linalg.norm(actual-targets,axis=-1)*1000
    row=json.loads((EXPERIMENT/f'{label}-robot.json').read_text())[0]
    spec,_,_,bindings,_=build_from_mjcf(row['xml'],row['body'])
    if list(spec.joint_names)!=list(motion['joint_names']):raise ValueError('Native joint binding mismatch')
    q=motion['q'];joint_names=[];joint_values=[]
    for j,names in enumerate(bindings):
        for axis,name in enumerate(names):joint_names.append(name);joint_values.append(q[:,j,axis])
    fingers=np.load(folder/'wuji_angles.npz')['angles']
    for si,side in enumerate(['left','right']):
        for finger in range(5):
            for joint in range(4):joint_names.append(f'{side}_finger{finger+1}_joint{joint+1}');joint_values.append(fingers[:,si,finger*4+joint])
    joint_values=np.stack(joint_values,1)
    payload=dict(fps=30,frames=len(positions),positions=positions.astype('float32').tolist(),quaternions=quaternions.astype('float32').tolist(),tracking=dict(actual=actual.tolist(),target=targets.tolist(),error_mm=error.tolist(),orientation_deg=motion['wrist_error_deg'].tolist(),reference='Four proximal knuckle center + palm frame'),joints=dict(names=joint_names,values=joint_values.tolist(),unit='rad',representation='Native robot and Wuji joint coordinates; kinematic output'))
    payload['tracking']['fingertip_error_mm']=(np.linalg.norm(op['actual'][:,:,1:6]-op['source'][:,:,1:6],axis=-1)*1000).tolist()
    write(DEST/f'{label}-wuji.json',payload)
    robots.append(dict(id=label,label='H2' if label=='h2' else 'A3',hand='wuji',model_url=URL+f'{label}-wuji.glb',playback_url=URL+f'{label}-wuji.json',reference='palm',quality=dict(mean_mm=float(error.mean()),max_mm=float(error.max()))))
manifest=dict(id='egodex_add_remove_lid_1',title='add_remove_lid/1.hdf5',dataset='EgoDex',fps=30,frames=len(source_points),duration=(len(source_points)-1)/30,world='Original EgoDex world; Y up; metres',video=dict(url=URL+'original.mp4',start_seconds=0),cloud=dict(url=URL+'cloud.bin',fps=15,frames=cloud_frames,format='float32 XYZ, uint8 RGB, uint8 semantic label; offsets bytes'),source_hands_url=URL+'source-hands.json',objects=[],robots=robots,semantic=dict(task='Add and remove lids from paper cups',objects=['paper cup','plastic lid'],provenance='EgoDex source annotations; not a new VLM inference'),tactile=dict(status='unavailable',reason='No tactile sensor stream; visual contact estimator not yet run'),wrist_videos=[],geometry=dict(depth='DA3-Large estimate',segmentation='SAM 3.1 Object Multiplex',hidden_surfaces='SAM 3D generated hypotheses where available',ground_truth=False),bounds=[np.min(bounds,axis=0).tolist(),np.max(bounds,axis=0).tolist()])
exported_ids={row['id'] for row in robots}
manifest['robots'].extend(row for row in existing.get('robots',[]) if row['id'] not in exported_ids)
if depth_calibration:
    manifest['geometry']['depth_scale_calibration']=depth_calibration
for key in ['objects','patch_videos','semantic','tactile','spatial_memory','source_camera','hand_motion_audit_url']:
    if key in existing:manifest[key]=existing[key]
write(DEST/'scene.json',manifest)
print('PILOT_PACKED',len(cloud_frames),robots)
