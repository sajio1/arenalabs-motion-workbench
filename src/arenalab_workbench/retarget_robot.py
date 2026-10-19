"""One retargeting entry for any ingested upright humanoid and fixed hand metadata.

Each recording supplies its own immutable body prior and observed source hands.
Explicit work/output paths are mandatory; another recording is never a fallback. No robot identity selects solver settings or mounting behavior.
"""
from arenalab_workbench.settings import paths
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
from arenalab_workbench.refinement_runtime import run_refinement
import numpy as np,trimesh
from scipy.spatial.transform import Rotation
from alphamotion.engine.descriptor import build_from_mjcf
from greenwich_pipelines.robot_semantics import resolve_endpoints,ROLES
from arenalab_workbench.end_effectors import hand_angles,export_wuji

ROOT=paths.workbench
SOURCE=Path(os.environ['ARENA_SOURCE_WORK'])
WORK=Path(os.environ['ARENA_RETARGET_WORK'])
DEST=Path(os.environ['ARENA_EXPORT_DIR'])
PUBLIC='/assets/'+DEST.relative_to(str(paths.store)).as_posix()
S=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,separators=(',',':'),allow_nan=False));temporary.replace(path)

def progress(path,phase):
    if path:
        state=json.loads(Path(path).read_text()) if Path(path).exists() else {}
        write(path,{**state,'status':'running','phase':phase})

def run(robot_id,status_path=None):
    cfg=json.loads(paths.viewer_config.read_text());rows={r['id']:r for r in cfg['robots']}
    registry=paths.store / 'uploaded/registry.json'
    if registry.exists():rows.update({r['id']:r for r in json.loads(registry.read_text())})
    robot=rows[robot_id];row=robot['pipeline_row'].copy()
    if 'palm_frames' not in row:raise ValueError('Physical palm axes are required before humanoid hand retargeting')
    spec,*_=build_from_mjcf(row['xml'],row['body']);ids=resolve_endpoints(spec,row.get('endpoints'))
    row['endpoints']={role:spec.joint_names[i] for role,i in zip(ROLES,ids)}
    no_hand=json.loads((SOURCE/'input_manifest.json').read_text()).get('end_effector')=='none'
    source_hashes={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ['canonical.npz','smplx_motion.npz','observations.npz','source_hands.npz','operation-accuracy/source-targets.npz'] if (SOURCE/name).exists()}
    WORK.mkdir(parents=True,exist_ok=True);(WORK/'operation-accuracy').mkdir(exist_ok=True)
    for name in ['canonical.npz','smplx_motion.npz','observations.npz','source_hands.npz','sources.json']:
        destination=WORK/name
        if (SOURCE/name).exists() and not destination.exists():destination.symlink_to(SOURCE/name)
    destination=WORK/'operation-accuracy/source-targets.npz'
    if not destination.exists():destination.symlink_to(SOURCE/'operation-accuracy/source-targets.npz')
    write(WORK/f'{robot_id}-robot.json',[row]);job=dict(schema='greenwich.job.v1',id='workbench_'+robot_id,pipeline='fullbody',source='sources.json',robots=f'{robot_id}-robot.json',output=f'{robot_id}-result',runtime=str(paths.config / 'runtime.cloud.json'),fps=30)
    previous=WORK/f'{robot_id}-result'
    if previous.exists():previous.rename(WORK/f'{robot_id}-result.previous-{time.time_ns()}')
    write(WORK/f'{robot_id}-job.json',job)
    environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','3'),EGL_DEVICE_ID=os.environ.get('EGL_DEVICE_ID','3'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',PYTHONPATH=paths.pythonpath(),ARENA_EGO_WORKDIR=str(WORK),ARENA_OPERATION_MODE='tcp' if no_hand else 'palm')
    progress(status_path,'AlphaMotion body retargeting')
    environment['ARENA_FRAME_CACHE_DIR']=str(SOURCE/'frame-stream'/robot_id)
    with (WORK/f'{robot_id}-model.log').open('w') as log:
        subprocess.run([sys.executable,'-m','greenwich_pipelines.cli','run',str(WORK/f'{robot_id}-job.json')],cwd=WORK,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    raw=list((WORK/f'{robot_id}-result').rglob('prime_original.npz'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in raw}
    motion=next((WORK/f'{robot_id}-result').rglob('motion.npz'))
    angles=None if no_hand else hand_angles(WORK/'source_hands.npz')
    local=export_wuji(row,motion,angles,SOURCE/'wuji',WORK/'initial-visuals'/robot_id)
    np.savez_compressed(WORK/f'operation-accuracy/{robot_id}-local-points.npz',points=local)
    output=WORK/'operation-accuracy/palm'/robot_id;environment['ARENA_OUTPUTDIR']=str(output)
    progress(status_path,'Palm-reference body and arm refinement')
    with (WORK/f'{robot_id}-refine.log').open('w') as log:
        run_refinement([sys.executable,str(Path(__file__).with_name('refine_operation.py')),robot_id],env=environment,cwd=WORK,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
    return export_result(robot_id,robot,row,output,angles,hashes,source_hashes,status_path)

def export_result(robot_id,robot,row,output,angles,hashes,source_hashes,status_path=None):
    local=export_wuji(row,output/'motion.npz',angles,SOURCE/'wuji',output)
    progress(status_path,'Exporting native geometry and joint outputs')
    with np.load(output/'robot-render.npz') as saved:visual={k:saved[k] for k in saved.files}
    with np.load(output/'motion.npz') as saved:motion={k:saved[k] for k in saved.files}
    folder=DEST/('robot-'+robot_id);folder.mkdir(exist_ok=True);scene=trimesh.Scene();count=visual['positions'].shape[1]
    for i in range(count):
        mesh=trimesh.Trimesh(visual[f'v{i}'],visual[f'f{i}'],process=False);color=np.asarray(visual[f'c{i}']).ravel();color=color*255 if color.max()<=1 else color;mesh.visual.vertex_colors=np.tile(color.astype(np.uint8),(len(mesh.vertices),1));scene.add_geometry(mesh,node_name=f'geom_{i}',geom_name=f'geom_{i}')
    scene.export(folder/'model.glb')
    spec,_,_,bindings,_=build_from_mjcf(row['xml'],row['body']);wrist_ids=[spec.joint_names.index(row['endpoints'][side+'_wrist']) for side in ['left','right']]
    from alphamotion.engine.nets.rotations import rot6d_to_matrix
    import torch
    C=np.array([[0,1,0],[0,0,1],[1,0,0]],float)
    R=rot6d_to_matrix(torch.tensor(motion['rot6d'])).numpy()[:,wrist_ids]
    actual=(motion['world_position_cm'][:,wrist_ids]/100+np.einsum('tsij,tsj->tsi',R,local[:,:,0]@C.T))@C@S
    target=np.load(WORK/'operation-accuracy/source-targets.npz')['points'][:,:,0]@S
    error=np.linalg.norm(actual-target,axis=-1)*1000
    positions=visual['positions']@S;rotations=S.T@visual['rotations'];n=len(positions)
    joint_names=[];joint_values=[]
    for j,names in enumerate(bindings):
        for axis,name in enumerate(names):joint_names.append(name);joint_values.append(motion['q'][:,j,axis])
    for si,side in enumerate([] if angles is None else ['left','right']):
        for finger in range(5):
            for joint in range(4):joint_names.append(f'{side}_finger{finger+1}_joint{joint+1}');joint_values.append(angles[:,si,finger*4+joint])
    payload=dict(fps=30,frames=n,positions=positions.tolist(),quaternions=Rotation.from_matrix(rotations.reshape(-1,3,3)).as_quat().reshape(n,count,4).tolist(),tracking=dict(actual=actual.tolist(),target=target.tolist(),error_mm=error.tolist(),orientation_deg=motion['wrist_error_deg'].tolist(),reference='Observed wrist reference' if angles is None else 'Four proximal knuckle center + palm frame'),joints=dict(names=joint_names,values=np.stack(joint_values,1).tolist(),unit='rad',representation='Native robot joint coordinates; no dexterous hand' if angles is None else 'Native robot and Wuji joint coordinates; kinematic output'))
    all_actual=(motion['world_position_cm'][:,wrist_ids,None,:]/100+np.einsum('tsij,tskj->tski',R,local@C.T))@C@S
    all_targets=np.load(WORK/'operation-accuracy/source-targets.npz')['points']@S
    if angles is not None:payload['tracking']['fingertip_error_mm']=(np.linalg.norm(all_actual[:,:,1:6]-all_targets[:,:,1:6],axis=-1)*1000).tolist()
    write(folder/'playback.json',payload)
    result=dict(encoder_policy=__import__('greenwich_pipelines.input_policy',fromlist=['POLICY']).POLICY,id=robot_id,label=robot['label'],hand='none' if angles is None else 'wuji',model_url=f'{PUBLIC}/robot-{robot_id}/model.glb',playback_url=f'{PUBLIC}/robot-{robot_id}/playback.json',reference='wrist' if angles is None else 'palm',quality=dict(mean_mm=float(error.mean()),max_mm=float(error.max()),orientation_max_deg=float(motion['wrist_error_deg'].max()),minimum_sole_height_mm=float(motion['sole_surface_height_cm'].min()*10)),mounting=json.loads((output/'hand-mounts.json').read_text()))
    from arenalab_workbench.gpu_scheduler import merge_scene
    def update(current):
        current['robots']=[r for r in current['robots'] if r['id']!=robot_id]+[result]
        if 'ground_height_m' in motion:
            from arenalab_workbench.support_reference import POLICY
            current['geometry']['ground']=dict(height_m=float(motion['ground_height_m']),estimated=True,policy=POLICY,provenance='User-declared floor' if json.loads((SOURCE/'input_manifest.json').read_text()).get('height_calibration') else 'SMPL-X reconstructed low/slow soles; not a measured floor',height_calibration=json.loads((SOURCE/'input_manifest.json').read_text()).get('height_calibration'))
    merge_scene(DEST/'scene.json',update)
    assert all(hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==digest for name,digest in source_hashes.items()),'Source targets or priors changed'
    write(folder/'validation.json',dict(source_id=json.loads((SOURCE/'input_manifest.json').read_text())['clip'],source_hashes=source_hashes,original_priors_unchanged=True,raw_sha256=hashes,quality=result['quality'],policy='same palm-reference refinement for every humanoid',hardware_calibration=False,encoder_policy=__import__('greenwich_pipelines.input_policy',fromlist=['POLICY']).POLICY,support_policy=__import__('arenalab_workbench.support_reference', fromlist=['*']).POLICY))
    if status_path:write(status_path,dict(status='completed',phase='Ready',scene_url=PUBLIC+'/scene.json',robot_id=robot_id))
    print('RETARGET_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('robot_id');parser.add_argument('--status',type=Path);args=parser.parse_args();run(args.robot_id,args.status)
