"""One retargeting entry for any ingested upright humanoid and fixed hand metadata.

Each recording supplies its own immutable body prior and observed source hands.
Explicit work/output paths are mandatory; another recording is never a fallback. No robot identity selects solver settings or mounting behavior.
"""
from arenalab_workbench.settings import paths
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
import numpy as np,trimesh
from scipy.spatial.transform import Rotation
from alphamotion.engine.descriptor import build_from_mjcf
from greenwich_pipelines.robot_semantics import resolve_endpoints,ROLES
from arenalab_workbench.end_effectors import hand_angles,export_wuji,operation_local_points
from arenalab_workbench.motion_backend import selected_backend,backend_fingerprint,build_request,run_gpu_motion,source_clock,SOURCE_FILES,prepare_gpu_version

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
    global WORK,SOURCE
    if selected_backend()=='gpu_hybrid' and os.environ.get('ARENA_GPU_WORK_ROOT'):
        WORK=Path(os.environ['ARENA_GPU_WORK_ROOT'])
    cfg=json.loads(paths.viewer_config.read_text());rows={r['id']:r for r in cfg['robots']}
    registry=paths.store / 'uploaded/registry.json'
    if registry.exists():rows.update({r['id']:r for r in json.loads(registry.read_text())})
    robot=rows[robot_id];row=robot['pipeline_row'].copy()
    from arenalab_motion.robot_metadata import ensure_robot_metadata
    row=ensure_robot_metadata(row)
    if 'palm_frames' not in row:raise ValueError('Physical palm axes are required before humanoid hand retargeting')
    spec,*_=build_from_mjcf(row['xml'],row['body']);ids=resolve_endpoints(spec,row.get('endpoints'))
    row['endpoints']={role:spec.joint_names[i] for role,i in zip(ROLES,ids)}
    from arenalab_workbench.motion_backend import normalized_source
    SOURCE=normalized_source(SOURCE)
    no_hand=json.loads((SOURCE/'input_manifest.json').read_text()).get('end_effector','none')=='none'
    source_hashes={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in SOURCE_FILES if (SOURCE/name).exists()}
    WORK.mkdir(parents=True,exist_ok=True);(WORK/'operation-accuracy').mkdir(exist_ok=True)
    for name in ['canonical.npz','smplh_motion.npz','human_reference.json','native_source.npz','native_source.json','observations.npz','source_hands.npz','sources.json','six-point-reference.npz','source-descriptor.npz']:
        destination=WORK/name
        if (SOURCE/name).exists() and not destination.exists():destination.symlink_to(SOURCE/name)
    destination=WORK/'operation-accuracy/source-targets.npz'
    if not destination.exists():destination.symlink_to(SOURCE/'operation-accuracy/source-targets.npz')
    if selected_backend()=='gpu_hybrid':
        # Branch before the legacy fullbody 100+100 stages. The new runner owns
        # immutable neural prior generation as well as whole-body refinement.
        work_root=WORK
        angles=None if no_hand else hand_angles(WORK/'source_hands.npz')
        frames=len(source_clock(SOURCE)[0])
        local=operation_local_points(row,angles,SOURCE/'wuji',frames)
        WORK,request,cached=prepare_gpu_version(work_root,robot_id,row,SOURCE,'tcp' if no_hand else 'palm',local)
        os.environ['ARENA_GPU_WORK_ROOT']=str(work_root)
        output=Path(request['output_dir'])
        if not (WORK/f'{robot_id}-robot.json').exists():write(WORK/f'{robot_id}-robot.json',[row])
        target=WORK/'operation-accuracy/source-targets.npz';target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.symlink_to(SOURCE/'operation-accuracy/source-targets.npz')
        progress(status_path,'GPU skeleton, surface and wrist motion refinement')
        if cached is not None:_,hashes=cached
        else:
            with (WORK/f'{robot_id}-gpu-motion.log').open('w') as log:
                _,hashes=run_gpu_motion(request,env=os.environ.copy(),cwd=WORK,stdout=log,stderr=subprocess.STDOUT)
        return export_result(robot_id,robot,row,output,angles,hashes,source_hashes,status_path,
                             reuse_local=local if cached is not None else None)
    raise RuntimeError('The unified GPU backend is required')

def export_result(robot_id,robot,row,output,angles,hashes,source_hashes,status_path=None,reuse_local=None):
    backend=selected_backend()
    numerical_report=json.loads((output/'report.json').read_text()) if (output/'report.json').exists() else {}
    encoder_policy=(numerical_report.get('encoder_policy','gpu_hybrid: see numerical runner report') if backend=='gpu_hybrid' else __import__('greenwich_pipelines.input_policy',fromlist=['POLICY']).POLICY)
    support_policy=(numerical_report.get('support_policy','gpu_hybrid: see numerical runner report') if backend=='gpu_hybrid' else __import__('arenalab_workbench.support_reference',fromlist=['POLICY']).POLICY)
    native_qpos=output/'gpu_qpos.npz' if backend=='gpu_hybrid' else None
    if native_qpos is not None and not native_qpos.is_file():raise FileNotFoundError('GPU export requires native world qpos for independent mesh/FK parity: '+str(native_qpos))
    if reuse_local is not None and (output/'robot-render.npz').is_file() and (output/'hand-mounts.json').is_file():
        local=reuse_local
    else:local=export_wuji(row,output/'motion.npz',angles,SOURCE/'wuji',output,native_qpos_path=native_qpos)
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
    target_path=output/'source-targets.npz' if (output/'source-targets.npz').is_file() else WORK/'operation-accuracy/source-targets.npz'
    target=np.load(target_path)['points'][:,:,0]@S
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
    all_targets=np.load(target_path)['points']@S
    if angles is not None:payload['tracking']['fingertip_error_mm']=(np.linalg.norm(all_actual[:,:,1:6]-all_targets[:,:,1:6],axis=-1)*1000).tolist()
    write(folder/'playback.json',payload)
    native_parity=None
    if native_qpos is not None:
        from arenalab_workbench.native_export_parity import audit_native_export
        native_parity=audit_native_export(row,native_qpos,folder/'playback.json',output/'robot-render.npz',local)
        write(folder/'native-export-parity.json',native_parity)
    result=dict(trajectory_scale=numerical_report.get('trajectory_scale','preserve'),pipeline_version='resident-skeleton-surface-wrist-v3',motion_policy_version='resident-skeleton-surface-wrist-v3',source_policy=numerical_report.get('source_policy'),source_policy_version='unified-provided-body-or-three-point-native-h-v1',solver=numerical_report.get('solver',{}),routing=numerical_report.get('source_policy'),encoder_policy=encoder_policy,motion_backend=backend,id=robot_id,label=robot['label'],hand='none' if angles is None else 'wuji',model_url=f'{PUBLIC}/robot-{robot_id}/model.glb',playback_url=f'{PUBLIC}/robot-{robot_id}/playback.json',reference='wrist' if angles is None else 'palm',quality=dict(mean_mm=float(error.mean()),p95_mm=float(np.quantile(error,.95)),p99_mm=float(np.quantile(error,.99)),max_mm=float(error.max()),orientation_max_deg=float(motion['wrist_error_deg'].max()),minimum_sole_height_mm=float(motion['sole_surface_height_cm'].min()*10)),mounting=json.loads((output/'hand-mounts.json').read_text()))
    from arenalab_workbench.gpu_scheduler import merge_scene
    source_manifest=json.loads((SOURCE/'input_manifest.json').read_text())
    ground_provenance=numerical_report.get('ground_provenance') or (
        'Source-derived fixed support plane from observed sole tracks; not a measured scene floor'
        if source_manifest.get('source_kind')=='direct_six' else 'SMPL reconstructed support; not a measured floor')
    from arenalab_workbench.source_preview import export_source_preview
    from arenalab_workbench.source_display import source_scene_metadata
    source_metadata=source_scene_metadata(export_source_preview(SOURCE,DEST),numerical_report,S.T)
    def update(current):
        current.update(source_metadata)
        if numerical_report.get('trajectory_scale')!='match_robot':current.pop('source_transform',None)
        current['robots']=[r for r in current['robots'] if r['id']!=robot_id]+[result]
        if 'ground_height_m' in motion:
            current['geometry']['ground']=dict(height_m=float(motion['ground_height_m']),estimated=True,policy=support_policy,provenance='User-declared floor' if source_manifest.get('height_calibration') else ground_provenance,height_calibration=source_manifest.get('height_calibration'))
    merge_scene(DEST/'scene.json',update)
    assert all(hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==digest for name,digest in source_hashes.items()),'Source targets or priors changed'
    native_work=output.parents[2] if output.parent.name=='palm' and output.parent.parent.name=='operation-accuracy' else WORK
    write(folder/'validation.json',dict(source_id=json.loads((SOURCE/'input_manifest.json').read_text())['clip'],source_hashes=source_hashes,original_priors_unchanged=True,raw_sha256=hashes,quality=result['quality'],policy='same selected motion backend for every humanoid',hardware_calibration=False,encoder_policy=encoder_policy,support_policy=support_policy,motion_backend=backend,motion_backend_fingerprint=numerical_report.get('motion_backend_fingerprint',backend_fingerprint()),trajectory_scale=numerical_report.get('trajectory_scale','preserve'),solver=numerical_report.get('solver',{}),scale_transform=numerical_report.get('scale_transform'),native_export_parity=native_parity,native_work_dir=str(native_work.resolve()),native_work_root=os.environ.get('ARENA_GPU_WORK_ROOT',str(WORK.resolve())),native_output_dir=str(output.resolve()),source_work=str(SOURCE.resolve())))
    if status_path:write(status_path,dict(status='completed',phase='Ready',scene_url=PUBLIC+'/scene.json',robot_id=robot_id))
    print('RETARGET_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('robot_id');parser.add_argument('--status',type=Path);args=parser.parse_args();run(args.robot_id,args.status)
