"""Installed private full-body motion backend: unchanged pre-tolerance v3 solver.

This executable performs no network operation. All supplied source body joints
are encoded; source human reconstruction is never replaced by model completion.
All robot rows use the same numerical settings. Optional TCP offsets
remain local and do not modify independently generated finger motion.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from .hybrid_ops import HybridOps
from .solver import FullMotion
from .runtime import Runtime
from .sources import load_source, source_directory
from .prior import infer_prior,native_to_motion
from .geometry import UniversalOracle
from .robot_metadata import ensure_robot_metadata

C=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
POLICY=dict(version='private-fullbody-hybrid-v3',soft_iterations=60,
            polish_iterations=20,max_qp_iterations=96,cooldown=.6,
            hand_budget_floor_m=.003,hand_budget_margin_m=.0005,
            hand_position_weight=600.,root_orientation='measured source pelvis; immutable neural raw retained',
            surface='local semantic mesh shape prior; no learned UMR correspondence',
            qp_freeze='identical to external acceptance thresholds; no relaxation',
            line_search='nonlinear floor restoration inside cost-tested candidate, within root trust; no output repair',
            task_order=['root','chest','left_wrist','right_wrist','left_ankle','right_ankle'])
POLICY['source_scaling']='Explicit preserve or match_robot; match_robot uses the unchanged eight-segment anchored similarity'
POLICY['task_scope']='Kinematic retargeting; match_robot changes the source/object task dimensions together'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def saved_json(path,value):
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,(float,np.floating)) and not np.isfinite(x):return None
        return x
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')

def assert_inputs(request):
    for name,digest in request['source_sha256'].items():
        if sha(Path(request['source_work'])/name)!=digest:
            raise RuntimeError('Immutable source SHA changed: '+name)
    if request.get('local_point_file') and sha(request['local_point_file'])!=request['local_point_sha256']:
        raise RuntimeError('Local TCP data changed')

def points_from_request(request,source):
    """Keep original operation targets and physical local offsets separate."""
    T=request['frames'];local=np.zeros((T,6,3));target=np.array(source['obs_wrist_zup_m'],copy=True)
    file=request.get('local_point_file')
    if file:
        with np.load(file) as z:
            local[:,2:4]=z['points'][:,:,0]
        targets=Path(request['source_work'])/'operation-accuracy/source-targets.npz'
        if targets.is_file():
            with np.load(targets) as z:target=np.array(z['points'][:,:,0],copy=True)
        elif np.any(local):
            raise ValueError('Nonzero local TCP requires matching original operation targets')
    return target,local

def normalize_request(request):
    request=dict(request)
    schema=request.get('schema','arenalabs.private-motion-request.v1')
    if schema not in {'arenalabs.private-motion-request.v1','arenalabs.motion.v1'}:raise ValueError('Unsupported motion request schema: '+str(schema))
    if 'row' not in request:request['row']=request['robot']
    request['row']=dict(request['row'].get('pipeline_row',request['row']))
    if 'xml' not in request['row'] and request['row'].get('model'):request['row']['xml']=request['row']['model']
    request['row']=ensure_robot_metadata(request['row'])
    source=request.get('source',{})
    request.setdefault('source_work',source.get('path'))
    if not request.get('source_work'):raise ValueError('source_work or source.path is required')
    request.setdefault('robot_id',request['row']['name'])
    request.setdefault('scale_mode',source.get('scale_mode','preserve'))
    if request['scale_mode'] not in ('preserve','match_robot'):raise ValueError('scale_mode must be preserve or match_robot')
    if not request.get('source_sha256'):
        folder=Path(request['source_work'])
        request['source_sha256']={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.npz')}
    if not request.get('motion_backend_fingerprint'):
        files=sorted(Path(__file__).parent.glob('*.py'))
        request['motion_backend_fingerprint']=hashlib.sha256(''.join(sha(p) for p in files).encode()).hexdigest()
    if 'frames' not in request:
        raw=source_directory(request['source_work'])
        with np.load(raw/'native_source.npz',allow_pickle=False) as z:request['frames']=len(z['timestamps_s'])
    return request


def solve_one(request):
    request=normalize_request(request)
    assert_inputs(request)
    runtime=Runtime.read(request.get('runtime_config'));runtime.configure()
    start=time.perf_counter();out=Path(request['output_dir']);out.mkdir(parents=True,exist_ok=True)
    existing=list(out.iterdir())
    if existing and (set(p.name for p in existing)!={'request.json'} or normalize_request(json.loads((out/'request.json').read_text()))!=request):
        raise FileExistsError('Use a fresh output directory; prior runs and failed attempts are immutable: '+str(out))
    # Exclusive ownership prevents two workers from writing the same new slot.
    with (out/'attempt.json').open('x') as attempt:
        json.dump(dict(pid=os.getpid(),motion_backend_fingerprint=request['motion_backend_fingerprint']),attempt)
    saved_json(out/'request.json',request);saved_json(out/'policy.json',POLICY)
    if not runtime.device.startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('The resident motion solver requires an available CUDA device')
    torch.set_num_threads(2);torch.cuda.set_device(runtime.device)
    torch.backends.cuda.preferred_linalg_library('cusolver')
    torch.cuda.reset_peak_memory_stats()
    effective_source=request['source_work']
    if request['scale_mode']=='match_robot':
        from .scaling import materialize_source
        effective_source=out/'normalized_source'
        materialize_source(request['source_work'],request['row'],effective_source)
    src=load_source(effective_source)
    if src.get('morphology_scale'):saved_json(out/'morphology_scale.json',src['morphology_scale'])
    codec=runtime.load_codec()
    prior=infer_prior(request['row'],src,out,codec=codec)
    qworld=np.asarray(prior['qpos_world'],np.float64)
    T=len(qworld)
    if T!=request['frames']:raise ValueError('Source/output frame clock mismatch')
    setup=time.perf_counter();oracle=UniversalOracle(request['row'])
    origin=np.asarray([qworld[0,0],qworld[0,1],src['floor_z_m']],float)
    q=qworld.copy();q[:,:3]-=origin
    # A measured pelvis orientation must not be replaced by quantized model
    # orientation. Keep the model's local articulation, use the source root
    # frame, and preserve raw neural output as the explicit baseline.
    from scipy.spatial.transform import Rotation
    source_root_R=src['reference_world_R_zup'][:,0]
    q[:,3:7]=Rotation.from_matrix(source_root_R).as_quat(scalar_first=True)
    targets,unused_R=oracle.targets_from_prior(q)
    refs,bones,spos,snorm=targets
    refs=np.array(refs,copy=True)
    operation_request=dict(request,source_work=str(effective_source))
    saved_targets=Path(effective_source)/'operation-accuracy/source-targets.npz'
    if saved_targets.is_file():
        with np.load(saved_targets,allow_pickle=False) as z:operation_targets={k:z[k] for k in z.files}
    else:operation_targets=dict(points=src['obs_wrist_zup_m'][:,:,None,:],timestamps_s=src['times'])
    np.savez_compressed(out/'source-targets.npz',**operation_targets)
    taskgoal,offsets=points_from_request(operation_request,src)
    refs[:,2:4]=taskgoal-origin
    six=src['six_position_zup_m']
    # Foot targets preserve observed sole clearance and motion, calibrated to
    # each robot's own ankle-to-mesh sole offset at its neural initialization.
    _,foot_height=oracle.surface.heights(q)
    refs[:,4:6,:2]=six[:,4:6,:2]-origin[:2]
    refs[:,4:6,2]=(src['sole_low_z_m']-origin[2])+refs[:,4:6,2]-foot_height
    # Anatomical limb directions come from the supplied complete source body.
    # No six-point human completion is used; source provenance remains explicit.
    if 'inferred_bone_directions' in src:
        bones=np.asarray(src['inferred_bone_directions'])
        root_delta=source_root_R@src['inferred_world_R_zup'][:,0].transpose(0,2,1)
        bones=np.einsum('tij,tkj->tki',root_delta,bones)
    goal=np.asarray(prior['goal_body_R_native'])
    if request.get('local_point_file'):
        with np.load(request['local_point_file']) as z:
            if 'target_rotation_zup' in z:goal=np.asarray(z['target_rotation_zup'])
    low,_=oracle.surface.heights(q)
    q[:,2]+=np.maximum(.002-low,0)
    q[:,oracle.cols]=np.maximum(np.minimum(q[:,oracle.cols],oracle.hi),oracle.lo)
    # Feasible geometry initialization is separate from immutable neural output.
    np.savez_compressed(out/'initial_qpos.npz',qpos=q+np.pad(origin[None],((0,0),(0,q.shape[1]-3))),fps=30)
    setup_s=time.perf_counter()-setup
    budget=np.full((T,2),np.inf,dtype=float)
    op=HybridOps(oracle,q,(refs,bones,spos,snorm),goal,budget,device=runtime.device,task_offsets=offsets)
    op.weight[2:4]=POLICY['hand_position_weight']
    with torch.no_grad():
        assembly=op.linearize(op.base)
        rc,*_=op.residual_only(op.base,op.base,op.refs,op.bones,op.spos,op.snorm,op.orientation,op.budget)
        parity=float((rc-assembly['residual'].square().sum(-1)).abs().max())
        if parity>1e-6:raise AssertionError(('Objective/Jacobian residual parity',parity))
        del assembly,rc
        soft=FullMotion(op)
        soft_result,soft_record=soft.solve(POLICY['soft_iterations'],POLICY['max_qp_iterations'],POLICY['cooldown'])
        saved_json(out/'soft_solver.json',soft_record)
        if soft_record['status']=='QP_FAILED':
            raise RuntimeError('Soft SQP failed its numerical residual checks; result retained, not exported')
        soft_capture=soft.capture_s
        _,soft_error,_=op.verify(soft.q)
        budget=np.maximum(POLICY['hand_budget_floor_m'],soft_error.cpu().numpy()+POLICY['hand_budget_margin_m'])
        del soft
        # Tight budgets are computed once from achieved source tracking, never
        # from changing references. Unreachable targets remain visible in QA.
        op.budget=op.f(budget)
        polish=FullMotion(op,initial=op.f(soft_result))
        result,polish_record=polish.solve(POLICY['polish_iterations'],POLICY['max_qp_iterations'],POLICY['cooldown'])
        saved_json(out/'polish_solver.json',polish_record)
        if polish_record['status']=='QP_FAILED':
            raise RuntimeError('Polish SQP failed its numerical residual checks; result retained, not exported')
        polish_capture=polish.capture_s
        geo=op.geometry(polish.q,False)
        gpuP=geo['K']['xpos'].cpu().numpy()+origin
        gpuP[:,0]=0. # MuJoCo world body is a fixed coordinate origin.
        taskP=op.task_geometry(geo).cpu().numpy()+origin
        result[:,:3]+=origin
    np.savez_compressed(out/'gpu_qpos.npz',qpos=result,fps=30,hand_budget_m=budget)
    np.savez_compressed(out/'solver_targets.npz',target_position_world_m=refs+origin,
        target_rotation_world=goal,task_local_offsets=offsets,origin_world_m=origin,
        bone_directions=bones,task_body_names=np.array([oracle.b.m.body(int(i)).name for i in oracle.bids]),
        gpu_task_position_world_m=taskP,gpu_body_position_world_m=gpuP,floor_z_m=src['floor_z_m'],
        hand_budget_m=budget,surface_position_relative_prior=spos,surface_normal_prior=snorm)
    export_start=time.perf_counter()
    native_to_motion(request['row'],result,out,goal_R_native=goal,source=src,
                     operation_local_native_m=offsets[:,2:4])
    export_s=time.perf_counter()-export_start
    assert_inputs(request)
    raw={str(p.resolve()):sha(p) for p in out.rglob('*original.npz')}
    for name in ['native_prior.npz','source_inferred_prior.npz']:
        if (out/name).exists():raw[str((out/name).resolve())]=sha(out/name)
    if not raw:raise RuntimeError('No immutable neural prior produced')
    wall=time.perf_counter()-start
    scale=src.get('morphology_scale') if request['scale_mode']=='match_robot' else None
    scale_transform=dict(scale=float(scale['scale']) if scale else 1.,anchor_xyz_m=scale['anchor_world_m'] if scale else [0.,0.,src['floor']],world_up='z')
    def stop_reason(record):
        last=record['iterations'][-1]
        if record['status']=='QP_FAILED':return 'qp_numerical_failure'
        if not last.get('accepted',False):return 'line_search_rejected'
        if last.get('step_max',float('inf'))<2e-4:return 'accepted_step_below_2e-4'
        return 'iteration_budget_exhausted'
    solver=dict(status='stopped',converged=None,soft=dict(status=soft_record['status'],stop_reason=stop_reason(soft_record)),polish=dict(status=polish_record['status'],stop_reason=stop_reason(polish_record)),evidence='VALID_STOPPED checks motion constraints only; nonlinear KKT/global convergence is not established')
    report=dict(scale_transform=scale_transform,trajectory_scale=request['scale_mode'],solver=solver,motion_backend='gpu_hybrid',motion_backend_fingerprint=request['motion_backend_fingerprint'],
        robot_id=request['robot_id'],frames=T,fps=30,raw_sha256=raw,policy=POLICY,
        source_format=src['source_format'],
        source_policy=src.get('source_body_provenance',
            'Supplied complete body; provenance retained in input_manifest.json'),
        encoder_policy='supplied-full-body-all-visible-no-source-decode-v1',scale_mode=request['scale_mode'],
        morphology_scale=src['morphology_scale'],
        source_decoder_used=False,exact_source_output_fps=src['exact_fps'],
        support_policy='full local robot visual hull; conservative curved-primitive guard; fixed source floor plane',
        local_geometry=oracle.metadata,
        floor_z_m=src['floor_z_m'],objective_parity_max=parity,
        timing=dict(total_wall_s=wall,neural=prior.get('receipt',{}),
            geometry_setup_s=setup_s,soft_capture_s=soft_capture,polish_capture_s=polish_capture,
            soft=soft_record,polish=polish_record,native_export_s=export_s,
            refinement_active_s=soft_record['active_s']+polish_record['active_s'],
            refinement_wall_s=soft_record['wall_s']+polish_record['wall_s']),
        hardware=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
            peak_torch_allocated_mib=torch.cuda.max_memory_allocated()/2**20),
        scope='Kinematic conversion only. Optional match_robot scales human and object task together; no WBC, object contact, self-collision or dynamics certification. No multi-GPU scaling claim.',
        status='COMPUTED_REQUIRES_QUALITY_REVIEW')
    saved_json(out/'report.json',report)
    print('RESULT',json.dumps(dict(robot=request['robot_id'],output=str(out),wall_s=wall,
          refinement_active_s=report['timing']['refinement_active_s'])),flush=True)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_text())
    output=Path(request['output_dir'])
    historical=output.exists() and any(p.name!='request.json' for p in output.iterdir())
    try:solve_one(request)
    except Exception as exc:
        out=Path(request['output_dir']);out.mkdir(parents=True,exist_ok=True)
        if not historical and not isinstance(exc,FileExistsError) and not (out/'failure.json').exists():
            saved_json(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),robot=request.get('robot_id')))
        raise

if __name__=='__main__':main()
