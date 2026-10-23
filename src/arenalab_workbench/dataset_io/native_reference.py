"""Package authoritative native references without packaging customer robot models."""
from pathlib import Path
import json
import shutil
import numpy as np
from .native_reference_mujoco import SCHEMA, model_signature, sha, validate


def _continuous(q):
    q=np.asarray(q,dtype=np.float64).copy()
    norms=np.linalg.norm(q,axis=-1,keepdims=True)
    if not np.isfinite(q).all() or not np.allclose(norms,1,atol=1e-6,rtol=0):
        raise ValueError('Native motion has a non-unit quaternion')
    q/=norms
    for i in range(1,len(q)):
        flip=np.sum(q[i-1]*q[i],axis=-1)<0
        q[i]=np.where(np.asarray(flip)[...,None],-q[i],q[i])
    return q


def build_native_reference(motion_path, model_path, output, *, playback=None, robot_id=None):
    """Write reference.npz + manifest + standalone adapters; require local MJCF.

    motion_path is the immutable runner motion.npz (or gpu_qpos.npz). No motion
    inference, model publication, URDF transfer or trajectory repair occurs.
    """
    import mujoco as mj
    motion_path,model_path,output=Path(motion_path),Path(model_path),Path(output)
    if output.exists():raise FileExistsError('Native reference exports are immutable; use a fresh directory')
    model=mj.MjModel.from_xml_path(str(model_path))
    identity=model_signature(model)
    with np.load(motion_path,allow_pickle=False) as source:
        key='qpos_world' if 'qpos_world' in source else 'qpos'
        if key not in source:raise ValueError('Authoritative native qpos is absent')
        raw=np.asarray(source[key],dtype=np.float64).copy()
        if 'qpos_world' in source and 'qpos' in source and not np.array_equal(source['qpos_world'],source['qpos']):
            raise ValueError('Ambiguous native qpos and qpos_world')
        fps=float(source['fps'])
        source_time=np.asarray(source['timestamps_s']).copy() if 'timestamps_s' in source else None
        saved_fk={name:np.asarray(source[name]).copy() for name in ('native_body_position_m','native_body_rotation','descriptor_body_ids','prior_body_ids') if name in source}
    if fps!=30 or raw.ndim!=2 or raw.shape[1]!=model.nq or len(raw)<1 or not np.isfinite(raw).all():
        raise ValueError('Native reference requires finite complete native qpos at 30 Hz')
    free=np.flatnonzero(model.jnt_type==mj.mjtJoint.mjJNT_FREE)
    if len(free)!=1 or model.body_parentid[model.jnt_bodyid[free[0]]]!=0:
        raise ValueError('Native reference requires one floating root; multi-root/fixed-base topology needs an explicit contract')
    rootid=int(model.jnt_bodyid[free[0]]);rootqa=int(model.jnt_qposadr[free[0]]);rootva=int(model.jnt_dofadr[free[0]])
    n=len(raw);qpos=raw.copy();native_joints=[];scalar=[]
    kinds={int(mj.mjtJoint.mjJNT_FREE):('free',7,6,'m + unit quaternion wxyz'),
           int(mj.mjtJoint.mjJNT_BALL):('ball',4,3,'unit quaternion wxyz'),
           int(mj.mjtJoint.mjJNT_SLIDE):('slide',1,1,'m'),int(mj.mjtJoint.mjJNT_HINGE):('hinge',1,1,'rad')}
    for j in range(model.njnt):
        kind=int(model.jnt_type[j]);label,qw,vw,unit=kinds[kind];qa=int(model.jnt_qposadr[j]);va=int(model.jnt_dofadr[j])
        entry=dict(name=model.joint(j).name,type=label,mujoco_type=kind,unit=unit,native_qpos_address=qa,native_dof_address=va,qpos_width=qw,dof_width=vw,body=model.body(int(model.jnt_bodyid[j])).name)
        native_joints.append(entry)
        if label=='free':qpos[:,qa+3:qa+7]=_continuous(qpos[:,qa+3:qa+7])
        elif label=='ball':qpos[:,qa:qa+4]=_continuous(qpos[:,qa:qa+4])
        else:scalar.append(entry)
    velocity=np.zeros((n,model.nv),dtype=np.float64)
    for t in range(n-1):mj.mj_differentiatePos(model,velocity[t],1/30,qpos[t],qpos[t+1])
    if n>1:velocity[-1]=velocity[-2]
    state=mj.MjData(model);positions=[];quaternions=[];omega=[]
    for t in range(n):
        state.qpos[:]=qpos[t];state.qvel[:]=velocity[t];mj.mj_forward(model,state)
        positions.append(state.xpos[1:].copy());quaternions.append(state.xquat[1:].copy())
        omega.append(state.xmat[rootid].reshape(3,3)@velocity[t,rootva+3:rootva+6])
        if 'native_body_position_m' in saved_fk and 'descriptor_body_ids' in saved_fk:
            if not np.allclose(state.xpos[saved_fk['descriptor_body_ids']],saved_fk['native_body_position_m'][t],atol=2e-6,rtol=0):
                raise ValueError('Native source/model saved position FK differs')
        if 'native_body_rotation' in saved_fk and 'prior_body_ids' in saved_fk:
            if not np.allclose(state.xmat[saved_fk['prior_body_ids']].reshape(-1,3,3),saved_fk['native_body_rotation'][t],atol=2e-6,rtol=0):
                raise ValueError('Native source/model saved rotation FK differs')
    names=[j['name'] for j in scalar];qa=[j['native_qpos_address'] for j in scalar];va=[j['native_dof_address'] for j in scalar]
    arrays=dict(timestamp_s=np.arange(n,dtype=np.float64)/30,frame_index=np.arange(n,dtype=np.int64),
                joint_position=qpos[:,qa],joint_velocity=velocity[:,va],joint_names=np.asarray(names),
                root_position_m=qpos[:,rootqa:rootqa+3],root_quaternion_wxyz=qpos[:,rootqa+3:rootqa+7],
                root_linear_velocity_world_m_s=velocity[:,rootva:rootva+3],root_angular_velocity_world_rad_s=np.asarray(omega),
                native_qpos=qpos,native_qvel=velocity,link_pose_world=np.concatenate([positions,_continuous(quaternions)],axis=-1),
                link_names=np.asarray([model.body(i).name for i in range(1,model.nbody)]))
    if source_time is not None:
        if source_time.shape!=(n,) or not np.isfinite(source_time).all():raise ValueError('Invalid source timestamps')
        tolerance=max(1e-6,8*float(np.spacing(np.max(np.abs(source_time)))))
        if n>1 and not np.allclose(np.diff(source_time),1/30,atol=tolerance,rtol=0):raise ValueError('Source timestamps must follow the 30 Hz clock')
        arrays['source_timestamp_s']=source_time
    extra_names=[]
    arrays.update(additional_joint_names=np.asarray([],dtype=str),additional_joint_position=np.empty((n,0),dtype=np.float64),additional_joint_units=np.asarray([],dtype=str))
    if playback is not None:
        table=playback['joints'];pn=table['names'];pv=np.asarray(table['values'],float)
        if playback['fps']!=30 or playback['frames']!=n or pv.shape!=(n,len(pn)) or len(set(pn))!=len(pn) or not np.isfinite(pv).all():
            raise ValueError('Playback/native reference clock or joint table differs')
        for i,name in enumerate(names):
            if name not in pn or not np.allclose(pv[:,pn.index(name)],arrays['joint_position'][:,i],atol=1e-8,rtol=0):
                raise ValueError('Playback/native scalar joint differs: '+name)
        extra_names=[name for name in pn if name not in names]
        if extra_names:
            arrays['additional_joint_names']=np.asarray(extra_names)
            arrays['additional_joint_position']=pv[:,[pn.index(name) for name in extra_names]]
            units=table.get('units')
            if units is None:units=[table.get('unit','unspecified')]*len(pn)
            if isinstance(units,str):units=[units]*len(pn)
            if len(units)!=len(pn):raise ValueError('Additional joint units do not match columns')
            arrays['additional_joint_units']=np.asarray([units[pn.index(name)] for name in extra_names])
    output.mkdir(parents=True)
    np.savez_compressed(output/'reference.npz',**arrays)
    for name in ('native_reference_mujoco.py','native_reference_isaac.py'):
        source=Path(__file__).with_name(name)
        if source.is_file():shutil.copyfile(source,output/name)
    meta=dict(schema=SCHEMA,robot_id=robot_id,data='reference.npz',data_sha256=sha(output/'reference.npz'),frames=n,fps=30,
              nq=int(model.nq),nv=int(model.nv),root_link=model.body(rootid).name,root_joint=model.joint(int(free[0])).name,
              joint_names=names,joints=scalar,native_joints=native_joints,link_names=arrays['link_names'].tolist(),
              model=dict(included=False,local_model_required=True,source_file_sha256=sha(model_path),kinematic_sha256=identity,source_mujoco_version=mj.__version__,signature_scope='Named kinematic topology, axes, limits, reference coordinates and link transforms; not dynamics or visual mesh certification'),
              world=dict(up_axis='Z',handedness='right',length_unit='m',quaternion_order='wxyz',root_frame='Original local model floating-root link; no display transform or fitted alignment'),
              source=dict(file_name=motion_path.name,sha256=sha(motion_path),qpos_key=key,immutable_raw_retained=True,timestamp_semantics='Original source clock retained separately if present; frame interval must be 1/30 s within 1 microsecond source text-clock quantization or machine precision'),
              state_semantics='Current-frame generated kinematic position reference, not measured telemetry, effort or next-frame action',
              quaternion_policy='Derived package normalizes quaternion unit norm and temporal hemisphere only; immutable raw motion remains unchanged',
              derivatives=dict(method='MuJoCo mj_differentiatePos forward difference over 1/30 s; last sample repeats previous, one-frame clip zero',measured=False,joint_units='rad/s for hinge, m/s for slide',native_qvel='MuJoCo tangent-space coordinates by native_dof_address; free angular velocity in local root frame',root_angular_velocity_world_rad_s='Root angular velocity expressed in world Z-up frame'),
              additional_joints=dict(names=extra_names,units=arrays.get('additional_joint_units',np.asarray([])).tolist(),included=bool(extra_names),application_policy='Optional, preserved channels. Default body-only replay does not require hand models; --with-hands explicitly applies all channels by name to a matching local assembled model',model_validated_at_export=False,reason='Separately mounted end effectors require their own matching local model and mount metadata'),
              physics=dict(controller_included=False,actuator_commands=False,dynamic_tracking_validated=False,physical_feasibility_validated=False,source_actuator_count=int(model.nu),source_equality_count=int(model.neq)))
    (output/'manifest.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    receipt=validate(output,model_path,body_only=bool(extra_names))
    (output/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (output/'README.md').write_text('''# Native robot reference\n\nUse your matching original MJCF and meshes locally:\n\n```sh\npython native_reference_mujoco.py --bundle . --model /local/robot/model.xml --validate\npython native_reference_mujoco.py --bundle . --model /local/robot/model.xml\n```\n\nIsaac Sim 5.1 with a matching local URDF:\n\n```sh\n/path/to/isaac-sim/python.sh native_reference_isaac.py --bundle . --urdf /local/robot.urdf --headless --validate --output ./isaac-validation\n```\n\nNo URDF/MJCF, meshes, model weights or credentials are uploaded or included here. `manifest.json` binds the reference to named local model kinematics. The adapter rejects a different robot model and maps native state by joint name, not array ordering.\n\n`reference.npz` contains current-frame floating-base pose, named scalar joint positions, full native qpos, derived tangent velocities, named world link poses and a 30 Hz zero-based clock. `native_qpos` is the authoritative raw pose with only equivalent quaternion hemisphere/unit normalization in this derivative package. Source raw SHA is recorded; raw files remain unchanged. Z-up metres, wxyz quaternions, radians for hinge joints and metres for slides are explicit in the manifest. Do not apply the web viewer Y-up transform again.\n\nThe adapter uses `mj_forward` for kinematic reference replay. It does not drive actuators or demonstrate stable physics/WBC execution. Velocities are finite differences of references, not measured telemetry. Do not feed the LeRobot next-frame `action` array directly to simulator qpos. Additional mounted-hand coordinates are retained as optional channels. Default playback uses the body only and does not require hand models. Use --with-hands to apply every supplied body and hand channel to a matching local assembled model; this explicit mode rejects missing/untracked channels. --body-only is an alias for the default. No hand geometry or mounting transform is invented. Additional velocity is an explicit 30 Hz position finite difference. Body link FK and additional joint-state coverage are reported separately; extra-link FK ground truth is not available in this package.\n''')
    return dict(schema=SCHEMA,frames=n,scalar_joints=len(scalar),root_link=meta['root_link'],local_model_required=True,model_included=False,validation=receipt)
