"""Standalone local-model reader: kinematic references, never controller actions.

Requires NumPy and MuJoCo. The customer's MJCF and meshes stay on their device.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

SCHEMA = 'arenalabs.native-reference.v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def model_signature(model, *, body_subset=None, joint_subset=None):
    """Order/path independent kinematic identity; not a dynamics certificate."""
    import mujoco as mj
    body_names = [model.body(i).name for i in range(model.nbody)]
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    if any(not n for n in body_names[1:] + joint_names) or len(set(body_names)) != len(body_names) or len(set(joint_names)) != len(joint_names):
        raise ValueError('Native reference requires unique named bodies and joints')
    if body_subset is not None and not set(body_subset).issubset(body_names):raise ValueError('Local model is missing canonical body names')
    if joint_subset is not None and not set(joint_subset).issubset(joint_names):raise ValueError('Local model is missing canonical joint names')
    bodies = []
    for i in range(1, model.nbody):
        if body_subset is not None and body_names[i] not in body_subset:continue
        bodies.append(dict(name=body_names[i], parent=body_names[model.body_parentid[i]],
                           position=model.body_pos[i].tolist(), quaternion=model.body_quat[i].tolist()))
    joints = []
    widths = {int(mj.mjtJoint.mjJNT_FREE): 7, int(mj.mjtJoint.mjJNT_BALL): 4,
              int(mj.mjtJoint.mjJNT_SLIDE): 1, int(mj.mjtJoint.mjJNT_HINGE): 1}
    for i in range(model.njnt):
        if joint_subset is not None and joint_names[i] not in joint_subset:continue
        a = int(model.jnt_qposadr[i]); width = widths[int(model.jnt_type[i])]
        joints.append(dict(name=joint_names[i], body=body_names[model.jnt_bodyid[i]], type=int(model.jnt_type[i]),
                           axis=model.jnt_axis[i].tolist(), position=model.jnt_pos[i].tolist(),
                           qpos0=model.qpos0[a:a+width].tolist(), limited=int(model.jnt_limited[i]),
                           limits=model.jnt_range[i].tolist()))
    selected_nq=sum(widths[j['type']] for j in joints)
    selected_nv=sum({0:6,1:3,2:1,3:1}[j['type']] for j in joints)
    record = dict(nq=selected_nq, nv=selected_nv, bodies=sorted(bodies, key=lambda x:x['name']),
                  joints=sorted(joints, key=lambda x:x['name']))
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load(folder):
    folder = Path(folder).resolve()
    meta = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if meta.get('schema') != SCHEMA or meta.get('fps') != 30:
        raise ValueError('Unsupported native reference schema/clock')
    world = meta.get('world', {})
    if any(world.get(k) != v for k, v in dict(up_axis='Z', handedness='right', length_unit='m', quaternion_order='wxyz').items()):
        raise ValueError('Unsupported native reference world conventions')
    if meta.get('root_link') not in meta.get('link_names', []):
        raise ValueError('Native reference root link is not in the link table')
    path = (folder / meta['data']).resolve()
    if not path.is_relative_to(folder):
        raise ValueError('Native reference data path escapes bundle')
    if sha(path) != meta['data_sha256']:
        raise ValueError('Native reference checksum mismatch')
    with np.load(path, allow_pickle=False) as saved:
        frames = {key:saved[key] for key in saved.files}
    n = int(meta['frames']); j = len(meta['joint_names']); links = len(meta['link_names'])
    expected = dict(timestamp_s=(n,), frame_index=(n,), joint_position=(n,j), joint_velocity=(n,j),
                    root_position_m=(n,3), root_quaternion_wxyz=(n,4),
                    root_linear_velocity_world_m_s=(n,3), root_angular_velocity_world_rad_s=(n,3),
                    native_qpos=(n,int(meta['nq'])), native_qvel=(n,int(meta['nv'])), link_pose_world=(n,links,7))
    for key, shape in expected.items():
        if key not in frames or frames[key].shape != shape or not np.isfinite(frames[key]).all():
            raise ValueError('Invalid native reference array: ' + key)
    if n < 1 or not np.array_equal(frames['frame_index'], np.arange(n)) or not np.allclose(frames['timestamp_s'], np.arange(n)/30, atol=1e-10, rtol=0):
        raise ValueError('Native reference timestamps do not match the 30 Hz clock')
    for key in ('joint_names', 'link_names'):
        if frames[key].tolist() != meta[key] or len(set(meta[key])) != len(meta[key]):
            raise ValueError('Ambiguous native reference ' + key)
    for q in (frames['root_quaternion_wxyz'], frames['link_pose_world'][...,3:]):
        if not np.allclose(np.linalg.norm(q,axis=-1),1,atol=1e-6,rtol=0):
            raise ValueError('Non-unit native reference quaternion')
    if n > 1 and np.any(np.sum(frames['root_quaternion_wxyz'][:-1]*frames['root_quaternion_wxyz'][1:],axis=-1) < -1e-12):
        raise ValueError('Native reference root quaternion hemisphere is discontinuous')
    extra=meta.get('additional_joints',{}).get('names',[])
    if frames.get('additional_joint_names',np.asarray([])).tolist()!=extra or frames.get('additional_joint_position',np.empty((0,0))).shape!=(n,len(extra)) or not np.isfinite(frames['additional_joint_position']).all() or frames.get('additional_joint_units',np.asarray([])).tolist()!=meta.get('additional_joints',{}).get('units',[]):
        raise ValueError('Invalid additional joint channels')
    if len(set(extra))!=len(extra) or set(extra)&set(meta['joint_names']):
        raise ValueError('Ambiguous additional joint names')
    if 'source_timestamp_s' in frames:
        source=frames['source_timestamp_s']
        if source.shape!=(n,) or not np.isfinite(source).all():raise ValueError('Invalid source timestamps')
        tolerance=max(1e-6,8*float(np.spacing(np.max(np.abs(source)))))
        if n>1 and not np.allclose(np.diff(source),1/30,atol=tolerance,rtol=0):raise ValueError('Source timestamps must follow the 30 Hz clock')
    _validate_native_layout(meta,frames)
    return meta, frames


def _rotate(q, v):
    return v+2*np.cross(q[...,1:],np.cross(q[...,1:],v)+q[...,:1]*v)


def _validate_native_layout(meta, frames):
    """Reject ambiguous addresses and inconsistent redundant reference fields."""
    n=meta['frames'];nq=meta['nq'];nv=meta['nv'];native=meta.get('native_joints',[])
    seen=set();qslots=np.zeros(nq,dtype=int);vslots=np.zeros(nv,dtype=int);scalar=[];roots=[]
    kinds={0:('free',7,6,'m + unit quaternion wxyz'),1:('ball',4,3,'unit quaternion wxyz'),2:('slide',1,1,'m'),3:('hinge',1,1,'rad')}
    for joint in native:
        name=joint.get('name');kind=joint.get('mujoco_type')
        if not isinstance(name,str) or not name or name in seen or kind not in kinds:raise ValueError('Invalid native joint identity')
        seen.add(name);label,qw,vw,unit=kinds[kind]
        if joint.get('type')!=label or joint.get('qpos_width')!=qw or joint.get('dof_width')!=vw or joint.get('unit')!=unit:raise ValueError('Invalid native joint type/width/unit')
        qa=joint.get('native_qpos_address');va=joint.get('native_dof_address')
        if not isinstance(qa,int) or not isinstance(va,int) or qa<0 or va<0 or qa+qw>nq or va+vw>nv:raise ValueError('Native joint address outside state')
        qslots[qa:qa+qw]+=1;vslots[va:va+vw]+=1
        if label in ('free','ball'):
            quat=frames['native_qpos'][:,qa+3:qa+7] if label=='free' else frames['native_qpos'][:,qa:qa+4]
            if not np.allclose(np.linalg.norm(quat,axis=-1),1,atol=1e-6,rtol=0):raise ValueError('Non-unit native joint quaternion')
            if n>1 and np.any(np.sum(quat[:-1]*quat[1:],axis=-1)<-1e-12):raise ValueError('Native joint quaternion hemisphere is discontinuous')
            if label=='free':roots.append(joint)
        else:scalar.append(joint)
    if not np.all(qslots==1) or not np.all(vslots==1):raise ValueError('Native joint addresses do not partition state')
    if scalar!=meta['joints'] or [x['name'] for x in scalar]!=meta['joint_names']:raise ValueError('Scalar/native joint metadata differ')
    qa=[x['native_qpos_address'] for x in scalar];va=[x['native_dof_address'] for x in scalar]
    if not np.allclose(frames['joint_position'],frames['native_qpos'][:,qa],atol=1e-12,rtol=0) or not np.allclose(frames['joint_velocity'],frames['native_qvel'][:,va],atol=1e-12,rtol=0):raise ValueError('Named joint/native state arrays disagree')
    if len(roots)!=1 or roots[0]['name']!=meta['root_joint'] or roots[0]['body']!=meta['root_link']:raise ValueError('Ambiguous native root metadata')
    qa=roots[0]['native_qpos_address'];va=roots[0]['native_dof_address'];root=frames['native_qpos'][:,qa:qa+7]
    if not np.allclose(frames['root_position_m'],root[:,:3],atol=1e-12,rtol=0) or not np.allclose(frames['root_quaternion_wxyz'],root[:,3:],atol=1e-12,rtol=0):raise ValueError('Root/native state arrays disagree')
    if not np.allclose(frames['root_linear_velocity_world_m_s'],frames['native_qvel'][:,va:va+3],atol=1e-12,rtol=0) or not np.allclose(frames['root_angular_velocity_world_rad_s'],_rotate(root[:,3:],frames['native_qvel'][:,va+3:va+6]),atol=1e-10,rtol=0):raise ValueError('Root/native velocity arrays disagree')
    link=frames['link_pose_world'][:,meta['link_names'].index(meta['root_link'])]
    if not np.allclose(link[:,:3],root[:,:3],atol=1e-10,rtol=0) or not np.allclose(np.abs(np.sum(link[:,3:]*root[:,3:],axis=-1)),1,atol=1e-10,rtol=0):raise ValueError('Root/link pose arrays disagree')


def _declared_mimics(model, known_names, missing_names):
    """Resolve only explicit active MJCF scalar joint equalities, topologically.

    MuJoCo contract: y-y0 = sum(a[k]*(x-x0)**k), not y=poly(x).
    https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-joint
    """
    import mujoco as mj
    pending={};known=set(known_names)
    for i in range(model.neq):
        if int(model.eq_type[i])!=int(mj.mjtEq.mjEQ_JOINT) or not model.eq_active0[i]:continue
        slave=int(model.eq_obj1id[i]);driver=int(model.eq_obj2id[i]);name=model.joint(slave).name
        if name not in missing_names:continue
        if name in pending:raise ValueError('Multiple local equality relations for unmapped joint: '+name)
        if int(model.jnt_type[slave]) not in (2,3) or (driver>=0 and int(model.jnt_type[driver]) not in (2,3)):
            raise ValueError('Mimic equality requires scalar joints')
        qa=int(model.jnt_qposadr[slave]);va=int(model.jnt_dofadr[slave])
        dqa=int(model.jnt_qposadr[driver]) if driver>=0 else None;dva=int(model.jnt_dofadr[driver]) if driver>=0 else None
        pending[name]=dict(name=name,driver=model.joint(driver).name if driver>=0 else None,
                           qpos_address=qa,dof_address=va,driver_qpos_address=dqa,driver_dof_address=dva,
                           reference=float(model.qpos0[qa]),driver_reference=float(model.qpos0[dqa]) if driver>=0 else 0.,
                           coefficients=model.eq_data[i,:5].tolist(),source='Explicit active local MJCF joint equality')
    resolved=[]
    while pending:
        ready=[name for name,r in pending.items() if r['driver'] is None or r['driver'] in known]
        if not ready:break
        for name in ready:
            resolved.append(pending.pop(name));known.add(name)
    unresolved=set(missing_names)-{r['name'] for r in resolved}
    if unresolved:raise ValueError('Untracked/cyclic joints without a resolvable explicit local mimic equality: '+', '.join(sorted(unresolved)))
    return resolved


def load_model(folder, model_path, *, body_only=True):
    import mujoco as mj
    meta, frames = load(folder)
    model = mj.MjModel.from_xml_path(str(Path(model_path).resolve()))
    native_names=[joint['name'] for joint in meta['native_joints']]
    extras=meta.get('additional_joints',{}).get('names',[])
    available={model.joint(i).name for i in range(model.njnt)}
    missing=(set(native_names) if body_only else set(native_names)|set(extras))-available
    unknown=available-set(native_names)-set(extras)
    if missing:
        raise ValueError(f'Local model/reference joint channels differ: missing={sorted(missing)}, untracked={sorted(unknown)}. For --with-hands supply a matching assembled body/end-effector model; default body replay does not require hand models; undeclared mimic joints need an explicit mapping.')
    mimics=[] if body_only else _declared_mimics(model,set(native_names)|set(extras),unknown)
    if model_signature(model,body_subset=meta['link_names'],joint_subset=native_names) != meta['model']['kinematic_sha256']:
        raise ValueError('Local model kinematics do not match this reference; use the matching original MJCF and assets')
    mappings = []
    for joint in meta['native_joints']:
        local = model.joint(joint['name']); jid = local.id
        if int(model.jnt_type[jid]) != joint['mujoco_type']:
            raise ValueError('Local joint type differs: ' + joint['name'])
        mappings.append((int(model.jnt_qposadr[jid]), joint['native_qpos_address'], joint['qpos_width'],
                         int(model.jnt_dofadr[jid]), joint['native_dof_address'], joint['dof_width']))
    extra_mappings=[]
    if not body_only:
        for index,(name,unit) in enumerate(zip(extras,meta['additional_joints']['units'])):
            jid=model.joint(name).id;kind=int(model.jnt_type[jid])
            if (kind,unit) not in ((int(mj.mjtJoint.mjJNT_HINGE),'rad'),(int(mj.mjtJoint.mjJNT_SLIDE),'m')):
                raise ValueError('Additional local joint type/unit differs: '+name)
            extra_mappings.append((int(model.jnt_qposadr[jid]),int(model.jnt_dofadr[jid]),index))
    extra_velocity=np.zeros_like(frames['additional_joint_position'])
    if len(extra_velocity)>1:
        extra_velocity[:-1]=np.diff(frames['additional_joint_position'],axis=0)*30
        extra_velocity[-1]=extra_velocity[-2]
    return dict(model=model, state=mj.MjData(model), meta=meta, frames=frames, mappings=mappings,
                body_only=body_only,extra_mappings=extra_mappings,extra_velocity=extra_velocity,mimics=[] if body_only else mimics)


def apply_frame(context, frame):
    """Set one reference sample and evaluate FK; deliberately does not mj_step."""
    import mujoco as mj
    model, state, frames = context['model'], context['state'], context['frames']
    if not 0 <= frame < context['meta']['frames']:
        raise IndexError('Reference frame out of range')
    mj.mj_resetData(model, state)
    for qa, source_qa, qw, va, source_va, vw in context['mappings']:
        state.qpos[qa:qa+qw] = frames['native_qpos'][frame,source_qa:source_qa+qw]
        state.qvel[va:va+vw] = frames['native_qvel'][frame,source_va:source_va+vw]
    for qa,va,index in context['extra_mappings']:
        state.qpos[qa]=frames['additional_joint_position'][frame,index]
        state.qvel[va]=context['extra_velocity'][frame,index]
    for relation in context['mimics']:
        driver=relation['driver_qpos_address'];dva=relation['driver_dof_address']
        x=float(state.qpos[driver])-relation['driver_reference'] if driver is not None else 0.
        coeff=relation['coefficients']
        state.qpos[relation['qpos_address']]=relation['reference']+sum(a*x**k for k,a in enumerate(coeff))
        state.qvel[relation['dof_address']]=(sum(k*coeff[k]*x**(k-1) for k in range(1,5))*state.qvel[dva]) if dva is not None else 0.
    if not np.isfinite(state.qpos).all() or not np.isfinite(state.qvel).all():raise ValueError('Nonfinite mapped local state')
    state.time = float(frames['timestamp_s'][frame])
    mj.mj_forward(model, state)
    return state


def validate(folder, model_path, *, body_only=True):
    context = load_model(folder, model_path, body_only=body_only)
    model, meta, frames = context['model'], context['meta'], context['frames']
    ids = [model.body(name).id for name in meta['link_names']]
    position = rotation = additional = 0.
    for frame in range(meta['frames']):
        state = apply_frame(context, frame); expected = frames['link_pose_world'][frame]
        position = max(position, float(np.max(np.linalg.norm(state.xpos[ids]-expected[:,:3],axis=-1))))
        # Chord distance respects quaternion double coverage, avoiding acos roundoff.
        actual = state.xquat[ids]; goal = expected[:,3:]
        rotation = max(rotation, float(np.max(np.minimum(np.linalg.norm(actual-goal,axis=-1),np.linalg.norm(actual+goal,axis=-1)))))
        for qa,va,index in context['extra_mappings']:
            additional=max(additional,abs(float(state.qpos[qa]-frames['additional_joint_position'][frame,index])))
    if position > 2e-6 or rotation > 2e-6:
        raise ValueError(f'Native reference FK mismatch: {position} m, quaternion chord {rotation}')
    return dict(schema=SCHEMA, frames=meta['frames'], checked_link_poses=meta['frames']*len(ids),
                kinematic_replay_passed=True, position_max_m=position, quaternion_chord_max=rotation,
                dynamics_rollout_validated=False, controller_used=False, local_model_only=True,
                body_only=body_only,all_reference_channels_applied=not(body_only and meta['additional_joints']['names']),
                applied_additional_joint_names=[] if body_only else meta['additional_joints']['names'],
                unapplied_additional_joint_names=meta['additional_joints']['names'] if body_only else [],
                additional_joint_position_max_abs=additional,additional_link_fk_reference_available=False,
                derived_mimic_joint_names=[r['name'] for r in context['mimics']],
                mimic_semantics='Only explicit active local MJCF equality polynomials relative to qpos0; no guessed coupling')


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--model',required=True,type=Path,help='Matching local original MJCF; no file is uploaded')
    parser.add_argument('--validate',action='store_true')
    scope=parser.add_mutually_exclusive_group()
    scope.add_argument('--with-hands',dest='body_only',action='store_false',help='Explicitly require and apply every mounted hand/gripper channel using a matching local assembled model')
    scope.add_argument('--body-only',dest='body_only',action='store_true',help='Default: replay body reference; optional hand/gripper channels are not required')
    parser.set_defaults(body_only=True)
    args=parser.parse_args()
    if args.validate:
        print(json.dumps(validate(args.bundle,args.model,body_only=args.body_only),indent=2));return
    import mujoco.viewer
    context=load_model(args.bundle,args.model,body_only=args.body_only)
    extras=context['meta'].get('additional_joints',{}).get('names',[])
    if extras and args.body_only:print('Body reference only. Additional mounted component channels require their local model/mounts and are not applied: '+', '.join(extras))
    with mujoco.viewer.launch_passive(context['model'],context['state']) as viewer:
        frame=0
        while viewer.is_running():
            start=time.monotonic();apply_frame(context,frame);viewer.sync();frame=(frame+1)%context['meta']['frames']
            time.sleep(max(0,1/30-(time.monotonic()-start)))


if __name__ == '__main__':
    main()
