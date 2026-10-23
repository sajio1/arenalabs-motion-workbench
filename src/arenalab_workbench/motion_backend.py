"""Local motion backend boundary; never falls back to a different algorithm.

The runner owns neural initialization and body refinement. CV and independent
end-effector components keep their existing interfaces. This module validates
artifact integrity/schema, not physical feasibility or task acceptance.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import uuid

import numpy as np

SCHEMA = 'arenalabs.private-motion-request.v1'
BACKENDS = {'gpu_hybrid'}
SOURCE_FILES = ('canonical.npz', 'observations.npz', 'smplx_motion.npz',
                'smplh_motion.npz', 'human_reference.json', 'native_source.npz', 'source_provenance.json', 'source_hands.npz', 'input_manifest.json',
                'six-point-reference.npz', 'source-descriptor.npz', 'source_mesh.npz', 'objects.npz',
                'operation-accuracy/source-targets.npz')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def selected_backend(env=None):
    env = os.environ if env is None else env
    name = env.get('ARENA_MOTION_BACKEND', 'gpu_hybrid')
    if name not in BACKENDS:
        raise ValueError(f'Unknown ARENA_MOTION_BACKEND={name!r}; expected gpu_hybrid')
    return name


def runner_path(env=None):
    env = os.environ if env is None else env
    value = env.get('ARENA_GPU_MOTION_RUNNER')
    if not value:
        from importlib.util import find_spec
        spec = find_spec('arenalab_motion.runner')
        if spec is None or not spec.origin:
            raise RuntimeError('Install the unified GPU motion backend')
        return Path(spec.origin).resolve()
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Configured GPU motion runner does not exist: {path}')
    return path


def runtime_config(env=None):
    env = os.environ if env is None else env
    from arenalab_workbench.settings import paths
    return str(Path(env.get('ARENA_MOTION_RUNTIME', str(paths.config / 'runtime.cloud.json'))).resolve())


def trajectory_scale(env=None):
    env = os.environ if env is None else env
    mode = env.get('ARENA_TRAJECTORY_SCALE', 'preserve')
    if mode not in {'preserve', 'match_robot'}:
        raise ValueError('Unknown trajectory scale mode')
    return mode


def backend_fingerprint(env=None):
    """Portable code + explicit runtime/model identity; never numerical policy by robot ID."""
    env = os.environ if env is None else env
    from importlib.util import find_spec
    selected_backend(env)
    files = [Path(__file__), Path(__file__).with_name('retarget_robot.py'),
             Path(__file__).with_name('end_effectors.py'), Path(__file__).with_name('motion_sources.py')]
    files += [Path(__file__).with_name(name+'.py') for name in
              ('body_inference','human_reference','egoallo_runtime','unified_conversion','body_prior','gripper_body')]
    spec = find_spec('arenalab_motion')
    if spec and spec.submodule_search_locations:
        for root in spec.submodule_search_locations:
            files += sorted(Path(root).rglob('*.py'))
    sources = {str(p.relative_to(Path(__file__).parent.parent)): sha(p) for p in files if p.is_file()}
    runtime = Path(runtime_config(env))
    config = json.loads(runtime.read_text()) if runtime.is_file() else {}
    weights = Path(config.get('weights', '.'))
    if not weights.is_absolute(): weights = runtime.parent / weights
    models = {p.name: cached_sha(p) for p in (weights/'config.json', weights/'model.safetensors') if p.is_file()}
    from arenalab_workbench.settings import paths
    body_root=Path(env.get('ARENA_EGOALLO_ROOT',str(paths.egoallo)))
    body_assets={}
    for family in ('smpl','smplh','smplx'):
        value=env.get('ARENA_'+family.upper()+'_MODEL')
        path=Path(value) if value else (body_root/'model.npz' if family=='smplh' else None)
        if path is not None and path.is_file():body_assets[family]=cached_sha(path)
    checkpoint=body_root/'egoallo_checkpoint_april13'
    if checkpoint.exists():
        candidates=sorted(checkpoint.rglob('*')) if checkpoint.is_dir() else [checkpoint]
        for path in candidates:
            if path.is_file():body_assets['egoallo/'+str(path.relative_to(body_root))]=cached_sha(path)
    identity = dict(backend='gpu_hybrid', policy='resident-skeleton-surface-wrist-v3', sources=sources,
                    runtime=config, models=models, body_assets=body_assets)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


from functools import lru_cache
@lru_cache(maxsize=256)
def _cached_sha(path, size, mtime):
    return sha(path)


def cached_sha(path):
    path = Path(path).resolve(); stat = path.stat()
    return _cached_sha(str(path), stat.st_size, stat.st_mtime_ns)


def robot_asset_identity(xml):
    """Hash included XML/URDF and mesh dependencies, not just the entry XML."""
    import xml.etree.ElementTree as ET
    pending=[Path(xml).resolve()]; seen={}
    while pending:
        path=pending.pop()
        if str(path) in seen: continue
        if not path.is_file(): raise FileNotFoundError(path)
        seen[str(path)]=cached_sha(path)
        if path.suffix.lower() not in {'.xml','.urdf'}: continue
        tree=ET.parse(path).getroot(); compiler=tree.find('compiler')
        meshdir=compiler.get('meshdir','') if compiler is not None else ''
        for node in tree.iter():
            raw=node.get('file') or node.get('filename')
            if not raw or raw.startswith('package://'): continue
            dep=Path(raw)
            if not dep.is_absolute():
                dep=path.parent / (meshdir if node.tag=='mesh' else '') / dep
            pending.append(dep.resolve())
    return seen


def cache_matches(report, env=None):
    return (report.get('motion_backend') == selected_backend(env)
            and report.get('motion_backend_fingerprint') == backend_fingerprint(env)
            and report.get('trajectory_scale', 'preserve') == trajectory_scale(env))


def select_native_work_env(env,report,source_work,robot_id):
    """Reuse native paths only within the same backend; rollback gets its own root."""
    env=dict(env);source=Path(source_work).resolve();backend=selected_backend(env)
    if report.get('motion_backend','legacy')!=backend:
        env['ARENA_RETARGET_WORK']=str(source/'retarget'/robot_id/'backend-runs'/backend)
        env.pop('ARENA_GPU_WORK_ROOT',None);env.pop('ARENA_REFINED_OUTPUT',None)
        return env
    for field,key in [('native_work_dir','ARENA_RETARGET_WORK'),
                      ('native_work_root','ARENA_GPU_WORK_ROOT'),
                      ('native_output_dir','ARENA_REFINED_OUTPUT')]:
        value=report.get(field)
        if value and Path(value).resolve().is_relative_to(source):env[key]=value
    return env


def prepare_gpu_version(work_root,robot_id,row,source_work,operation_reference,local_points,env=None):
    """Choose an immutable work version, or reuse an integrity-checked result.

    The familiar operation-accuracy/palm layout is retained inside each version
    so renderers and dataset exports can follow validation.native_work_dir.
    Existing versions, failed attempts and absolute prior paths never move.
    """
    root=Path(work_root).resolve();points=np.asarray(local_points)
    seed=build_request(robot_id,row,source_work,root/'unused',operation_reference,env=env)
    identity=dict(robot_id=robot_id,row=row,model_assets=robot_asset_identity(row['xml']),source_sha256=seed['source_sha256'],
                  operation_reference=operation_reference,trajectory_scale=trajectory_scale(env),points_shape=points.shape,
                  points_dtype=str(points.dtype),points_sha256=hashlib.sha256(points.tobytes()).hexdigest())
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    base=root/'gpu-versions'/seed['motion_backend_fingerprint'][:24]/key
    def request_at(work):
        return build_request(robot_id,row,source_work,work/'operation-accuracy/palm'/robot_id,
                             operation_reference,work/'operation-local-points.npz',env=env)
    candidates=([base] if base.exists() else [])+sorted(base.parent.glob(base.name+'-retry-*'))
    for candidate in candidates:
        try:
            request=request_at(candidate);output=Path(request['output_dir'])
            old=json.loads((output/'request.json').read_text())
            report=json.loads((output/'report.json').read_text())
            if (output/'gpu_qpos.npz').is_file() and old==request and cache_matches(report,env):
                return candidate,request,validate_result(request)
        except (OSError,ValueError,RuntimeError,KeyError):
            pass # Preserve an incomplete/corrupt attempt and create a new sibling.
    if base.exists():base=base.with_name(base.name+'-retry-'+uuid.uuid4().hex[:12])
    base.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(base/'operation-local-points.npz',points=points)
    return base,request_at(base),None


def source_clock(source_work):
    """Accept a canonical body or the original direct-six export, never invent either."""
    source=Path(source_work)
    if (source/'native_source.npz').is_file():
        clock_file=source/'native_source.npz';kind='provided_full_body'
    elif (source/'canonical.npz').is_file():
        clock_file=source/'canonical.npz';kind='canonical_body'
    elif (source/'six-point-reference.npz').is_file() and (source/'source-descriptor.npz').is_file():
        clock_file=source/'six-point-reference.npz';kind='direct_six'
    else:
        raise FileNotFoundError('Source requires canonical.npz, or both six-point-reference.npz and source-descriptor.npz')
    with np.load(clock_file,allow_pickle=False) as data:times=np.asarray(data['timestamps_s'],float)
    if (times.ndim!=1 or len(times)<3 or not np.isfinite(times).all()
            or not np.allclose(np.diff(times),1/30,atol=1e-5,rtol=0)):
        raise ValueError('GPU motion bridge requires a finite contiguous 30 Hz source clock')
    with np.load(source/'observations.npz',allow_pickle=False) as observations:
        if observations['wrist_position'].shape!=(len(times),2,3):
            raise ValueError('Observed wrists and source clock must have identical frame counts')
        if 'timestamps_s' in observations:
            observed=np.asarray(observations['timestamps_s'])
            if observed.shape!=times.shape or not np.allclose(observed,times,atol=1e-5,rtol=0):
                raise ValueError('Observation and source clocks differ; explicit alignment required')
    return times,kind


def build_request(robot_id, row, source_work, output_dir, operation_reference,
                  local_point_file=None, raw_priors=None, env=None):
    if not isinstance(robot_id, str) or not robot_id:
        raise ValueError('A robot_id is required')
    if not all(row.get(key) for key in ('xml', 'body', 'endpoints', 'palm_frames')):
        raise ValueError('Robot row requires xml, body, endpoints and palm_frames physical metadata')
    if not Path(row['xml']).is_file():
        raise FileNotFoundError(row['xml'])
    if operation_reference not in {'palm', 'tcp'}:
        raise ValueError('operation_reference must be palm or tcp')
    source = Path(source_work).resolve()
    output = Path(output_dir).resolve()
    for filename in ('observations.npz',):
        if not (source / filename).is_file():
            raise FileNotFoundError(source / filename)
    times,source_kind=source_clock(source)
    if local_point_file is not None:
        local_point_file = Path(local_point_file).resolve()
        with np.load(local_point_file, allow_pickle=False) as saved:
            points = saved['points']
        if (points.ndim != 4 or points.shape[:2] != (len(times), 2)
                or points.shape[-1] != 3 or not np.isfinite(points).all()):
            raise ValueError('Operation local points must be finite [frames, 2, points, 3] Z-up metres')
    return dict(schema=SCHEMA, robot_id=robot_id, row=dict(row),
                source_work=str(source), output_dir=str(output), fps=30,
                frames=len(times), source_kind=source_kind, operation_reference=operation_reference,
                local_point_file=None if local_point_file is None else str(local_point_file),
                raw_priors=raw_priors,
                source_sha256={name: sha(source / name) for name in SOURCE_FILES
                               if (source / name).is_file()},
                local_point_sha256=None if local_point_file is None else sha(local_point_file),
                motion_backend='gpu_hybrid', motion_backend_fingerprint=backend_fingerprint(env),
                runtime_config=runtime_config(env), scale_mode=trajectory_scale(env),
                privacy='Local runner only; no implicit downloads, uploads or cloud workers',
                acceptance='Artifact integrity and schema only; no physical-feasibility claim')


def validate_result(request):
    source, output = Path(request['source_work']), Path(request['output_dir'])
    for name, digest in request['source_sha256'].items():
        if sha(source / name) != digest:
            raise RuntimeError(f'Immutable source changed during motion conversion: {name}')
    if request.get('local_point_file') and sha(request['local_point_file']) != request['local_point_sha256']:
        raise RuntimeError('Operation local point input changed during motion conversion')
    report = json.loads((output / 'report.json').read_text())
    raw = report.get('raw_sha256')
    if not isinstance(raw, dict) or not raw:
        raise ValueError('Runner report must include a nonempty raw_sha256 mapping of immutable priors')
    hashes = {}
    for filename, digest in raw.items():
        path = Path(filename)
        path = path.resolve() if path.is_absolute() else (output / path).resolve()
        if path == (output / 'motion.npz').resolve():
            raise ValueError('Refined motion cannot be its own immutable prior')
        if sha(path) != digest:
            raise RuntimeError(f'Immutable raw prior hash mismatch: {path}')
        hashes[str(path)] = digest
    with np.load(output / 'motion.npz', allow_pickle=False) as motion:
        required = {'q', 'root_t', 'rot6d', 'world_position_cm', 'joint_names',
                    'constraint_joint_indices', 'wrist_error_deg', 'sole_surface_height_cm', 'fps'}
        if required - set(motion.files):
            raise ValueError(f'Runner motion is missing native Grid fields: {sorted(required-set(motion.files))}')
        n, j = request['frames'], len(motion['joint_names'])
        shapes = {'q': (n, j, 3), 'root_t': (n, 3), 'rot6d': (n, j, 6),
                  'world_position_cm': (n, j, 3), 'wrist_error_deg': (n, 2)}
        for key, shape in shapes.items():
            if motion[key].shape != shape or not np.isfinite(motion[key]).all():
                raise ValueError(f'Invalid native motion array {key}; expected finite {shape}')
        if len(set(motion['joint_names'].tolist())) != j or float(motion['fps']) != 30:
            raise ValueError('Native motion must have unique joint names and fps=30')
        sole = motion['sole_surface_height_cm']
        if sole.shape != (n, 2) or not np.isfinite(sole).all():
            raise ValueError('Expected finite sole_surface_height_cm [frames,2]')
        indices = motion['constraint_joint_indices']
        if indices.ndim != 1 or len(indices) != 2 or not np.isin(indices, np.arange(j)).all():
            raise ValueError('Expected two valid native constraint_joint_indices')
    return report, hashes


def run_gpu_motion(request, *, env=None, cwd=None, stdout=None, stderr=None):
    env = dict(os.environ if env is None else env)
    output = Path(request['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / 'request.json'
    if any(output.iterdir()):
        raise FileExistsError('Motion output must be fresh; retain previous attempts and select a new version')
    request_path.write_text(json.dumps(request, indent=2, allow_nan=False) + '\n')
    command = [env.get('ARENA_GPU_MOTION_PYTHON', sys.executable), '-m', 'arenalab_motion.runner',
               '--request', str(request_path)]
    subprocess.run(command, env=env, cwd=cwd, stdout=stdout, stderr=stderr, check=True)
    report, hashes = validate_result(request)
    report.update(motion_backend='gpu_hybrid', trajectory_scale=request.get('scale_mode','preserve'),
                  motion_backend_fingerprint=request['motion_backend_fingerprint'])
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report, hashes


def normalized_source(source_work):
    """Stage a fresh normalized source; never replace original observed targets."""
    source=Path(source_work).resolve()
    if (source/'native_source.npz').is_file(): return source
    from arenalab_workbench.motion_sources import prepare_body_source
    manifest=json.loads((source/'input_manifest.json').read_text())
    inputs={name:cached_sha(source/name) for name in SOURCE_FILES if (source/name).is_file()}
    spec=manifest.get('motion_source')
    if spec:
        raw=Path(spec['path']);raw=raw if raw.is_absolute() else source/raw
        inputs['supplied_motion']=cached_sha(raw)
        if spec.get('objects_path'):
            objects=Path(spec['objects_path']);objects=objects if objects.is_absolute() else source/objects
            inputs['supplied_objects']=cached_sha(objects)
        model=Path(os.environ.get('ARENA_'+spec['family'].upper()+'_MODEL',''))
    else:model=source/'model.npz'
    if model.is_file(): inputs['body_model']=cached_sha(model)
    inputs['adapter']=sha(Path(__file__).with_name('motion_sources.py'))
    key=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    output=source/'unified-sources'/key
    # Several target GPUs can consume one source concurrently. Only one process
    # may materialize that immutable source version; followers verify its receipt.
    import fcntl
    output.parent.mkdir(parents=True,exist_ok=True)
    with (output.parent/('.'+key+'.lock')).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=output/'source_provenance.json'
        if receipt.exists():
            report=json.loads(receipt.read_text())
            if report.get('output_sha256') and all((output/name).is_file() and cached_sha(output/name)==digest for name,digest in report['output_sha256'].items()):
                return output
            raise ValueError('Immutable normalized source changed')
        return prepare_body_source(source,output)
