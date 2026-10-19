"""Shared resumable body inference entry. Enable only with the versioned source policy."""
from arenalab_workbench.settings import paths
from pathlib import Path
import json,subprocess,time,shutil,os,signal
from arenalab_workbench.egoallo_runtime import gpu_environment
from arenalab_workbench.window_plan import digest,save_json

def run(root,status=None):
    root=Path(root);code=Path(__file__).parent
    output=root/'windowed-v1';source_hash=digest(root/'observations.npz')
    identity=dict(observations_sha256=source_hash,
        denoiser_sha256=digest(root/'egoallo_checkpoint_april13/checkpoints_3000000/model.safetensors'),
        smplh_sha256=digest(root/'model.npz'),smplx_sha256=digest(root/'SMPLX_NEUTRAL.npz'))
    identity_path=root/'inference_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise RuntimeError('Inputs or model weights changed; refusing stale checkpoints')
    save_json(identity_path,identity)
    env=gpu_environment();env.update(ARENA_EGO_WORKDIR=str(root),ARENA_WINDOW_OUTPUT=str(output),
        PYTHONPATH=paths.pythonpath(),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',EGL_DEVICE_ID=os.environ.get('EGL_DEVICE_ID','3'))
    python=str(paths.egoallo_python)
    complete=False
    if (output/'progress.json').exists() and (output/'manifest.json').exists():
        progress=json.loads((output/'progress.json').read_text());manifest=json.loads((output/'manifest.json').read_text())
        if manifest['source_sha256']!=source_hash or any(digest(code/name)!=value for name,value in manifest['code_sha256'].items()):
            raise RuntimeError('Window inputs or code changed; use a new versioned workspace')
        complete=(progress.get('stage')=='complete' and (output/'egoallo_smplh.npz').exists()
                  and progress['output_sha256']==digest(output/'egoallo_smplh.npz'))
    if not complete:
        stream=None;stream_log=None;stream_started=time.time();rid=os.environ.get('ARENA_STREAM_ROBOT_ID')
        if rid:
            stream_log=(root/'window-robot-stream.log').open('a')
            stream=subprocess.Popen([str(paths.motion_python),str(code/'window_robot_stream.py'),str(root),rid],env=env,stdout=stream_log,stderr=subprocess.STDOUT,start_new_session=True)
        with (root/'run_egoallo_windowed.log').open('a') as log:
            proc=subprocess.Popen([python,str(code/'run_egoallo_windowed.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                if status:
                    progress=output/'progress.json'
                    value=json.loads(progress.read_text()) if progress.exists() else {}
                    phase=f"Recovering SMPL-X · window {value.get('window',1)}/{value.get('total_windows','…')} · {value.get('completed_frames',0)} frames saved"
                    if rid:
                        stream_path=root/'frame-stream'/rid/'progress.json'
                        if stream_path.exists():
                            current=json.loads(stream_path.read_text());value['robot_stream']=current
                            phase+=f" · AlphaMotion {current.get('committed_frames',0)} frames converted"
                    save_json(status,dict(status='running',phase=phase,body_progress=value))
                time.sleep(1)
            if proc.returncode:
                if stream and stream.poll() is None:os.killpg(stream.pid,signal.SIGTERM)
                if stream_log:stream_log.close()
                if status:save_json(status,dict(status='failed',phase='Body window failed; completed checkpoints retained'))
                raise subprocess.CalledProcessError(proc.returncode,proc.args)
        # Never hold the producer's GPU lease waiting for a second GPU. If no
        # consumer was scheduled, final conversion uses its normal batch path.
        if stream and stream.poll() is None:
            stream_path=root/'frame-stream'/rid/'progress.json'
            allocated=stream_path.exists() and json.loads(stream_path.read_text()).get('started_at',0)>=stream_started
            if not allocated:os.killpg(stream.pid,signal.SIGTERM)
        if stream_log:stream_log.close()
    assert digest(root/'observations.npz')==source_hash
    result=output/'egoallo_smplh.npz';dest=root/'egoallo_smplh.npz'
    if dest.exists():
        if digest(dest)!=digest(result):raise RuntimeError('Refusing to replace an immutable body prior')
    else:shutil.copyfile(result,dest)
    if not (root/'canonical.npz').exists():
        if status:save_json(status,dict(status='running',phase='Transferring the continuous body prior to SMPL-X'))
        with (root/'transfer_smplx.py.log').open('a') as log:
            subprocess.run([python,str(code/'transfer_smplx.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert digest(root/'observations.npz')==source_hash
    return root
