"""Cross-process GPU leases. One heavy stage per card; never changes model policy."""
from arenalab_workbench.settings import paths
from contextlib import contextmanager
from pathlib import Path
import hashlib, json, os, subprocess, time, uuid

ROOT = Path(os.environ.get('ARENA_SCHEDULER_ROOT', str(paths.workbench / 'scheduler')))

def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(data)); tmp.replace(path)

@contextmanager
def resource_lock(key):
    import fcntl  # Linux worker locking; pure modules remain importable on clients.
    path = ROOT / 'resources' / (hashlib.sha256(key.encode()).hexdigest() + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield

def memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return {int(a): (int(b), int(c)) for a,b,c in (line.split(',') for line in result.stdout.splitlines())}

def alive(pid):
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False

def resident_only(gpu, record):
    """Recognize our idle worker, while retaining protection for external jobs."""
    if not record or record.get('state')!='idle':return False
    try:
        uuid_result=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True)
        result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        uuid=uuid_result.stdout.strip();pids={int(b.strip()) for a,b in (line.split(',') for line in result.stdout.splitlines()) if a.strip()==uuid}
        return bool(pids) and pids=={record['pid']}
    except (ValueError,OSError,subprocess.CalledProcessError):return False

@contextmanager
def lease(label, env=None, minimum_mb=40000):
    import fcntl
    env = dict(os.environ if env is None else env)
    # A leased orchestrator may launch sequential children on its own card.
    if env.get('ARENA_GPU_LEASE'):
        yield env; return
    ROOT.mkdir(parents=True, exist_ok=True)
    waiting = ROOT / 'waiting'; waiting.mkdir(exist_ok=True)
    token = f'{time.time_ns():020d}-{os.getpid()}-{uuid.uuid4().hex}'
    ticket = waiting / (token + '.json')
    record = dict(gpu_pool=list(map(int,env.get('ARENA_GPU_POOL','0,1,2,3').split(','))), minimum_mb=minimum_mb, pid=os.getpid(), label=label, state='queued', queued_at=time.time(), job=os.environ.get('ARENA_JOB_ID'))
    write(ticket, record); handle = None
    try:
        while handle is None:
            with resource_lock('allocation'):
                for p in waiting.glob('*.json'):
                    if not alive(json.loads(p.read_text())['pid']): p.unlink(missing_ok=True)
                queue = sorted(waiting.glob('*.json'))
                if ticket in queue:
                    available = memory()
                    for gpu in map(int, env.get('ARENA_GPU_POOL', '0,1,2,3').split(',')):
                        free, used = available.get(gpu, (0, 999999))
                        # External workloads are not part of our leases. Leave them alone.
                        from .motion_worker import resident_record,stop_idle
                        resident=resident_record(ROOT,gpu)
                        ours=used>1024 and resident_only(gpu,resident)
                        if used>1024 and not ours:continue
                        if free<minimum_mb:
                            if not ours or not stop_idle(ROOT,gpu):continue
                            free,used=memory().get(gpu,(0,999999))
                            if free<minimum_mb or used>1024:continue
                        # FIFO among jobs eligible for this card; a pinned job must not idle other cards.
                        older=[json.loads(p.read_text()) for p in queue[:queue.index(ticket)]]
                        if any(gpu in item.get('gpu_pool',[0,1,2,3]) and free>=item.get('minimum_mb',40000) for item in older):continue
                        candidate = (ROOT / f'gpu-{gpu}.lock').open('a+')
                        try: fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError: candidate.close(); continue
                        handle = candidate
                        record.update(state='running', gpu=gpu, started_at=time.time())
                        write(ROOT / 'events' / (token + '.json'), record)
                        ticket.unlink(missing_ok=True)
                        env.update(CUDA_VISIBLE_DEVICES=str(gpu), EGL_DEVICE_ID=str(gpu), ARENA_GPU_LEASE=token,
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', ARENA_GPU_LOCK_FD=str(handle.fileno()))
                        break
            if handle is None: time.sleep(.5)
        yield env
    except BaseException as error:
        record.update(state='failed', error=type(error).__name__)
        raise
    finally:
        ticket.unlink(missing_ok=True)
        if handle is not None:
            if record['state'] != 'failed': record['state']='completed'
            record['finished_at']=time.time()
            write(ROOT / 'events' / (token + '.json'), record)
            handle.close()

def run(command, *, label=None, env=None, **kwargs):
    check=kwargs.pop('check',False)
    try:
        with lease(label or Path(str(command[1])).name, env) as selected:
            if selected.get('ARENA_GPU_LOCK_FD'):
                fd=int(selected['ARENA_GPU_LOCK_FD'])
                try:os.fstat(fd)
                except OSError:pass
                else:kwargs['pass_fds']=tuple(set(kwargs.get('pass_fds',()))|{fd})
            return subprocess.run(command, env=selected, check=True, **kwargs)
    except subprocess.CalledProcessError as error:
        if check:raise
        return subprocess.CompletedProcess(command,error.returncode,error.stdout,error.stderr)

def merge_scene(path, update):
    """Serialize short read/modify/write operations, not GPU computation."""
    path=Path(path)
    with resource_lock('manifest:'+str(path.resolve())):
        data=json.loads(path.read_text())
        update(data)
        write(path,data)
