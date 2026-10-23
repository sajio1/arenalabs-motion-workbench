"""Local, serialized CUDA workers. Unix sockets and GPU leases never leave the host.

The caller passes its flock file descriptor with SCM_RIGHTS, so a disconnected
caller cannot release the GPU while the worker is still solving. Idle workers
hold model memory, not GPU leases. The scheduler may evict them for other stages.
"""
from pathlib import Path
import array
import contextlib
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import time


def process_start(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,ValueError,IndexError):return None


def resident_record(root,gpu):
    try:
        record=json.loads((Path(root)/f'resident-gpu-{gpu}.json').read_text())
        if process_start(record['pid'])!=record['process_start']:return None
        if Path(f'/proc/{record["pid"]}').stat().st_uid!=os.getuid():return None
        return record
    except (OSError,ValueError,KeyError):return None


def recv_exact(connection,size):
    out=bytearray()
    while len(out)<size:
        value=connection.recv(size-len(out))
        if not value:raise ConnectionError('Motion worker disconnected; inspect the retained attempt before retrying')
        out.extend(value)
    return bytes(out)


def exchange(path,payload,lease_fd=None):
    data=json.dumps(payload).encode();packet=struct.pack('!I',len(data))+data
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
        client.settimeout(60*60);client.connect(str(path))
        ancillary=[] if lease_fd is None else [(socket.SOL_SOCKET,socket.SCM_RIGHTS,array.array('i',[lease_fd]))]
        sent=client.sendmsg([packet],ancillary)
        if sent<len(packet):client.sendall(packet[sent:])
        size=struct.unpack('!I',recv_exact(client,4))[0]
        if size>1024*1024:raise RuntimeError('Invalid motion-worker response')
        return json.loads(recv_exact(client,size))


def stop_idle(root,gpu):
    record=resident_record(root,gpu)
    if not record:return True
    if record.get('state')!='idle':return False
    try:exchange(record['socket'],{'operation':'shutdown'})
    except OSError:return False
    deadline=time.monotonic()+15
    while process_start(record['pid'])==record['process_start'] and time.monotonic()<deadline:time.sleep(.05)
    return resident_record(root,gpu) is None


def run(request_path,*,env,stdout=None):
    from .gpu_scheduler import ROOT,resource_lock
    if not env.get('ARENA_GPU_LEASE') or not env.get('ARENA_GPU_LOCK_FD'):
        raise RuntimeError('Resident motion requires an active scheduler GPU lease')
    gpu=env['CUDA_VISIBLE_DEVICES']
    if not gpu.isdigit():raise ValueError('Resident worker requires one physical GPU index')
    # Different releases cannot reuse Python modules from an older worker.
    source=Path(__file__).resolve().parent.parent
    files=sorted((source/'arenalab_motion').glob('*.py'))+[Path(__file__)]
    version=hashlib.sha256(str(source).encode()+b''.join(p.read_bytes() for p in files)).hexdigest()[:16]
    directory=ROOT/'motion-workers';directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    address=directory/f'{gpu}-{version}.sock'
    if len(str(address).encode())>=104:raise ValueError('Scheduler path is too long for a private Unix socket')
    with resource_lock('resident-start:'+gpu):
        record=resident_record(ROOT,gpu)
        if record and record['socket']!=str(address):
            if not stop_idle(ROOT,gpu):raise RuntimeError('Previous motion worker is busy')
            record=None
        if not record:
            child=dict(env);child.pop('ARENA_GPU_LEASE',None);child.pop('ARENA_GPU_LOCK_FD',None)
            # No job-dependent paths are inherited by later jobs.
            for key in ('ARENA_JOB_ID','ARENA_SOURCE_WORK','ARENA_RETARGET_WORK','ARENA_EXPORT_DIR','ARENA_SCENE_WORK','ARENA_GPU_WORK_ROOT','ARENA_REFINED_OUTPUT'):
                child.pop(key,None)
            with (directory/f'{gpu}-{version}.log').open('ab') as log:
                worker=subprocess.Popen([env.get('ARENA_GPU_MOTION_PYTHON',sys.executable),'-m',__name__,
                    '--serve',str(address),'--gpu',gpu,'--root',str(ROOT)],env=child,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            deadline=time.monotonic()+90
            while not resident_record(ROOT,gpu):
                if worker.poll() is not None:raise RuntimeError('Motion worker startup failed; inspect its private log')
                if time.monotonic()>deadline:worker.terminate();raise TimeoutError('Motion worker startup timed out')
                time.sleep(.05)
    result=exchange(address,{'operation':'solve','request':str(Path(request_path).resolve())},int(env['ARENA_GPU_LOCK_FD']))
    if stdout is not None and result.get('log'):
        stdout.write(Path(result['log']).read_text(errors='replace'));stdout.flush()
    if not result.get('ok'):raise RuntimeError('Resident motion failed: '+result.get('error','unknown failure'))
    return result


def serve(address,gpu,root):
    from .gpu_scheduler import write
    from arenalab_motion.asset_cache import ResidentRuntime
    from arenalab_motion.runner import solve_one
    import torch
    import gc
    address=Path(address);root=Path(root);marker=root/f'resident-gpu-{gpu}.json'
    record=dict(pid=os.getpid(),process_start=process_start(os.getpid()),socket=str(address),gpu=gpu,state='idle')
    runtime=ResidentRuntime();address.unlink(missing_ok=True)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(str(address));os.chmod(address,0o600);server.listen(4);server.settimeout(1800)
        write(marker,record)
        try:
            while True:
                try:connection,_=server.accept()
                except socket.timeout:break
                fds=array.array('i');terminate=False
                with connection:
                    connection.settimeout(10)
                    try:
                        header,ancillary,flags,_=connection.recvmsg(4,socket.CMSG_SPACE(4*4))
                        for level,kind,value in ancillary:
                            if level==socket.SOL_SOCKET and kind==socket.SCM_RIGHTS:fds.frombytes(value[:len(value)//fds.itemsize*fds.itemsize])
                        if flags&socket.MSG_CTRUNC:raise ValueError('Truncated GPU lease descriptor')
                        if len(header)<4:header+=recv_exact(connection,4-len(header))
                        size=struct.unpack('!I',header)[0]
                        if size>65536:raise ValueError('Oversized local request')
                        command=json.loads(recv_exact(connection,size))
                        if command.get('operation')=='shutdown':result={'ok':True};terminate=True
                        elif command.get('operation')=='solve':
                            if len(fds)!=1:raise ValueError('GPU lease descriptor required')
                            # A real shared scheduler flock must remain owned for this solve.
                            lock=Path(os.readlink(f'/proc/self/fd/{fds[0]}')).resolve()
                            if lock!= (root/f'gpu-{gpu}.lock').resolve():raise ValueError('Wrong GPU lease descriptor')
                            request=json.loads(Path(command['request']).read_text())
                            output=Path(request['output_dir']);log_path=output/'resident-worker.log'
                            record.update(state='running');write(marker,record)
                            # Opening the log before solve_one would violate its fresh-output contract.
                            log_path=output.with_name(output.name+'-resident-worker.log')
                            with log_path.open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                                try:solve_one(request,resident=runtime);result={'ok':True,'log':str(log_path),'pid':os.getpid()}
                                except Exception as error:
                                    import traceback
                                    traceback.print_exc()
                                    if (output/'attempt.json').exists() and not (output/'failure.json').exists():
                                        write(output/'failure.json',dict(type=type(error).__name__,message=str(error)))
                                    result={'ok':False,'error':type(error).__name__+': '+str(error),'log':str(log_path)}
                                    # A failed CUDA context must never be reused; no silent retry.
                                    terminate=True
                            gc.collect();torch.cuda.empty_cache()
                            record.update(state='idle');write(marker,record)
                        else:raise ValueError('Unknown local worker operation')
                    except Exception as error:result={'ok':False,'error':str(error)}
                    finally:
                        for fd in fds:os.close(fd)
                    payload=json.dumps(result).encode()
                    try:connection.sendall(struct.pack('!I',len(payload))+payload)
                    except OSError:pass
                if terminate:break
        finally:
            address.unlink(missing_ok=True)
            current=resident_record(root,gpu)
            if current and current['pid']==os.getpid():marker.unlink(missing_ok=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--serve',required=True);parser.add_argument('--gpu',required=True);parser.add_argument('--root',required=True)
    args=parser.parse_args();serve(args.serve,args.gpu,args.root)
