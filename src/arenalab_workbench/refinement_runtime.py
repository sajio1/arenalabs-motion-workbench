"""Run the shared solver, retaining an eager fallback for capture incompatibility."""
import os,subprocess
from pathlib import Path

def run_refinement(command,**kwargs):
    check=kwargs.pop('check',False)
    env=dict(kwargs.get('env',os.environ));kwargs['env']=env
    result=subprocess.run(command,**kwargs)
    output=kwargs.get('stdout');path=Path(output.name) if hasattr(output,'name') and isinstance(output.name,str) else None
    if result.returncode and env.get('ARENA_REFINEMENT_BACKEND','cuda_graph')=='cuda_graph' and path:
        output.flush();tail=path.read_text(errors='replace')[-16000:]
        if 'CUDA' in tail and ('captur' in tail or 'CUDAGraph' in tail):
            output.write('\nCUDA Graph unavailable; retrying the unchanged objective in a fresh eager process.\n');output.flush()
            env['ARENA_REFINEMENT_BACKEND']='eager'
            result=subprocess.run(command,**kwargs)
    if check and result.returncode:raise subprocess.CalledProcessError(result.returncode,command)
    return result
