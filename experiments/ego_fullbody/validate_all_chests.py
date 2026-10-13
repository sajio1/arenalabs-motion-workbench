from pathlib import Path
import os,sys,subprocess
root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)))
for base in [root,root/'crouch']:
 for robot in ['h2','a3']:
  subprocess.run([sys.executable,str(root/'validate_native_chest.py'),robot],env=dict(os.environ,ARENA_EGO_WORKDIR=str(base)),check=True)
