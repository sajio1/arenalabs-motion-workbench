from arenalab_workbench.settings import paths
from pathlib import Path
import os,sys,json
pinned=paths.workbench / 'venv/bin/python'
if Path(sys.executable).absolute()!=pinned:os.execv(str(pinned),[str(pinned),*sys.argv])
os.environ['PYOPENGL_PLATFORM']='egl'
from arenalab_workbench.gripper_render import run
from arenalab_workbench.robot_preview import run as preview
body,dest,robot=sys.argv[1:]
meta=json.loads((Path(body)/'input_manifest.json').read_text())
if meta['dataset']=='EgoDex':
 from arenalab_workbench.mobile_render_input import prepare
 run(prepare(body,dest),robot,dest)
else:preview(body,robot,dest)
