from pathlib import Path
import os,sys,json
pinned=Path('/home/ubuntu/arenalab-experiments/manipulation4d/venv/bin/python')
if Path(sys.executable).absolute()!=pinned:os.execv(str(pinned),[str(pinned),*sys.argv])
os.environ['PYOPENGL_PLATFORM']='egl'
from gripper_render import run
from robot_preview import run as preview
body,dest,robot=sys.argv[1:]
meta=json.loads((Path(body)/'input_manifest.json').read_text())
if meta['dataset']=='EgoDex':
 from mobile_render_input import prepare
 run(prepare(body,dest),robot,dest)
else:preview(body,robot,dest)
