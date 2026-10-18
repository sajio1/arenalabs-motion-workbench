"""A perception result must not terminate full generation before all modalities."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import full_job

with TemporaryDirectory() as directory:
 root=Path(directory);store=root/'store';dest=store/'test';dest.mkdir(parents=True);work=root/'body';work.mkdir();status=root/'state.json';source=root/'source';source.write_bytes(b'original source');calls=[]
 def perceive(scene_id,stage_status):
  assert stage_status!=status,'Intermediate scene completion must use separate status'
  full_job.save(stage_status,{'status':'completed','robot_ready':False})
  full_job.save(dest/'scene.json',dict(id=scene_id,robots=[],geometry={},video={},semantic={}))
  return '/assets/test/scene.json'
 def execute(command,**kwargs):
  script=Path(command[1]).name;calls.append(script)
  assert json.loads(status.read_text())['status']=='running'
  data=json.loads((dest/'scene.json').read_text())
  if script=='retarget_robot.py':data['robots']=[{'id':'robot'}]
  if script=='render_recording.py':data['patch_videos']={'robot':'/arms.mp4'}
  if script=='contact_recording.py':data['tactile']={'url':'/contact.json'}
  full_job.save(dest/'scene.json',data)
 with patch.object(full_job,'STORE',store),patch.object(full_job,'ROOT',root),patch.object(full_job,'resolve_clip',lambda _:({'id':'clip'},dict(source=str(source)))),patch.object(full_job,'scene_run',perceive),patch.object(full_job,'body_run',lambda *_:work),patch.object(full_job.subprocess,'run',execute):
  result=full_job.run('clip','robot','wuji',status)
 assert calls==['retarget_robot.py','render_recording.py','contact_recording.py'],calls
 assert json.loads(status.read_text())['robot_ready'] is True
 assert result=='/assets/test/scene.json' and source.read_bytes()==b'original source'
print('PASS: scene-only intermediate cannot complete a full job; body/hand, composite and contact stages all execute')
