from pathlib import Path
import json
root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)))
cfg=json.loads(Path('/opt/arenalab/configs/viewer.cloud.json').read_text())
print('storage',cfg['storage'])
robots=cfg['robots']
store=Path(cfg['storage'])
for p in store.rglob('robot.json'):
 try:
  v=json.loads(p.read_text())
  if isinstance(v,dict):robots.append(v)
 except Exception:pass
print([(r.get('id'),r.get('label')) for r in robots])
for id,label in [('h2','h2'),('custom_8501018c27','a3')]:
 r=next((r for r in robots if r.get('id')==id),None)
 if r is None:r={'id':id,'pipeline_row':json.loads((store/'uploaded'/id/'descriptor.json').read_text())}
 print(label,r)
 (root/(label+'-robot.json')).write_text(json.dumps([r['pipeline_row']],indent=2))
 (root/(label+'-job.json')).write_text(json.dumps(dict(schema='greenwich.job.v1',id='egoallo-'+label,pipeline='fullbody',source='sources.json',robots=label+'-robot.json',output=label+'-result',runtime='/opt/arenalab/configs/runtime.cloud.json',fps=30),indent=2))
