from pathlib import Path
import json,zipfile
root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)))
summary={}
with zipfile.ZipFile(root/'observed-refinement-v3-delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
 for clip,base in [('jacket',root),('crouch',root/'crouch')]:
  summary[clip]={}
  for file in ['01-smplx.mp4','input_manifest.json','human_validation.json']:
   z.write(base/file,f'{clip}/{file}')
  for robot in ['h2','a3']:
   folder=base/'observed-refinement'/robot
   native=json.loads((folder/'native-validation.json').read_text())
   compact={k:{a:b for a,b in v.items() if a not in ['foot_floor_m','wrist_errors_cm','wrist_errors_deg']} for k,v in native.items()}
   summary[clip][robot]={'native':compact,'knee':json.loads((folder/'knee-diagnosis.json').read_text()),'solve':json.loads((folder/'report.json').read_text())}
   summary[clip][robot]['solve'].pop('history',None)
   for file in folder.iterdir():
    if file.suffix in ['.mp4','.png','.json'] or file.name=='motion.npz':z.write(file,f'{clip}/{robot}/{file.name}')
 z.writestr('validation-summary.json',json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
