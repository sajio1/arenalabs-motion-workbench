from pathlib import Path
import json
root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)))
for clip,base in [('jacket',root),('crouch',root/'crouch')]:
 for robot in ['h2','a3']:
  p=base/'observed-refinement'/robot
  if not (p/'foot-prior-validation.json').exists():
   print(clip,robot,'running');continue
  t=json.loads((p/'torso-validation.json').read_text())['versions']
  f=json.loads((p/'foot-prior-validation.json').read_text())
  s=json.loads((p/'report.json').read_text())
  print(clip,robot,json.dumps(dict(bias=s.get('knee_morphology_bias_deg'),pitch={k:v['pitch_abs_error_deg'] for k,v in t.items()},wrists=[t['after']['wrist_position_cm']['max'],t['after']['wrist_orientation_deg']['max']],acc={k:v['joint_acceleration_rad_s2']['max'] for k,v in t.items()},feet=f)))
