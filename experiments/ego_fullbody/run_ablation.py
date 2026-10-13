from pathlib import Path
import os,sys,json,subprocess
root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)))
policies={'no_wrist_orientation':{'wrist_orientation_weight':0.},'free_root_height':{'root_height_weight':0.},'free_knee':{'knee_weight':0.},'free_height_knee':{'root_height_weight':0.,'knee_weight':0.}}
policies.update(strong_body={'body_scale':.08},no_model={'model_weight':0.},strong_body_no_model={'body_scale':.08,'model_weight':0.})
case=sys.argv[1];robot=sys.argv[2];config={'body_scale':.25,**policies[case]};dest=root/'reach-diagnostics'/case/robot;dest.mkdir(parents=True,exist_ok=True);(dest/'diagnostic-policy.json').write_text(json.dumps(config,indent=2))
env=dict(os.environ,ARENA_DIAGNOSTIC_POLICY=json.dumps(config),ARENA_OUTPUTDIR=str(dest))
subprocess.run([sys.executable,str(root/'refine_observed.py'),robot],env=env,check=True)
