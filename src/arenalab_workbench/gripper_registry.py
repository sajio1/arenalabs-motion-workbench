"""Recorded gripper source lookup; no job or API imports."""
import json
from arenalab_workbench.settings import paths
ROOT=paths.workbench
REGISTRY=ROOT/'gripper-sources.json'
def sources():
 from arenalab_workbench.dataset_io.storage import uploads
 return {**(json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}),**{sid:r['gripper'] for sid,r in uploads().items() if r.get('gripper')}}
