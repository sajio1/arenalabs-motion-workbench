"""Height calibration lookup shared by workers and HTTP routes."""
from pathlib import Path
import json
from arenalab_workbench.settings import paths
ROOT=paths.workbench
REGISTRY=ROOT/'height-previews.json'
DEFAULTS=ROOT/'height-defaults.json'
CATALOG=paths.store / 'manipulation4d/catalog.json'

def read(path,default=None):return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)
def variants():return read(REGISTRY)
def variant(sid):return variants().get(sid)
def default_id(sid):return read(DEFAULTS).get(sid)
def public_entry(sid):
 v=variant(sid)
 from arenalab_workbench.dataset_io.storage import entry as uploaded_entry
 upload=uploaded_entry(sid)
 if upload:return upload['public']
 return v['public'] if v else next((x for x in read(CATALOG,[]) if x['id']==sid),None)
