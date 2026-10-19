"""Physical robot registry shared by HTTP routes and compute workers."""
from arenalab_workbench.settings import paths
from pathlib import Path
import json
def roster(config_path=None):
    config=json.loads(Path(config_path or paths.viewer_config).read_text());rows={r['id']:r for r in config['robots']}
    registry=paths.store / 'uploaded/registry.json'
    if registry.exists():rows.update({r['id']:r for r in json.loads(registry.read_text())})
    return {key:value for key,value in rows.items() if value.get('conversion_ready') and value.get('pipeline_row') and key not in config.get('hidden_robot_ids',[])}

def supported(robot):
    row=robot['pipeline_row']
    return bool(row.get('mobile_semantics')) if robot['kind']=='Wheeled' else 'palm_frames' in row
