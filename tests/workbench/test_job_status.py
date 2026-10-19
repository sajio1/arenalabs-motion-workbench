"""Regression for POST failing when its queued job starts before status returns."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
from arenalab_workbench.job_status import job_view


def test_job_status():
    with TemporaryDirectory() as directory:
        root=Path(directory)
        jobs={key:{'id':key,'created_at':0} for key in ['running','waiting','new']}
        for key,job in jobs.items():
            folder=root/key;folder.mkdir();(folder/'state.json').write_text(json.dumps({'status':'running' if key=='running' else 'queued'}))
        target=root/'new/state.json';original=Path.read_text;reads=[]
        def starts_during_read(path,*args,**kwargs):
            text=original(path,*args,**kwargs)
            if path==target:
                reads.append(path);path.write_text(json.dumps({'status':'running'}))
            return text
        with patch.object(Path,'read_text',starts_during_read):
            view=job_view(jobs['new'],jobs,root)
        assert view['status']=='queued' and view['queue_position']==2,view
        assert len(reads)==1
        assert job_view(jobs['new'],jobs,root)['status']=='running'
        assert job_view(jobs['waiting'],jobs,root)['queue_position']==1
        target.write_text(json.dumps({'status':'completed','scene_url':'/scene.json'}))
        assert job_view(jobs['new'],jobs,root)['scene_url']=='/scene.json'
    print('PASS: queued-to-running race, ordered queue positions, completion payload')
