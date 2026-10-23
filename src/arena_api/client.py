"""Small synchronous SDK; the HTTP contract is the single source of behavior."""
from pathlib import Path
import time
import uuid
import httpx

class ArenaError(RuntimeError):
    def __init__(self,response):
        self.status_code=response.status_code
        try:self.problem=response.json()
        except ValueError:self.problem={'detail':response.text[:300]}
        super().__init__(str(self.problem))

class ArenaClient:
    def __init__(self,base_url,*,workspace_token=None,timeout=60,transport=None):
        self.http=httpx.Client(base_url=base_url.rstrip('/')+'/',timeout=timeout,transport=transport)
        if workspace_token:self.http.cookies.set('arena_workspace',workspace_token)

    def close(self):self.http.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    def request(self,method,path,**kwargs):
        response=self.http.request(method,'api/v1/'+path.lstrip('/'),**kwargs)
        if response.is_error:raise ArenaError(response)
        return response.json()

    def session(self):return self.request('POST','sessions')
    def capabilities(self):return self.request('GET','capabilities')
    def recordings(self,workflow='manipulation',**filters):return self.request('GET','recordings',params={'workflow':workflow,**filters})
    def robots(self):return self.request('GET','robots')
    def end_effectors(self):return self.request('GET','end-effectors')
    def plan(self,**spec):return self.request('POST','plans',json=spec)

    def run(self,*,idempotency_key,**spec):
        """Caller keeps the key across network retries; no hidden retry/new key."""
        return self.request('POST','runs',json=spec,headers={'Idempotency-Key':idempotency_key})

    def batch(self,items,*,idempotency_key):
        return self.request('POST','batches',json={'items':items},headers={'Idempotency-Key':idempotency_key})

    def job(self,ident):return self.request('GET','jobs/'+ident)
    def jobs(self,**params):return self.request('GET','jobs',params=params)
    def playback(self,ident):return self.request('GET','jobs/'+ident+'/playback')
    def artifacts(self,ident):return self.request('GET','jobs/'+ident+'/artifacts')
    def retry(self,ident,*,idempotency_key):return self.request('POST','jobs/'+ident+'/retries',headers={'Idempotency-Key':idempotency_key})
    def episodes(self,**params):return self.request('GET','simulation-episodes',params=params)
    def catalog(self,page='locomotion'):return self.request('GET','catalogs/'+page)
    def height_preview(self,recording_id):return self.request('GET','recordings/'+recording_id+'/height-preview')
    def set_height(self,recording_id,minimum_height_cm):
        return self.request('POST','height-adjustments',json={'scene_id':recording_id,'minimum_height_cm':minimum_height_cm})
    def update_robot_metadata(self,robot_id,metadata):return self.request('PATCH','robots/'+robot_id+'/metadata',json=metadata)
    def delete_robot(self,robot_id):return self.request('DELETE','robots/'+robot_id)
    def episode(self,ident,collection='default'):return self.request('GET','simulation-episodes/'+ident,params={'collection':collection})
    def episode_playback(self,ident,collection='default'):return self.request('GET','simulation-episodes/'+ident+'/playback',params={'collection':collection})
    def artifact(self,ident):return self.request('GET','artifacts/'+ident)

    def geometry_frame(self,scene,frame,*,revision='',rgb='jpeg'):
        response=self.http.get('api/v1/geometry/frame',params={'scene':scene,'frame':frame,'revision':revision,'rgb':rgb})
        if response.is_error:raise ArenaError(response)
        return response.content

    def wait(self,ident,*,timeout=3600,poll_interval=2):
        end=time.monotonic()+timeout
        while True:
            job=self.job(ident)
            if job['status'] in {'succeeded','failed','interrupted','cancelled','outcome_unknown'}:return job
            if time.monotonic()>=end:raise TimeoutError('Job still running: '+ident)
            time.sleep(max(.1,min(poll_interval,end-time.monotonic())))

    def upload_recording(self,path):
        with Path(path).open('rb') as stream:
            return self.request('POST','recordings/import',files={'file':(Path(path).name,stream,'application/zip')})

    def upload_robot(self,path,page='locomotion'):
        with Path(path).open('rb') as stream:
            return self.request('POST','robots/import/'+page,files={'file':(Path(path).name,stream)})

    def upload_end_effector(self,path):
        with Path(path).open('rb') as stream:
            return self.request('POST','end-effectors',files={'file':(Path(path).name,stream)})

    def download(self,job_id,path):
        """Stream an export; never replace an existing file or publish a partial ZIP."""
        target=Path(path)
        if target.exists():raise FileExistsError(target)
        job=self.job(job_id)
        if job['workflow']!='export' or job['status']!='succeeded':raise ValueError('Export is not ready')
        import hashlib
        checksum=hashlib.sha256();size=0
        temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.partial')
        try:
            with self.http.stream('GET','api/v1/jobs/'+job_id+'/download',follow_redirects=True) as response:
                if response.is_error:
                    response.read();raise ArenaError(response)
                with temporary.open('xb') as stream:
                    for block in response.iter_bytes(1024*1024):
                        stream.write(block);checksum.update(block);size+=len(block)
            result=job['result']
            if result.get('bytes') is not None and result['bytes']!=size:raise ValueError('Export byte count mismatch')
            if result.get('sha256') and result['sha256']!=checksum.hexdigest():raise ValueError('Export SHA256 mismatch')
            # Hard-link creation is exclusive, avoiding overwrite races.
            import os
            os.link(temporary,target)
        finally:
            if temporary.exists():temporary.unlink()
        return target
