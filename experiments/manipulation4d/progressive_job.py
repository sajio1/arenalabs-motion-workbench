"""Shared stage publication. GPU computation is independent of display order."""
import json, time, threading
from pathlib import Path
from gpu_scheduler import resource_lock, write, merge_scene

ORDER = ('motion', 'scene', 'fov', 'contact')

class Progress:
    def __init__(self, status, url, robot):
        self.status=Path(status); self.url=url; self.robot=robot
        self.started=time.time(); self.ready={}; self.revision=0
        self.lock=threading.Lock()
    def publish(self, stage, phase):
        with self.lock:
            self.ready[stage]=True; self.revision+=1
            self.write('running',phase)
    def write(self, state, phase):
        write(self.status, dict(status=state,phase=phase,started_at=self.started,
            elapsed_seconds=time.time()-self.started,scene_url=self.url,
            robot_id=self.robot,robot_ready=bool(self.ready.get('motion')),
            ready=dict(self.ready),revision=self.revision))
    def complete(self):
        self.write('completed','Robot motion and scene ready')

def initialize(path, public, frames):
    path=Path(path)
    with resource_lock('manifest:'+str(path.resolve())):
        if path.exists():return
        write(path,dict(id=public['id'],title=public['title'],dataset=public['dataset'],
            fps=30,frames=frames,duration=(frames-1)/30,world='Source world',
            video=public['video'],end_effector=public.get('end_effector'),uploaded=public.get('uploaded',False),robots=[],objects=[],wrist_videos=[],
            semantic=dict(task='',objects=[],provenance='Generation in progress'),
            tactile=dict(status='unavailable',reason='Contact generation pending'),
            geometry=dict(source_trajectory_edited=False)))

def merge_perception(destination, fresh):
    def update(scene):
        for key in ('cloud','segmented_instances','objects','source_camera','semantic','world','video','wrist_videos'):
            if key in fresh:scene[key]=fresh[key]
        ground=scene.get('geometry',{}).get('ground')
        scene.setdefault('geometry',{}).update(fresh.get('geometry',{}))
        if ground:scene['geometry']['ground']=ground
    merge_scene(destination,update)
