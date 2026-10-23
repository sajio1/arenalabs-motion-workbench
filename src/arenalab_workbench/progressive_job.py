"""Shared stage publication. GPU computation is independent of display order."""
import json, time, threading
from pathlib import Path
from arenalab_workbench.gpu_scheduler import resource_lock, write, merge_scene

ORDER = ('motion', 'scene', 'fov', 'contact')

class Progress:
    def __init__(self, status, url, robot, selection=None, dependency_plan=None):
        self.status=Path(status); self.url=url; self.robot=robot
        self.started=time.time(); self.ready={}; self.revision=0
        self.lock=threading.Lock()
        self.selection=selection; self.dependency_plan=dependency_plan
    def publish(self, stage, phase):
        with self.lock:
            self.ready[stage]=True; self.revision+=1
            self.write('running',phase)
    def write(self, state, phase):
        payload=dict(status=state,phase=phase,started_at=self.started,
            elapsed_seconds=time.time()-self.started,scene_url=self.url,
            robot_id=self.robot,robot_ready=bool(self.ready.get('motion')),
            scene_ready=bool(self.ready.get('scene')),contact_ready=bool(self.ready.get('contact')),
            fov_ready=bool(self.ready.get('fov')),ready=dict(self.ready),revision=self.revision)
        if self.selection is not None:
            stages={'trajectory':'motion','point_cloud':'scene','contact':'contact','inpaint_fov':'fov'}
            effective=self.selection['effective_outputs']
            completed=[name for name in effective if self.ready.get(stages[name])]
            payload.update(self.selection,output_modalities=effective,completed_outputs=completed,
                output_status={name:('completed' if name in completed else 'failed' if state=='failed' else 'pending') for name in effective},
                dependency_plan=self.dependency_plan)
            # Internal FOV dependencies are not separately selected artifacts.
            for name,key in [('trajectory','robot_ready'),('point_cloud','scene_ready'),('contact','contact_ready'),('inpaint_fov','fov_ready')]:
                payload[key]=name in completed
        write(self.status,payload)
    def complete(self, message='Robot motion and scene ready'):
        self.write('completed',message)

def initialize(path, public, frames):
    path=Path(path)
    with resource_lock('manifest:'+str(path.resolve())):
        if path.exists():return
        write(path,dict(id=public['id'],title=public['title'],dataset=public['dataset'],
            fps=30,frames=frames,duration=(frames-1)/30,world='Source world',
            video=public.get('video'),end_effector=public.get('end_effector'),uploaded=public.get('uploaded',False),robots=[],objects=[],wrist_videos=[],
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
