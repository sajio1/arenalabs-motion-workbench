"""Bounded process-local static robot caches. No customer geometry leaves this host."""
from collections import OrderedDict
import copy
import hashlib
import json


def robot_key(row, **options):
    from arenalab_workbench.motion_backend import robot_asset_identity
    row=row.get('pipeline_row',row)
    identity=robot_asset_identity(row.get('model') or row['xml'])
    return hashlib.sha256(json.dumps([row,identity,options],sort_keys=True,default=str).encode()).hexdigest()


class StaticCache:
    def __init__(self, capacity=4):
        self.capacity=capacity;self.items=OrderedDict()

    def get(self,key,build):
        hit=key in self.items
        if hit:value=self.items.pop(key)
        else:value=build()
        self.items[key]=value
        while len(self.items)>self.capacity:self.items.popitem(last=False)
        return value,hit


_geometry=StaticCache()
_targets=StaticCache()


def geometry(row, slots=6):
    import mujoco
    from .geometry import UniversalOracle
    template,hit=_geometry.get(robot_key(row,slots=slots),lambda:UniversalOracle(row,slots))
    # Model, hulls and anatomical samples are static. Per-job MuJoCo state is not.
    oracle=copy.copy(template);oracle.b=copy.copy(template.b)
    oracle.b.d=mujoco.MjData(template.b.m)
    oracle.surface=copy.copy(template.surface);oracle.surface.b=oracle.b
    oracle.metadata=copy.deepcopy(template.metadata)
    return oracle,hit


def target(row,build):
    value,_=_targets.get(robot_key(row),build)
    return dict(value)


class ResidentRuntime:
    def __init__(self):self.codecs=StaticCache(capacity=1)

    def codec(self,runtime):
        key=(str(runtime.weights),runtime.device,json.dumps(runtime.identity(),sort_keys=True))
        codec,hit=self.codecs.get(key,runtime.load_codec)
        if hit:codec._grid_cold_load_s=0.
        return codec,hit
