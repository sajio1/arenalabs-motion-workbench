"""One geometric head reference definition for every embodiment.

The vendor URDFs do not provide a head optical-camera extrinsic. The mesh
centre or its front-surface projection is an explicitly uncalibrated
anatomical proxy. Neither is a measured optical centre.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import trimesh
from scipy.spatial.transform import Rotation

def reference(model, metadata, face_surface=False):
    joints=[j for j in range(model.njnt) if 'head' in model.joint(j).name.lower()]
    body=int(model.jnt_bodyid[joints[-1]])
    source=Path(metadata['source'])
    tree=ET.parse(source)
    link=tree.find("./link[@name='%s']" % model.body(body).name)
    package=source.parents[2]
    vertices=[]
    for visual in link.findall('visual'):
        mesh=visual.find('geometry/mesh')
        if mesh is None:continue
        filename=mesh.get('filename')
        path=package/filename.split('/',3)[-1] if filename.startswith('package://') else source.parent/filename
        loaded=trimesh.load(path,process=False)
        points=np.asarray(loaded.vertices)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
        origin=visual.find('origin')
        xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        vertices.append(points@Rotation.from_euler('xyz',rpy).as_matrix().T+xyz)
    if not vertices:raise ValueError('Head visual geometry required for declared anatomical proxy')
    vertices=np.concatenate(vertices)
    center=(vertices.min(0)+vertices.max(0))/2
    rest=mujoco.MjData(model);mujoco.mj_forward(model,rest)
    if face_surface:
        # An explicitly uncalibrated camera mounting reference on the face,
        # not the volume centre of the head or an invented headset thickness.
        forward=rest.xmat[body].reshape(3,3).T@np.array([1.,0.,0.])
        center=center+forward*((vertices@forward).max()-center@forward)
    return body,center,rest.xmat[body].reshape(3,3).T.copy()
