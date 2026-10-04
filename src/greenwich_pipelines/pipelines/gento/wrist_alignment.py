"""Forearm direction from the kinematic tree, independent of embodiment names."""
import mujoco
import numpy as np

def proximal_origins(model, wrists):
    rest=mujoco.MjData(model);mujoco.mj_forward(model,rest)
    origins=[]
    for wrist in wrists:
        parent=int(model.body_parentid[wrist])
        while parent>0 and np.linalg.norm(rest.xpos[wrist]-rest.xpos[parent])<1e-6:
            parent=int(model.body_parentid[parent])
        if parent==0:raise ValueError('No forearm segment upstream of wrist')
        origins.append(parent)
    return np.asarray(origins)

def angles(positions, rotations, wrists, proximal, semantic):
    forearm=positions[:,wrists]-positions[:,proximal]
    forearm/=np.maximum(np.linalg.norm(forearm,axis=-1,keepdims=True),1e-12)
    distal=(rotations[:,wrists]@semantic)[...,0]
    return np.rad2deg(np.arccos(np.clip((forearm*distal).sum(-1),-1,1)))
