"""Physical geometry only; no embodiment/action-specific behavior."""
import mujoco,numpy as np
from scipy.spatial import ConvexHull

def geometry(m,torso_name):
    d=mujoco.MjData(m);mujoco.mj_forward(m,d);torso=m.body(torso_name).id
    vertices=[]
    for g in range(m.ngeom):
        b=int(m.geom_bodyid[g]);fixed=True
        while b>1:
            if m.body_jntnum[b]:fixed=False;break
            b=int(m.body_parentid[b])
        if not fixed or m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        mesh=m.geom_dataid[g];start=m.mesh_vertadr[mesh];n=m.mesh_vertnum[mesh]
        vertices.append(m.mesh_vert[start:start+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
    vertices=np.concatenate(vertices);bottom=vertices[:,2].min()
    points=np.unique(np.round(vertices[vertices[:,2]<=bottom+.005,:2]-d.xpos[1,:2],6),axis=0)
    hull=ConvexHull(points);eq=hull.equations
    polygon=points[hull.vertices];center=polygon.mean(0)
    radius=float(np.min(-(eq[:,:2]@center+eq[:,2])))
    assert radius>.01,'Degenerate support geometry'
    return dict(torso=torso,up_local=d.xmat[torso].reshape(3,3).T@np.array([0.,0,1]),
                torso_xy_rest=(d.xpos[torso]-d.xpos[1])[:2],hull=eq,polygon=polygon,
                radius=radius,body_ipos=m.body_ipos.copy(),mass=m.body_mass/m.body_mass.sum())

def evaluate(p,r,g):
    """Numpy diagnostics for exactly the same physical quantities as the cost."""
    rb=r[:,1];local=np.einsum('tij,tj->ti',rb.transpose(0,2,1),p[:,g['torso']]-p[:,1])
    up=np.einsum('tij,j->ti',r[:,g['torso']],g['up_local'])
    cm=((p+np.einsum('tbij,bj->tbi',r,g['body_ipos']))*g['mass'][None,:,None]).sum(1)
    cm_local=np.einsum('tij,tj->ti',rb.transpose(0,2,1),cm-p[:,1])
    margin=-(cm_local[:,:2]@g['hull'][:,:2].T+g['hull'][:,2]).max(1)
    return dict(torso_tilt_deg=np.degrees(np.arccos(np.clip(up[:,2],-1,1))),
                torso_horizontal_offset_m=np.linalg.norm(local[:,:2]-g['torso_xy_rest'],axis=1),
                com_margin_m=margin,com_world=cm)
