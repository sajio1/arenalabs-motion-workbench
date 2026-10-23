"""Private, deterministic geometry adapter for the universal GPU retarget core.

The solver-facing object matches gpu_retarget_stack_20261022.hybrid_ops.HybridOps.
All preparation and reference extraction here is CPU-only and outside the hot
loop. No robot-name branches select math, targets, weights, or constraints.

Surface slots are actual local robot mesh points and mesh normals, assigned to
anatomical regions using the physical tree. This is a local shape-preservation
prior around an immutable decoded robot motion. It is NOT a trained UMR
human-to-robot surface correspondence and must not be reported as one.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, itertools, json, time
import numpy as np
import mujoco as mj
from scipy.spatial import ConvexHull, QhullError

REGIONS = ('root','left_hip','right_hip','left_knee','right_knee',
           'left_ankle','right_ankle','left_shoulder','right_shoulder',
           'left_elbow','right_elbow','left_wrist','right_wrist')
TASKS = ('root','chest','left_wrist','right_wrist','left_ankle','right_ankle')
BONE_ROLES = tuple((side+'_'+a,side+'_'+b) for side in ('left','right')
                   for a,b in (('shoulder','elbow'),('elbow','wrist'),
                               ('hip','knee'),('knee','ankle')))

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _chain(m, bid):
    result=[]
    while bid>0:result.append(int(bid));bid=int(m.body_parentid[bid])
    return result[::-1]

def _box_vertices(size):
    return np.asarray(list(itertools.product((-1.,1.),repeat=3)))*np.asarray(size)

def _mesh_data(m, mid):
    start,n=int(m.mesh_vertadr[mid]),int(m.mesh_vertnum[mid])
    vertices=np.asarray(m.mesh_vert[start:start+n],dtype=np.float64).copy()
    start,n=int(m.mesh_faceadr[mid]),int(m.mesh_facenum[mid])
    faces=np.asarray(m.mesh_face[start:start+n],dtype=np.int64)
    normals=np.zeros_like(vertices)
    if len(faces):
        areas=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
        for k in range(3):np.add.at(normals,faces[:,k],areas)
    length=np.linalg.norm(normals,axis=1)
    good=length>1e-14
    normals[good]/=length[good,None]
    # Degenerate vertices remain valid for exact floor minima, but are excluded
    # from normal-prior sampling instead of inventing a radial surface normal.
    return vertices,normals,good

def _hull_ids(vertices):
    if len(vertices)<=4:return np.arange(len(vertices),dtype=np.int64)
    try:return np.sort(ConvexHull(vertices).vertices)
    except QhullError:
        # Flat or degenerate geometry: retaining every vertex is exact.
        return np.arange(len(vertices),dtype=np.int64)

def _foot_roots(m, semantics):
    roots=[]
    for side in ('left','right'):
        knee=int(semantics[side+'_knee']);ankle=int(semantics[side+'_ankle'])
        chain=_chain(m,ankle)
        assert knee in chain,(side,'ankle must descend from knee')
        distal=chain[chain.index(knee)+1:]
        candidates=[]
        for b in distal:
            a,n=int(m.body_jntadr[b]),int(m.body_jntnum[b])
            names=[m.joint(j).name.lower() for j in range(a,a+n)] if n else []
            if any('ankle' in name for name in names):candidates.append(b)
        # Physical endpoint metadata always remains authoritative. If anatomical
        # joint names are absent, include the endpoint subtree and disclose it.
        roots.append(candidates[0] if candidates else ankle)
    return roots

class VisualSurface:
    """Exact mesh/box plane minima, conservative bounds for smooth cylinders.

    Hull vertices are sufficient for every rigidly rotated plane-distance
    query. Geom-local coordinates already contain MuJoCo mesh compilation
    scaling/centering, and are transformed with geom_xmat/geom_xpos.
    """
    def __init__(self,b,semantics):
        self.b=b;m=b.m
        # MuJoCo's URDF importer labels actual visual geoms group 1. Existing
        # vendor MJCF also uses group 2. Collision-only proxies stay excluded.
        visual=[g for g in range(m.ngeom) if m.geom_bodyid[g]>0 and m.geom_group[g] in (1,2) and m.geom_rgba[g,3]>0]
        if not visual:
            visual=[g for g in range(m.ngeom) if m.geom_bodyid[g]>0 and not m.geom_contype[g] and not m.geom_conaffinity[g] and m.geom_rgba[g,3]>0]
        if not visual:raise ValueError('No declared visual geoms; provide explicit visual metadata')
        self.geom_ids=np.asarray(visual,dtype=np.int64)
        roots=_foot_roots(m,semantics)
        foot_body={b:(0 if roots[0] in _chain(m,b) else 1 if roots[1] in _chain(m,b) else -1) for b in range(1,m.nbody)}
        self.mesh_cache={};vertices=[];gids=[];feet=[];self.records=[];self.primitives=[]
        self.sample_data={}
        for gi in visual:
            kind=int(m.geom_type[gi]);bid=int(m.geom_bodyid[gi])
            if kind==mj.mjtGeom.mjGEOM_MESH:
                mid=int(m.geom_dataid[gi])
                if mid not in self.mesh_cache:self.mesh_cache[mid]=_mesh_data(m,mid)
                v,n,good=self.mesh_cache[mid];hull=_hull_ids(v)
                self.sample_data[gi]=(v,n,good)
            elif kind==mj.mjtGeom.mjGEOM_BOX:
                v=_box_vertices(m.geom_size[gi]);hull=np.arange(8)
                # Box corners are nonsmooth: sample exact planar face centers
                # with unambiguous actual outward face normals for the prior.
                n=np.r_[np.eye(3),-np.eye(3)];p=n*m.geom_size[gi]
                self.sample_data[gi]=(p,n,np.ones(6,dtype=bool))
            elif kind==mj.mjtGeom.mjGEOM_CYLINDER:
                # A circumscribed regular prism encloses the actual cylinder at
                # every orientation. Its global Hausdorff excess is bounded by
                # r*(sec(pi/N)-1), so it cannot miss ground penetration. This
                # changes only the floor guard; rendering/kinematics use the
                # unchanged native cylinder. Exact analytic heights are exposed
                # separately for reporting and verification.
                radius,half=m.geom_size[gi,:2];sides=128
                angles=np.arange(sides)*2*np.pi/sides
                radial=np.c_[np.cos(angles),np.sin(angles)]
                expanded=radius/np.cos(np.pi/sides)
                v=np.concatenate([np.c_[radial*expanded,np.full(sides,z)] for z in (-half,half)])
                hull=np.arange(len(v));error=float(expanded-radius)
                p=np.r_[np.c_[radial*radius,np.zeros(sides)],[[0,0,-half],[0,0,half]]]
                n=np.r_[np.c_[radial,np.zeros(sides)],[[0,0,-1],[0,0,1]]]
                self.sample_data[gi]=(p,n,np.ones(len(p),dtype=bool))
                self.primitives.append(dict(geom=gi,kind='cylinder',sides=sides,max_outward_excess_m=error))
            else:
                raise NotImplementedError(f'Visual geom {gi} type {kind} lacks exact full-hull representation; no silent coverage loss')
            vertices.append(v[hull]);gids.extend([gi]*len(hull));feet.extend([foot_body[bid]]*len(hull))
            self.records.append(dict(geom=gi,body=m.body(bid).name,kind=kind,vertices=len(v),hull_vertices=len(hull),foot=foot_body[bid]))
        self.vertices=np.concatenate(vertices).astype(np.float64)
        self.ids=np.asarray(gids,dtype=np.int64);self.feet=np.asarray(feet,dtype=np.int64)
        self.foot_roots=roots
        self.groups=[np.flatnonzero(self.feet==side) for side in (0,1,-1)]
        if any(len(x)==0 for x in self.groups):raise ValueError('Expected nonempty left-foot, right-foot and remaining-body hull groups')
        self.coverage=dict(visual_geom_count=len(visual),covered_visual_geom_count=len(self.records),
                           full_hull_vertices=len(self.vertices),group_sizes=[len(x) for x in self.groups],
                           foot_root_bodies=[m.body(i).name for i in roots],
                           unsupported_visual_geoms=[],mesh_plane_minimum_exact=True,
                           plane_minimum_exact=not self.primitives,primitive_guards=self.primitives,
                           max_conservative_height_error_m=max([x['max_outward_excess_m'] for x in self.primitives],default=0.),
                           collision_proxies_included=False)

    def positions(self,qpos):
        qs=np.atleast_2d(np.asarray(qpos,dtype=float));result=[]
        for q in qs:
            self.b.d.qpos[:]=q;mj.mj_kinematics(self.b.m,self.b.d)
            result.append(np.einsum('nij,nj->ni',self.b.d.geom_xmat[self.ids].reshape(-1,3,3),self.vertices)+self.b.d.geom_xpos[self.ids])
        return np.asarray(result)

    def heights(self,qpos):
        qs=np.atleast_2d(np.asarray(qpos,dtype=float));low=[];feet=[]
        for q in qs:
            self.b.d.qpos[:]=q;mj.mj_kinematics(self.b.m,self.b.d)
            z=np.einsum('ij,ij->i',self.b.d.geom_xmat[self.ids].reshape(-1,3,3)[:,2],self.vertices)+self.b.d.geom_xpos[self.ids,2]
            low.append(z.min());feet.append([z[self.feet==s].min() for s in (0,1)])
        return np.asarray(low),np.asarray(feet)

    def exact_heights(self,qpos):
        """Actual visual surface plane minima; analytic for native cylinders."""
        qs=np.atleast_2d(np.asarray(qpos,dtype=float));low=[];feet=[];m,d=self.b.m,self.b.d
        primitive_ids={x['geom'] for x in self.primitives}
        for q in qs:
            d.qpos[:]=q;mj.mj_kinematics(m,d);minimum=[];which=[]
            for record in self.records:
                gi=record['geom'];indices=np.flatnonzero(self.ids==gi)
                if gi in primitive_ids:
                    rz=d.geom_xmat[gi].reshape(3,3)[2];radius,half=m.geom_size[gi,:2]
                    value=d.geom_xpos[gi,2]-radius*np.linalg.norm(rz[:2])-half*abs(rz[2])
                else:value=(self.vertices[indices]@d.geom_xmat[gi].reshape(3,3)[2]+d.geom_xpos[gi,2]).min()
                minimum.append(value);which.append(record['foot'])
            minimum=np.asarray(minimum);which=np.asarray(which)
            low.append(minimum.min());feet.append([minimum[which==s].min() for s in (0,1)])
        return np.asarray(low),np.asarray(feet)

class UniversalOracle:
    """One construction policy for every row; compatible with HybridOps."""
    def __init__(self,row,surface_slots_per_region=6):
        tick=time.perf_counter();self.row=row
        m=mj.MjModel.from_xml_path(str(row['xml']));d=mj.MjData(m)
        free=np.flatnonzero(m.jnt_type==mj.mjtJoint.mjJNT_FREE)
        scalar=np.flatnonzero(m.jnt_type==mj.mjtJoint.mjJNT_HINGE)
        if len(free)!=1 or int(m.jnt_qposadr[free[0]])!=0:raise ValueError('One free root at qpos[0:7] is required')
        if len(scalar)+1!=m.njnt:raise ValueError('This adapter requires a free root and scalar hinges')
        if m.neq:raise ValueError('Mechanical equality constraints must be represented explicitly before use')
        if np.max(m.body_jntnum)>1:raise ValueError('KinTree requires one joint per body')
        self.b=SimpleNamespace(m=m,d=d,name=row['name'],xml=Path(row['xml']),zero=m.qpos0.copy())
        self.cols=np.r_[np.arange(3),m.jnt_qposadr[scalar]].astype(np.int64)
        self.vcols=np.r_[np.arange(3),m.jnt_dofadr[scalar]].astype(np.int64);self.d=len(self.cols)
        names=row['semantic_metadata']['semantics']
        self.semantic_ids={k:mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,n) for k,n in names.items()}
        assert all(i>0 for i in self.semantic_ids.values())
        assert int(m.jnt_bodyid[free[0]])==self.semantic_ids['root']
        self.bids=[self.semantic_ids[k] for k in TASKS]
        self.bones=[(self.semantic_ids[a],self.semantic_ids[c],self.semantic_ids[a],self.semantic_ids[c]) for a,c in BONE_ROLES]
        self.directions=np.array([x for x in itertools.product((-1.,0.,1.),repeat=3) if any(x)])
        self.directions/=np.linalg.norm(self.directions,axis=1)[:,None]
        self.lo=np.r_[np.full(3,-np.inf),np.where(m.jnt_limited[scalar],m.jnt_range[scalar,0],-np.inf)]
        self.hi=np.r_[np.full(3,np.inf),np.where(m.jnt_limited[scalar],m.jnt_range[scalar,1],np.inf)]
        self.trust=np.r_[np.full(3,.035),np.full(len(scalar),.15)]
        self.surface=VisualSurface(self.b,self.semantic_ids);self.floor_groups=self.surface.groups
        self._sample_slots(int(surface_slots_per_region))
        self.build_seconds=time.perf_counter()-tick
        self.metadata=dict(schema='universal.geometry.v1',model_sha256=_sha(row['xml']),
                           surface_policy='deterministic actual mesh anatomical shape-preservation slots; no learned UMR correspondence',
                           surface_slots=len(self.geom),slots_per_region=int(surface_slots_per_region),
                           slots_by_region={r:int(np.sum(self.reg==i)) for i,r in enumerate(REGIONS)},
                           sampling='26 fixed world directions per visual geom, then deterministic farthest-point coverage per tree-nearest semantic region',
                           normal_policy='area-weighted triangle vertex normals; degenerate normals excluded; box face centers use exact normals',
                           root_orientation='fixed input quaternion; optimized variables are root xyz plus native hinge coordinates',
                           floor=self.surface.coverage,build_seconds=self.build_seconds)

    def _sample_slots(self,count):
        if count<1:raise ValueError('At least one surface slot per region is required')
        m,d=self.b.m,self.b.d;d.qpos[:]=m.qpos0;mj.mj_kinematics(m,d)
        region_ids=np.asarray([self.semantic_ids[k] for k in REGIONS]);chains=[_chain(m,b) for b in region_ids]
        def region(b):
            ch=_chain(m,b);scores=[]
            for i,other in enumerate(chains):
                common=0
                for x,y in zip(ch,other):
                    if x!=y:break
                    common+=1
                distance=len(ch)+len(other)-2*common
                # Equal distance prefers a true ancestor, then deeper anatomy;
                # final index makes repeated runs independent of hash order.
                scores.append((distance,region_ids[i] not in ch,-len(other),i))
            return min(scores)[-1]
        candidates=[[] for _ in REGIONS]
        for gi,(vertices,normals,good) in self.surface.sample_data.items():
            valid=np.flatnonzero(good)
            if not len(valid):continue
            R=d.geom_xmat[gi].reshape(3,3);world=vertices[valid]@R.T+d.geom_xpos[gi]
            chosen=np.unique(valid[np.argmax(world@self.directions.T,axis=0)])
            reg=region(int(m.geom_bodyid[gi]))
            for vi in chosen:candidates[reg].append((gi,int(vi),vertices[vi],normals[vi],R@vertices[vi]+d.geom_xpos[gi]))
        records=[]
        for reg,pool in enumerate(candidates):
            if not pool:continue
            p=np.asarray([x[4] for x in pool]);center=d.xpos[region_ids[reg]]
            selected=[int(np.argmax(np.sum((p-center)**2,axis=1)))]
            distance=np.sum((p-p[selected[0]])**2,axis=1)
            while len(selected)<min(count,len(pool)):
                distance[selected]=-1;next_index=int(np.argmax(distance));selected.append(next_index)
                distance=np.minimum(distance,np.sum((p-p[next_index])**2,axis=1))
            records.extend((reg,*pool[i]) for i in selected)
        if not records:raise ValueError('No usable actual mesh surface normals')
        self.reg=np.asarray([x[0] for x in records],dtype=np.int64)
        self.geom=np.asarray([x[1] for x in records],dtype=np.int64)
        self.local=np.asarray([x[3] for x in records]);self.normal=np.asarray([x[4] for x in records])
        self.anchor=region_ids[self.reg]
        self.slot_keys=[REGIONS[r]+'_'+str(int(np.sum(self.reg[:i]==r))) for i,r in enumerate(self.reg)]
        assert np.isfinite(self.local).all() and np.isfinite(self.normal).all()
        assert np.max(abs(np.linalg.norm(self.normal,axis=1)-1))<1e-10

    def fk(self,qpos):
        qpos=np.atleast_2d(np.asarray(qpos,dtype=np.float64));P=[];R=[]
        if qpos.shape[1]!=self.b.m.nq or not np.isfinite(qpos).all():raise ValueError('Invalid native qpos')
        for q in qpos:
            self.b.d.qpos[:]=q;mj.mj_kinematics(self.b.m,self.b.d)
            P.append(self.b.d.xpos.copy());R.append(self.b.d.xmat.reshape(-1,3,3).copy())
        return np.asarray(P),np.asarray(R)

    def targets_from_prior(self,qpos):
        """Extract immutable decoded-robot geometry targets, in native Z-up metres.

        Returns ((refs[T,6,3], bones[T,8,3], spos[T,S,3], snorm[T,S,3]),
                 wrist_orientation[T,2,3,3]). Caller may replace refs/bone dirs
        with calibrated source observations. Surface targets remain relative to
        their named robot anchors, not purported human surface correspondence.
        """
        qpos=np.atleast_2d(np.asarray(qpos,dtype=np.float64));refs=[];bones=[];spos=[];snorm=[];orientation=[]
        if qpos.shape[1]!=self.b.m.nq or not np.isfinite(qpos).all():raise ValueError('Invalid native qpos')
        a=np.array([x[0] for x in self.bones]);c=np.array([x[1] for x in self.bones])
        for q in qpos:
            self.b.d.qpos[:]=q;mj.mj_kinematics(self.b.m,self.b.d);d=self.b.d
            refs.append(d.xpos[self.bids].copy());vec=d.xpos[c]-d.xpos[a];length=np.linalg.norm(vec,axis=-1)
            if length.min()<1e-8:raise ValueError('Degenerate anatomical segment')
            bones.append(vec/length[:,None]);orientation.append(d.xmat[self.bids[2:4]].reshape(2,3,3).copy())
            R=d.geom_xmat[self.geom].reshape(-1,3,3)
            point=np.einsum('nij,nj->ni',R,self.local)+d.geom_xpos[self.geom]
            spos.append(point-d.xpos[self.anchor]);snorm.append(np.einsum('nij,nj->ni',R,self.normal))
        return tuple(np.asarray(x) for x in (refs,bones,spos,snorm)),np.asarray(orientation)

    def save_geometry(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,geom=self.geom,local=self.local,normal=self.normal,anchor=self.anchor,region=self.reg,
                            slot_keys=np.asarray(self.slot_keys),floor_geom=self.surface.ids,floor_local=self.surface.vertices,
                            floor_foot=self.surface.feet,cols=self.cols,vcols=self.vcols,bids=self.bids,bones=self.bones)
        path.with_suffix('.json').write_text(json.dumps(self.metadata,indent=2)+'\n')
