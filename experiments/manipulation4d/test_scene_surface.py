"""Repeated noisy observations of ONE wall must not become two parallel walls."""
import numpy as np
from scene_surface import fuse_background

H=W=80;T=24
z=np.ones((T,H,W),dtype='f4');z[::2]+=.015;z[1::2]-=.015
k=np.array([[80.,0,40],[0,80.,40],[0,0,1.]])
e=np.repeat(np.eye(4)[None,:3],T,0)
depth=dict(depth=z,images=np.full((T,H,W,3),180,dtype='u1'),conf=np.ones_like(z),intrinsics=np.repeat(k[None],T,0),extrinsics=e,ids=np.arange(T)*2)
original=z.copy();camera=e.copy()
p,colors,report=fuse_background(depth,np.ones_like(z,bool))
assert len(p)>1000,report
span=float(np.percentile(p[:,2],99)-np.percentile(p[:,2],1))
assert span<.008,(span,report)
assert abs(float(np.median(p[:,2]))-1)<.005
assert np.array_equal(original,z) and np.array_equal(camera,e)
assert len(colors)==len(p)
print('PASS: one noisy plane -> one zero-crossing surface; p99-p1 thickness',span,'m; source arrays unchanged')
