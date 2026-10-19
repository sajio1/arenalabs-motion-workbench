"""Non-overlap loop proposals verified by masked image and 3D correspondences."""
import numpy as np
from .geometry import robust_similarity, apply_transform


def describe(image, static):
    import cv2
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    descriptor = cv2.resize(np.where(static,gray,0),(16,16)).astype(float).ravel()
    descriptor -= descriptor.mean()
    descriptor /= max(np.linalg.norm(descriptor),1e-8)
    keypoints,features = cv2.SIFT_create(nfeatures=1600).detectAndCompute(gray,static.astype("uint8")*255)
    xy = np.array([k.pt for k in keypoints],dtype=float).reshape(-1,2)
    return descriptor,xy,features


def verified_loop(left,right,metric=False):
    """Both frames are in their own chunk gauge. Never use human/object pixels."""
    import cv2
    _,a,fa = describe(left["images"],left["valid"] & (left["labels"]==0))
    _,b,fb = describe(right["images"],right["valid"] & (right["labels"]==0))
    if fa is None or fb is None or min(len(fa),len(fb))<20:
        return None,dict(accepted=False,reason="insufficient static features")
    matches = cv2.BFMatcher().knnMatch(fa,fb,k=2)
    good = [m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance < .7*n.distance]
    if len(good)<20:
        return None,dict(accepted=False,reason="insufficient distinctive matches",matches=len(good))
    ua = np.rint(a[[m.queryIdx for m in good]]).astype(int)
    ub = np.rint(b[[m.trainIdx for m in good]]).astype(int)
    for xy,frame in [(ua,left),(ub,right)]:
        xy[:,0]=xy[:,0].clip(0,frame["depth"].shape[1]-1)
        xy[:,1]=xy[:,1].clip(0,frame["depth"].shape[0]-1)
    pa,pb = left["points"][ua[:,1],ua[:,0]],right["points"][ub[:,1],ub[:,0]]
    try:
        transform,audit = robust_similarity(pb,pa,metric=metric)
        keep = np.linalg.norm(apply_transform(transform,pb)-pa,axis=-1)<.05
        xy = ua[keep]
        area = float(np.prod(np.ptp(xy,axis=0))/np.prod(left["depth"].shape))
        if keep.sum()<20 or area<.05:
            return None,dict(accepted=False,reason="poor inlier coverage",inliers=int(keep.sum()),area=area)
        return (pa[keep],pb[keep]),dict(accepted=True,spatial_coverage=area,**audit)
    except ValueError as error:
        return None,dict(accepted=False,reason=str(error),matches=len(good))
