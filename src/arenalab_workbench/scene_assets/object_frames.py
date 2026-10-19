"""Original-resolution SAM/RGB paired with geometrically mapped MapAnything depth."""
from pathlib import Path
import numpy as np
from .store import GeometryStore,NpzArrays
from .reconstruct import MaskFrames
from .geometry import backproject


class ObjectFrames(GeometryStore):
    def __init__(self,geometry,work):
        super().__init__(geometry)
        self.work=Path(work)
        self.source_ids=np.load(self.work/"frames.npz")["ids"]
        self.source_index={int(frame):i for i,frame in enumerate(self.source_ids)}
        self.images=sorted((self.work/"frames").glob("*.jpg"))
        self.masks=MaskFrames(self.work,self.source_ids,self.path.parent/"object-input-arrays")
        calibration=next((self.work/p for p in ["cameras.npz","camera-intrinsics.npz"] if (self.work/p).exists()),None)
        self.source_k=NpzArrays(calibration,self.path.parent/"object-input-arrays/calibration")["intrinsic"] if calibration else None
        # Enforce the same semantic ids as the geometry, including older jobs.
        original={(r["mask_key"],r["source_instance"]):r["id"] for r in self.metadata["instances"]}
        self.remap={r["id"]:original[(r["mask_key"],r["source_instance"])] for r in self.masks.instances}

    def frame(self,index):
        import cv2
        frame=super().frame(index)
        prepared=self.source_index[int(frame["ids"])]
        rgb=cv2.cvtColor(cv2.imread(str(self.images[prepared])),cv2.COLOR_BGR2RGB)
        labels=self.masks.frame(prepared)
        remapped=np.zeros_like(labels);remapped[labels==65535]=65535
        for current,original in self.remap.items():
            remapped[labels==current]=original
        if "preprocessing_affine" in self:
            affine=np.asarray(self["preprocessing_affine"][index])
            k=np.linalg.inv(affine)@frame["k"]
        elif self.source_k is not None:
            k=np.asarray(self.source_k)
            affine=frame["k"]@np.linalg.inv(k)
        else:
            raise ValueError("Original-to-geometry pixel mapping is missing")
        h,w=rgb.shape[:2]
        inverse=np.linalg.inv(affine)[:2]
        depth=cv2.warpAffine(frame["depth"],inverse,(w,h),flags=cv2.INTER_NEAREST)
        valid=cv2.warpAffine(frame["valid"].astype("uint8"),inverse,(w,h),flags=cv2.INTER_NEAREST)>0
        frame.update(images=rgb,labels=remapped,depth=depth,valid=valid,k=k,points=backproject(depth,k,frame["poses"]))
        return frame
