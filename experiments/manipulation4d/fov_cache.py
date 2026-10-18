"""Lossless source-frame inpainting cache, shared across robot embodiments."""
from pathlib import Path
import hashlib, os, uuid
import cv2
import numpy as np
from gpu_scheduler import resource_lock

from video_io import read_at

def inpaint(work, image, mask):
    digest=hashlib.sha256(b'cv2-telea-radius4-v1'+cv2.__version__.encode())
    digest.update(str((image.shape,image.dtype,mask.shape,mask.dtype)).encode())
    digest.update(image.tobytes());digest.update(mask.tobytes())
    key=digest.hexdigest();path=Path(work)/'fov-background'/key[:2]/(key+'.npy')
    with resource_lock('fov-background:'+str(path)):
        if path.exists():return np.load(path,allow_pickle=False)
        result=cv2.inpaint(image,mask,4,cv2.INPAINT_TELEA)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        with temporary.open('wb') as stream:np.save(stream,result,allow_pickle=False)
        os.replace(temporary,path)
        return result
