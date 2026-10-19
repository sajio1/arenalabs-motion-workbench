"""Exact video-frame access with sequential decoding for short forward steps."""
import cv2
import numpy as np

def read_at(capture,seconds):
    fps=capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0:
        capture.set(cv2.CAP_PROP_POS_MSEC,seconds*1000)
        return capture.read()
    target=round(seconds*fps)
    position=capture.get(cv2.CAP_PROP_POS_FRAMES)
    gap=target-round(position)
    if abs(position-round(position))<.01 and 0<=gap<=8:
        for _ in range(gap):
            if not capture.grab():return False,None
    else:
        capture.set(cv2.CAP_PROP_POS_MSEC,seconds*1000)
    return capture.read()
