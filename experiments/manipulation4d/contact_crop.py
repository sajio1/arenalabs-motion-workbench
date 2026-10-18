"""Keep projected hand crops inside the image without stretching visible evidence."""
import numpy as np
from PIL import Image


def hand_crop(image, uv, valid):
    height, width = image.shape[:2]
    inside = valid & np.isfinite(uv).all(axis=1)
    inside &= (uv[:, 0] >= 0) & (uv[:, 0] < width)
    inside &= (uv[:, 1] >= 0) & (uv[:, 1] < height)
    if inside.sum() < 5:
        return image, False
    xy = uv[inside]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    margin = max(25., float(np.max(hi - lo)) * .35)
    x0, y0 = np.maximum(np.floor(lo - margin), [0, 0]).astype(int)
    x1, y1 = np.minimum(np.ceil(hi + margin), [width, height]).astype(int)
    return image[y0:y1, x0:x1], True


def letterbox(image):
    """Padding preserves pixels and prevents pathological processor aspect ratios."""
    side = max(image.size)
    result = Image.new(image.mode, (side, side), 0)
    result.paste(image, ((side-image.width)//2, (side-image.height)//2))
    return result
