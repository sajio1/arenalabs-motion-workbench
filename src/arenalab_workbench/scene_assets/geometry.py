"""Static-overlap registration and global similarity pose graph for chunked maps."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def windows(n, size=32, overlap=8):
    if n < 1 or size < 2 or not 0 < overlap < size:
        raise ValueError("Require frames > 0 and 0 < overlap < window size")
    result = []
    start = 0
    while start < n:
        stop = min(start + size, n)
        result.append(np.arange(start, stop))
        if stop == n:
            break
        start = stop - overlap
    return result


def apply_transform(transform, points):
    return points @ transform[:3, :3].T + transform[:3, 3]


def backproject(depth, intrinsic, pose):
    """Z-depth to source world with unchanged declared optical calibration."""
    y,x=np.indices(depth.shape)
    rays=np.stack([x,y,np.ones_like(x)],axis=-1)@np.linalg.inv(intrinsic).T
    return apply_transform(pose,rays*depth[...,None])


def similarity(source, target, metric=True):
    source, target = np.asarray(source), np.asarray(target)
    if len(source) < 12 or source.shape != target.shape:
        raise ValueError("Insufficient static overlap for registration")
    if np.linalg.matrix_rank(source - source.mean(0)) < 2:
        raise ValueError("Degenerate static overlap")
    x, y = source - source.mean(0), target - target.mean(0)
    u, singular, vt = np.linalg.svd(x.T @ y / len(x))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ np.diag(sign) @ u.T
    scale = 1. if metric else (singular * sign).sum() / np.mean(np.sum(x*x, axis=1))
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = target.mean(0) - scale * rotation @ source.mean(0)
    return matrix


def robust_similarity(source, target, metric=True, threshold=.05):
    """Fixed shared policy, robust to depth outliers; never fits task trajectories."""
    finite = np.isfinite(source).all(-1) & np.isfinite(target).all(-1)
    source, target = source[finite], target[finite]
    rng = np.random.default_rng(0)
    best = np.zeros(len(source), dtype=bool)
    if len(source) < 12:
        raise ValueError("Insufficient static overlap")
    for _ in range(96):
        subset = rng.choice(len(source), 12, replace=False)
        try:
            matrix = similarity(source[subset], target[subset], metric)
        except ValueError:
            continue
        valid = np.linalg.norm(apply_transform(matrix, source)-target, axis=1) < threshold
        if valid.sum() > best.sum():
            best = valid
    if best.sum() < 12 or best.mean() < .25:
        raise ValueError("Static chunk overlap registration failed; keep separate components")
    matrix = similarity(source[best], target[best], metric)
    return matrix, dict(inliers=int(best.sum()), total=len(best), rmse_m=float(np.sqrt(np.mean(
        np.sum((apply_transform(matrix, source[best])-target[best])**2, axis=-1)))))


def _decode(value, metric):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(value[:3]).as_matrix() * (1 if metric else np.exp(value[6]))
    matrix[:3, 3] = value[3:6]
    return matrix


def pose_graph(initial, edges, metric=True):
    """edges=(a,b,points_a,points_b), including verified loop closures; node0 fixed."""
    if len(initial) == 1:
        return initial, {"success": True, "edges": 0}
    connected = {0}
    for _ in initial:
        for a, b, _, _ in edges:
            if a in connected or b in connected:
                connected.update([a, b])
    if len(connected) != len(initial):
        raise ValueError("Disconnected pose graph; do not merge unrelated maps")
    width = 6 if metric else 7
    encoded = []
    for matrix in initial[1:]:
        scale = np.cbrt(np.linalg.det(matrix[:3, :3]))
        value = np.r_[Rotation.from_matrix(matrix[:3, :3]/scale).as_rotvec(), matrix[:3, 3]]
        encoded.append(value if metric else np.r_[value, np.log(scale)])
    def matrices(x):
        return [initial[0], *[_decode(v, metric) for v in x.reshape(-1, width)]]
    def residual(x):
        transforms = matrices(x)
        return np.concatenate([(apply_transform(transforms[a], pa)-apply_transform(transforms[b], pb)).ravel()
                               for a, b, pa, pb in edges])
    # A dense Jacobian grows quadratically with video length. Each residual
    # touches only its two chunk nodes; sparse finite differences preserve that.
    from scipy.sparse import lil_matrix
    sparsity = lil_matrix((sum(len(pa)*3 for _,_,pa,_ in edges), (len(initial)-1)*width), dtype=bool)
    offset = 0
    for a,b,pa,_ in edges:
        for node in (a,b):
            if node:
                sparsity[offset:offset+len(pa)*3,(node-1)*width:node*width] = True
        offset += len(pa)*3
    solved = least_squares(residual, np.asarray(encoded).ravel(), loss="soft_l1", f_scale=.02, max_nfev=120,
                           jac_sparsity=sparsity.tocsr())
    if not solved.success:
        raise ValueError("Pose graph did not converge")
    return matrices(solved.x), {"success": True, "edges": len(edges), "residual_rmse_m": float(np.sqrt(np.mean(solved.fun**2)))}
