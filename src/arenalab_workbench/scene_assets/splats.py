"""Adaptive gsplat training with SAM static masks and metric depth regularization."""
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.spatial import cKDTree

from .formats import write_json,sha256
from .store import GeometryStore


def save_ply(path, means, scales, quaternions, opacity_logits, rgb):
    from plyfile import PlyData, PlyElement
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.empty(len(means), dtype=[(name, "f4") for name in names])
    values = np.column_stack([means, np.zeros_like(means), (rgb-.5)/.28209479177387814,
                              opacity_logits, scales, quaternions])
    for index, name in enumerate(names):
        data[name] = values[:, index]
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))


def run(geometry_path, output, converter, iterations=7000, max_gaussians=80000):
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    store = GeometryStore(geometry_path)
    # Bounded reservoir: even an hour-long source keeps at most max_gaussians
    # seeds. Training transfers one randomly selected frame at a time to CUDA.
    rng = np.random.default_rng(0)
    priorities,points,colors = np.empty(0),np.empty((0,3)),np.empty((0,3))
    for i in range(len(store)):
        frame = store.frame(i)
        mask = (frame["valid"] & (frame["labels"]==0))[::4,::4]
        p,c = frame["points"][::4,::4][mask],frame["images"][::4,::4][mask]/255.
        priorities = np.r_[priorities,rng.random(len(p))]
        points,colors = np.concatenate([points,p]),np.concatenate([colors,c])
        if len(points)>max_gaussians:
            selected = np.argpartition(priorities,-max_gaussians)[-max_gaussians:]
            priorities,points,colors = priorities[selected],points[selected],colors[selected]
    _,selected = np.unique(np.floor(points/.01).astype("int64"),axis=0,return_index=True)
    points,colors = points[selected],colors[selected]
    if len(points) < 20:
        raise ValueError("Not enough static geometry for 3DGS")
    neighbours = cKDTree(points).query(points, k=2)[0][:,1]
    initial_scale = np.clip(neighbours, .003, .08)
    def parameter(value):
        return torch.nn.Parameter(torch.tensor(value, dtype=torch.float32, device="cuda"))
    params = torch.nn.ParameterDict(dict(means=parameter(points),
        scales=parameter(np.log(np.repeat(initial_scale[:,None],3,axis=1))),
        quats=parameter(np.tile([1.,0,0,0],(len(points),1))),opacities=parameter(np.full(len(points),-1.)),
        colors=parameter(np.log(np.clip(colors,.001,.999)/(1-np.clip(colors,.001,.999))))))
    scene_scale = max(float(np.linalg.norm(np.quantile(points,.95,axis=0)-np.quantile(points,.05,axis=0)))/2,.01)
    rates = dict(means=.00016*scene_scale,scales=.005,quats=.001,opacities=.05,colors=.0025)
    optimizers = {key:torch.optim.Adam([value],lr=rates[key],eps=1e-15) for key,value in params.items()}
    strategy = DefaultStrategy(refine_stop_iter=min(15000,max(501,iterations-1000)))
    strategy.check_sanity(params,optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    h,w = store["depth"].shape[1:]
    losses = []
    started = time.monotonic()
    for step in range(iterations):
        frame = store.frame(int(rng.integers(len(store))))
        mask = torch.as_tensor(frame["valid"] & (frame["labels"]==0),device="cuda")
        if not mask.any():
            continue
        target = torch.as_tensor(frame["images"],device="cuda",dtype=torch.float32)/255
        depth = torch.as_tensor(frame["depth"],device="cuda",dtype=torch.float32)
        render,alpha,info = rasterization(means=params["means"],quats=F.normalize(params["quats"],dim=-1),
            scales=params["scales"].exp(),opacities=params["opacities"].sigmoid(),colors=params["colors"].sigmoid(),
            viewmats=torch.as_tensor(np.linalg.inv(frame["poses"])[None],device="cuda",dtype=torch.float32),
            Ks=torch.as_tensor(frame["k"][None],device="cuda",dtype=torch.float32),width=w,height=h,packed=True,render_mode="RGB+ED")
        strategy.step_pre_backward(params,optimizers,strategy_state,step,info)
        image = render[0,...,:3]
        # Masked SSIM uses only windows containing entirely static valid pixels.
        a,b = image.permute(2,0,1)[None],target.permute(2,0,1)[None]
        pool = lambda value:F.avg_pool2d(value,7,stride=1,padding=3)
        ma,mb = pool(a),pool(b)
        ssim = ((2*ma*mb+.01**2)*(2*(pool(a*b)-ma*mb)+.03**2))/((ma.square()+mb.square()+.01**2)*
                (pool(a*a)-ma.square()+pool(b*b)-mb.square()+.03**2))
        static_window = pool(mask[None,None].float())>.999
        structure = (1-ssim)[static_window.expand_as(ssim)].mean() if static_window.any() else image.sum()*0
        photometric = (image[mask]-target[mask]).abs().mean()
        depth_mask = mask & (alpha[0,...,0]>.5) & (depth>0)
        depth_loss = ((render[0,...,3][depth_mask]-depth[depth_mask]).abs()/depth[depth_mask].clamp_min(.05)).mean() if depth_mask.any() else image.sum()*0
        loss = .8*photometric+.2*structure+.1*depth_loss
        if not torch.isfinite(loss):
            raise ValueError("3DGS loss became nonfinite; retain failed result")
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step();optimizer.zero_grad(set_to_none=True)
        strategy.step_post_backward(params,optimizers,strategy_state,step,info,packed=True)
        if step % 100 == 0 or step == iterations-1:
            losses.append(dict(step=step, loss=float(loss.detach()),gaussians=len(params["means"])))
            print(losses[-1],flush=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    def numpy(value):
        return value.detach().cpu().numpy()
    ply = output / "static.ply"
    save_ply(ply,numpy(params["means"]),numpy(params["scales"]),numpy(F.normalize(params["quats"],dim=-1)),
             numpy(params["opacities"]),numpy(params["colors"].sigmoid()))
    subprocess.run([str(converter),str(ply),str(output/"static.spz")],check=True)
    write_json(output/"static.json",dict(backend="gsplat DefaultStrategy",iterations=iterations,gaussians=len(params["means"]),
               seconds=time.monotonic()-started,losses=losses,static_mask="valid & labels==0",
               world_frame="Same numeric source-world frame as geometry camera poses",
               quaternion_order="wxyz",spz_codec="Official paired ply_to_spz / spz_to_ply defaults",
               coverage="observed viewpoints; unseen surfaces not verified",adaptive_densification=True,
               losses_used=["static masked L1","static-window SSIM","MapAnything metric depth"],
               memory_policy="Bounded seed reservoir and one CUDA training frame"))
    write_json(output/"input-receipt.json",dict(geometry=str(geometry_path),
        reconstruction_sha256=sha256(Path(geometry_path).with_suffix(".json")),
        code_sha256=sha256(__file__)))
