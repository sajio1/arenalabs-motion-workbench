"""Generate only selected foreground assets, conditioned on estimated scene depth."""
from pathlib import Path
import os, sys, json, time, argparse
os.environ.setdefault('CUDA_VISIBLE_DEVICES','3')
os.environ['LIDRA_SKIP_INIT']='true'
ROOT=Path('/home/ubuntu/arenalab-experiments/manipulation4d')
sys.path.insert(0,str(ROOT/'sam3d/notebook'))
import cv2, numpy as np, torch, trimesh
from inference import Inference, SceneVisualizer
from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera
from pytorch3d.transforms import Transform3d

parser=argparse.ArgumentParser();parser.add_argument('--mask-background',action='store_true');parser.add_argument('--objects-json',type=Path,default=ROOT/'cup-lids/selected-objects.json');args=parser.parse_args()
rows=json.loads(args.objects_json.read_text())
model=Inference(str(ROOT/'sam3d-weights/checkpoints/pipeline.yaml'),compile=False)
for row in rows:
    start=time.time();folder=Path(row['folder'])
    image=cv2.cvtColor(cv2.imread(str(folder/'image.png')),cv2.COLOR_BGR2RGB)
    mask=cv2.imread(str(folder/'mask.png'),0)>0
    pointmap=torch.from_numpy(np.load(folder/'camera-pointmap.npy')).cuda()
    if args.mask_background:
        image=image.copy();image[~mask]=127
        folder=folder/'isolated';folder.mkdir(exist_ok=True)
    conversion=Transform3d().rotate(camera_to_pytorch3d_camera(device='cuda').rotation).to('cuda')
    pointmap=conversion.transform_points(pointmap)
    output=model(image,mask,seed=42,pointmap=pointmap)
    mesh=output.get('glb')
    if mesh is None:raise RuntimeError('Object model did not produce a mesh')
    if isinstance(mesh,trimesh.Scene):mesh=mesh.to_mesh()
    mesh.export(folder/'canonical.glb')
    # Undo the official GLB display-axis rotation before applying model pose.
    canonical=np.asarray(mesh.vertices)@np.array([[1,0,0],[0,0,-1],[0,1,0]]).T
    points=SceneVisualizer.object_pointcloud(points_local=torch.as_tensor(canonical,dtype=torch.float32,device='cuda')[None],quat_l2c=output['rotation'],trans_l2c=output['translation'],scale_l2c=output['scale']).points_list()[0]
    camera=conversion.inverse().transform_points(points[None])[0].detach().cpu().numpy()
    extrinsics=np.asarray(row['camera_extrinsics'])
    mesh.vertices=(camera-extrinsics[:3,3])@extrinsics[:3,:3]
    if not np.isfinite(mesh.vertices).all():raise ValueError('Nonfinite reconstructed mesh')
    mesh.export(folder/'world.glb')
    pose={key:output[key].detach().cpu().numpy().tolist() for key in ['rotation','translation','scale']}
    report=dict(model='SAM 3D Objects',segmentation='SAM 3.1',background_masked=args.mask_background,seed=42,seconds=time.time()-start,keyframe=row['source_frame'],vertices=len(mesh.vertices),faces=len(mesh.faces),bounds_world_m=mesh.bounds.tolist(),watertight=bool(mesh.is_watertight),visual_kind=mesh.visual.kind,pose=pose,depth='DA3 estimate conditioned on original camera trajectory',hidden_surfaces='generated hypothesis, not observed scan',source_trajectory_edited=False)
    report['pointmap_depth_calibration']=row.get('pointmap_depth_calibration')
    (folder/'reconstruction.json').write_text(json.dumps(report,indent=2))
    print('OBJECT_RECONSTRUCTED',row['kind'],json.dumps(report),flush=True)
print('RECONSTRUCTION_COMPLETE',flush=True)
