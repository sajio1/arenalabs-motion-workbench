"""Package measured head-camera and bilateral EE observations for ArenaLabs.

Requires numpy. Video is never used here to fabricate missing trajectories.
See --help and the repository's docs/dataset-io.md for the tensor contract.
"""
import argparse
import json
from pathlib import Path
import subprocess
import zipfile
import io


def egodex_observations(path):
    """Lossless axis conversion of existing observations, with no body inference."""
    import h5py
    import numpy as np
    c = np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    with h5py.File(path) as source:
        camera=source['transforms/camera'][()].copy()
        camera[:,:3,:3]=c@camera[:,:3,:3]
        camera[:,:3,3]=camera[:,:3,3]@c.T
        n=len(camera);points=[];confidence=[];orientation=[]
        for side in ['left','right']:
            names=[side+'Hand']+[side+'Thumb'+s for s in ['Knuckle','IntermediateBase','IntermediateTip','Tip']]
            for finger in ['Index','Middle','Ring','Little']:
                names += [side+finger+'Finger'+s for s in ['Knuckle','IntermediateBase','IntermediateTip','Tip']]
            points.append(np.stack([source['transforms/'+name][:,:3,3] for name in names],1)@c.T)
            confidence.append(np.stack([source['confidences/'+name][()] for name in names],1))
            orientation.append(c@source['transforms/'+side+'Hand'][:,:3,:3])
        points=np.stack(points,1)
        result=dict(timestamps_s=np.arange(n)/30,wrist_position=points[:,:,0],hand_points=points,hand_confidence=np.stack(confidence,1),camera_to_world=camera,ee_rotation=np.stack(orientation,1),intrinsic=source['camera/intrinsic'][()])
    stream=io.BytesIO();np.savez_compressed(stream,**result)
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--observations', type=Path, help='NPZ: timestamps_s, wrist_position, ee_rotation, camera_to_world, intrinsic; optional observed hand_points/hand_confidence or gripper_width_m')
    inputs.add_argument('--egodex-hdf5', type=Path, help='Convert original EgoDex head camera and observed 21-point hands (requires h5py)')
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--floor-z', type=float, required=True, help='Floor height in the same Z-up world, in metres')
    parser.add_argument('--end-effector', choices=['wuji', 'robotiq-2f-140', 'none'], required=True)
    parser.add_argument('--title', default='My recording')
    parser.add_argument('--video-start', type=float, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.egodex_hdf5 and args.end_effector!='wuji':
        parser.error('The native EgoDex adapter provides observed fingers; select wuji')
    if args.output.exists():
        parser.error('Output already exists; use a fresh filename')
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'json', str(args.video)]))['streams'][0]
    manifest = dict(schema='arenalabs.recording.v1', title=args.title, fps=30, units='m', world_up='z', end_effector=args.end_effector, floor_z_m=args.floor_z, image_size=[probe['width'], probe['height']], video='original.mp4', observations='observations.npz', video_start_seconds=args.video_start)
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr('recording.json', json.dumps(manifest, indent=2))
        if args.egodex_hdf5:
            archive.writestr('observations.npz',egodex_observations(args.egodex_hdf5))
        else:
            archive.write(args.observations, 'observations.npz')
        archive.write(args.video, 'original.mp4')
    print(str(args.output.resolve()))


if __name__ == '__main__':
    main()
