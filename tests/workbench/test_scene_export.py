import os
import pytest
pytestmark = pytest.mark.skipif(os.name != "posix", reason="Scene worker uses POSIX file locks; tested on the Linux compute host")
"""CPU fixture: all visible instances survive export; source limbs are excluded."""
from pathlib import Path
import json,subprocess,sys,tempfile
import numpy as np


def test_scene_export():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);out=root/'public';out.mkdir();(root/'masks').mkdir();n=24;T=n//2;H=W=16
        def save(name,value):(root/name).write_text(json.dumps(value))
        save('manifest.json',dict(source=dict(frames=n),public=dict(id='fixture',title='fixture',dataset='Synthetic test',video=dict(url='/fixture.mp4',start_seconds=0)),destination=str(out),url='/fixture/',source_sha256='unchanged'))
        save('preparation.json',{});save('semantic.json',dict(objects=['container','tool'],task='Fixture',provenance='Fixture'));save('prompts.json',[['object-01','container'],['object-02','tool'],['human-hand','hand']]);save('depth.json',dict(source_trajectory_edited=False))
        K=np.array([[100.,0,480],[0,100.,270],[0,0,1]])
        k=np.diag([W/960,H/540,1])@K;pose=np.repeat(np.eye(4)[None],n,0);np.savez(root/'cameras.npz',poses=pose,intrinsic=K)
        np.savez(root/'depth.npz',depth=np.ones((T,H,W)),conf=np.ones((T,H,W)),images=np.zeros((T,H,W,3),dtype='u1'),intrinsics=np.repeat(k[None],T,0),extrinsics=pose[:T,:3],ids=np.arange(T)*2)
        mask=np.zeros((T,540,960),dtype='u2');mask[:,70:200,120:360]=1;mask[:,320:450,600:840]=2
        np.savez(root/'masks/object-01-masks.npz',masks=mask,frame_indices=np.arange(T))
        tool=np.zeros_like(mask);tool[:,70:200,600:840]=1;np.savez(root/'masks/object-02-masks.npz',masks=tool,frame_indices=np.arange(T))
        hand=np.zeros_like(mask);hand[:,380:,0:200]=1;np.savez(root/'masks/human-hand-masks.npz',masks=hand,frame_indices=np.arange(T))
        subprocess.run([sys.executable,'-m','arenalab_workbench.export_scene_cloud',str(root)],check=True,stdout=subprocess.DEVNULL)
        result=json.loads((out/'scene.json').read_text());audit=json.loads((out/'segmentation.json').read_text())
        assert len(audit['objects'])==3,audit
        assert len(result['segmented_instances']['frames'])==T
        # Each full-resolution source mask contributes a separate depth-bound instance.
        assert all(len(f['instances'])==3 for f in result['segmented_instances']['frames'])
        assert result['robots']==[] and result['objects']==[]
        assert 'spatial_memory' not in result and not (out/'background.bin').exists()
        assert 'surface_fusion' not in result['geometry']
        assert result['source_camera']['poses']==pose.tolist()
        assert result['geometry']['source_trajectory_edited'] is False
        # A missing mask frame must fail instead of silently rendering unlabeled depth.
        np.savez(root/'masks/human-hand-masks.npz',masks=hand[:-1],frame_indices=np.arange(T-1))
        rejected=subprocess.run([sys.executable,'-m','arenalab_workbench.export_scene_cloud',str(root)],capture_output=True,text=True)
        assert rejected.returncode!=0 and 'frame coverage mismatch' in rejected.stderr
        # Equal frame counts with different source timestamps must also fail.
        np.savez(root/'masks/human-hand-masks.npz',masks=hand,frame_indices=np.arange(T),source_frame_indices=np.arange(T)*2+1)
        rejected=subprocess.run([sys.executable,'-m','arenalab_workbench.export_scene_cloud',str(root)],capture_output=True,text=True)
        assert rejected.returncode!=0 and 'source timestamp mismatch' in rejected.stderr
        print('PASS: all category/instance boxes, frame alignment, unchanged source camera, no substituted robot or object meshes')
