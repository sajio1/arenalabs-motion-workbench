"""Attach only proven source-media mappings; never infer sync from a title."""
import json
from pathlib import Path
import cv2,h5py,numpy as np
from greenwich_pipelines.viewer.export import array_json
from greenwich_pipelines.pipelines.nymeria.input import read_source

def main():
    repo=Path(__file__).resolve().parents[1];path=repo/'configs/viewer.local.json'
    config=json.loads(path.read_text());count={};sources={}
    originals=Path('C:/Users/Eric/Documents/Codex/2026-09-06/ni-q/outputs/distinct-action-corpus/originals')
    paired=list(originals.rglob('*.mp4'))
    curated=Path('C:/Users/Eric/Documents/Codex/2026-09-06/ni-q/work/a3-100-drive-inputs/calibrated_sources')
    syncs=[(p,json.loads(p.read_text())) for p in curated.glob('*/source_ego.sync.json')]
    for clip in config['clips']:
        clip.pop('video',None)
        if clip['page']=='manipulation':
            source=Path(clip['source']);candidates=[source.with_suffix('.mp4')]
            with h5py.File(source) as data:
                ref=np.asarray(data['transforms/camera'][:min(clip['frames'],10)])
                # Copies and cached prefixes require an actual pose match to an original.
                if not candidates[0].exists():candidates=paired
                for video in candidates:
                    hdf=video.with_suffix('.hdf5')
                    if not video.exists() or not hdf.exists():continue
                    with h5py.File(hdf) as original:
                        if 'transforms/camera' not in original:continue
                        values=original['transforms/camera']
                        if len(values)<clip['frames'] or not np.array_equal(values[:len(ref)],ref):continue
                        # Verify all frames of the prefix, including both wrist channels.
                        if not all(k in original and np.array_equal(data[k][:clip['frames']],original[k][:clip['frames']]) for k in ['transforms/camera','transforms/leftHand','transforms/rightHand']):continue
                        full_frames=len(values)
                    capture=cv2.VideoCapture(str(video));frames=round(capture.get(cv2.CAP_PROP_FRAME_COUNT));fps=capture.get(cv2.CAP_PROP_FPS);capture.release()
                    if frames!=full_frames or abs(fps-30)>.1:continue
                    clip['video']=dict(path=str(video),start_seconds=0,sync_method='Paired recording; frame counts and source poses verified')
                    break
        elif clip['page']=='localmanipulation' and clip.get('source_window'):
            sequence=clip['metadata']['Sequence'];source=clip['source']
            for sync_path,sync in syncs:
                if sync.get('source_id')!=sequence or not sync.get('synchronization_verified'):continue
                folder=sync_path.parent;mapping=np.load(folder/'clock_mapping.npz')
                if source not in sources:sources[source]=read_source(source)['sample_timestamp_us']
                clock=sources[source];a=clip['source_window']['start_frame'];b=clip['source_window']['end_frame'];query=clock[a:b]*1000
                ref=mapping['timecode_ns'];device=mapping['device_ns']
                if query[0]<ref[0]-2e7 or query[-1]>ref[-1]+2e7:continue
                times=(np.interp(query,ref,device)-device[0])/1e9
                clip['video']=dict(path=str(folder/'source_ego.mp4'),frame_times=np.maximum(times,0).tolist(),start_seconds=0,sync_method='Verified head device timestamps mapped from Xsens timecode',timestamp_error_max_ms=sync['timestamp_error_max_ms'])
                break
        if clip.get('video'):clip['video']['browser_transcode']=True
        if clip.get('video'):count[clip['page']]=count.get(clip['page'],0)+1
    array_json(path,config);array_json(Path(config['storage'])/'source-video-report.json',dict(available=count,policy='Only verified mappings are served; unmatched source videos remain unavailable'))
    print(json.dumps(count))

if __name__=='__main__':main()
