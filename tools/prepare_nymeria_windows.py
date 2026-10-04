"""Build 500 distinct approximately 15-second Nymeria excerpts."""
import argparse,json
import numpy as np
from collections import Counter,deque
from pathlib import Path
from greenwich_pipelines.viewer.export import array_json

def stamp(frame):
    seconds=frame//30
    return f'{seconds//60:02d}:{seconds%60:02d}'

def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=500);p.add_argument('--seconds',type=int,default=15);p.add_argument('--stride',type=int,default=15);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];path=repo/'configs/viewer.local.json';config=json.loads(path.read_text())
    recordings=[c for c in config['clips'] if c['page']=='localmanipulation' and c.get('catalog_recording')]
    if not recordings:
        manifest=Path(config['storage'])/'nymeria-window-recordings.json'
        recordings=json.loads(manifest.read_text())
    width=a.seconds*30;stride=a.stride*30
    # Prefer existing verified, curated 15-second action excerpts before generic windows.
    curated=[]
    video_root=Path('C:/Users/Eric/Documents/Codex/2026-09-06/ni-q/work/a3-100-drive-inputs/calibrated_sources')
    from greenwich_pipelines.pipelines.nymeria.input import read_source
    for sync_path in sorted(video_root.glob('*/source_ego.sync.json')):
        sync=json.loads(sync_path.read_text())
        if not sync.get('synchronization_verified') or sync.get('output_frames')!=width:continue
        record=next((r for r in recordings if r['metadata']['Sequence']==sync['source_id']),None)
        if record is None:continue
        mapping=np.load(sync_path.parent/'clock_mapping.npz')
        clock=read_source(record['source'])['sample_timestamp_us']
        start=int(np.argmin(abs(clock-mapping['timecode_ns'][0]/1000)))
        if start+width<=len(clock):curated.append((record,start))
    queues=[deque((r,start) for start in range(0,r['frames']-width+1,stride)) for r in sorted(recordings,key=lambda r:r['id'])]
    chosen=curated[:a.count]
    while len(chosen)<a.count and any(queues):
        for queue in queues:
            if queue and len(chosen)<a.count:
                candidate=queue.popleft();r,start=candidate
                if not any(old['id']==r['id'] and abs(begin-start)<width for old,begin in chosen):chosen.append(candidate)
    if len(chosen)<a.count:raise ValueError(f'Only {len(chosen)} distinct complete windows available')
    windows=[]
    for record,start in chosen:
        end=start+width;sequence=record['metadata']['Sequence'];title=f'xdata.npz [{stamp(start)}–{stamp(end)}]'
        windows.append(dict(id=f"{record['id']}_f{start:07d}",page='localmanipulation',family=sequence,title=title,
            source_name=record['source_name'],source_format='Nymeria · Xsens window',description=f'{a.seconds}-second window · On-demand conversion',
            frames=width,fps=30,duration=(width-1)/30,source=record['source'],source_window=dict(start_frame=start,end_frame=end),
            version='Current pipeline · Quality not evaluated',metadata={'Sequence':sequence,'Start':stamp(start),'End':stamp(end),'Window seconds':a.seconds,'Stride seconds':a.stride,'Overlapping':a.stride<a.seconds,'Source recording seconds':round(record['duration'],3),'Source clock':record['metadata']['Clock'],'Source frame range':[start,end],'Input endpoints':['Left wrist','Right wrist','Left ankle','Right ankle'],'Scope':'Time window from a local recording','Boundary policy':'Existing curated excerpt' if (record,start) in curated else 'Fixed-duration window; complete activity not guaranteed'}))
    keep=[c for c in config['clips'] if c['page']!='localmanipulation' or c.get('cache')]
    config['clips']=keep+windows
    array_json(path,config);array_json(Path(config['storage'])/'nymeria-window-recordings.json',recordings)
    report=dict(windows=len(windows),window_seconds=a.seconds,stride_seconds=a.stride,overlapping=a.stride<a.seconds,recordings=len(recordings),per_recording=dict(Counter(w['family'] for w in windows)),selection='Verified curated excerpts first, then non-overlapping round-robin windows; no quality filtering',curated_excerpts=len(curated),unique_ranges=len({(w['source'],w['source_window']['start_frame']) for w in windows}))
    array_json(Path(config['storage'])/'nymeria-window-report.json',report);print(json.dumps(report))

if __name__=='__main__':main()
