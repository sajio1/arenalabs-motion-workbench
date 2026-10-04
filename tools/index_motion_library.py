"""Index every local recording, reading headers rather than converting recordings.

Discovery is explicit (--workspace, optional --roots). No downloads, renamed
actions, quality filtering, or source modifications. Duplicate recording copies
are collapsed; complete source records and historical excerpts stay distinct.
"""
import argparse, hashlib, json, re, subprocess
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
from greenwich_pipelines.viewer.export import array_json

def bvh_header(path):
    joints=[];frames=None;step=None
    with path.open(errors='replace') as f:
        for line in f:
            text=line.strip()
            if text.startswith(('ROOT ','JOINT ')):joints.append(text.split(maxsplit=1)[1])
            if text.startswith('Frames:'):frames=int(text.split(':')[1])
            if text.startswith('Frame Time:'):step=float(text.split(':')[1]);break
    if not frames or not step:raise ValueError('Missing BVH frame header')
    return frames,step,joints

def discover(roots):
    found=set()
    for root in roots:
        result=subprocess.run(['rg','--files','--no-ignore',str(root),'-g','*.bvh','-g','xdata.npz','-g','*.hdf5','-g','!**/node_modules/**','-g','!**/.cache/**','-g','!**/.git/**','-g','!**/.venv/**'],capture_output=True,text=True,encoding='utf-8')
        if result.returncode not in (0,1):raise RuntimeError(result.stderr)
        found.update(Path(p).resolve() for p in result.stdout.splitlines())
    return sorted(found)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,required=True);parser.add_argument('--roots',type=Path,nargs='*',default=[]);args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];config_path=repo/'configs/viewer.local.json';config=json.loads(config_path.read_text());storage=Path(config['storage'])
    roots=[args.workspace.resolve(),*[p.resolve() for p in args.roots]];files=discover(roots)
    # Retain actual cached excerpts for immediate replay, explicitly identified.
    samples=[c for c in config['clips'] if c.get('cache') and not c.get('catalog_recording')]
    for clip in samples:
        clip['title']=Path(clip['source']).name if clip['page']=='localmanipulation' else clip['source_name'];clip['family']='Cached excerpts';clip.setdefault('metadata',{})['Scope']='Historical excerpt; not the full recording'
    records={};unreadable=[]
    bind=args.workspace/'2026-09-08/https-huggingface-co-datasets-bones-studio/outputs/seed-100-contact/source/soma_base_skel_minimal.bvh'
    for path in files:
        if any(part.lower() in {'fixtures','tests'} for part in path.parts):continue
        try:
            if path.suffix=='.bvh':
                if not re.search(r'__A\d+',path.stem):continue
                native,step,names=bvh_header(path);page='locomotion';key=('soma',path.name)
                frames=int(np.floor((native-1)*step*30+1e-6))+1;duration=(frames-1)/30
                family=next((label for token,label in [('dance','Dance'),('jump','Jumping'),('walk','Walking'),('run','Running'),('jog','Running'),('crouch','Transitions'),('crawl','Crawling'),('stretch','Exercise'),('lunge','Exercise'),('idle','Idle')] if token in path.name.lower()),'Other')
                metadata={'Source frames':native,'Source frame rate':round(1/step,3),'Skeleton joints':len(names),'Joint names':names,'Category origin':'Filename-derived','Scope':'Full recording','Validation':'Deferred until conversion'}
                fmt='SOMA · BVH';title=path.name
            elif path.name=='xdata.npz':
                with np.load(path,allow_pickle=False) as d:
                    required={'segment_tXYZ','segment_qWXYZ','tpose_segment_tXYZ','tpose_segment_qWXYZ','timestamps_us','foot_contacts'}
                    if not required.issubset(d.files):raise ValueError('Missing Xsens channels: '+', '.join(sorted(required-set(d.files))))
                    timestamps=d['timestamps_us'];native=len(timestamps)
                    from greenwich_pipelines.pipelines.nymeria.xsens_clock import recording_clock
                    clock,report=recording_clock(timestamps)
                page='localmanipulation';sequence=path.parent.name if path.parent.name!='body' else path.parent.parent.name;key=('nymeria',sequence)
                frames=int(np.floor((clock[-1]-clock[0])*30/1e6))+1;duration=(frames-1)/30;family='Recordings';fmt='Nymeria · Xsens';title=sequence+'/xdata.npz'
                metadata={'Sequence':sequence,'Source frames':native,'Input endpoints':['Left wrist','Right wrist','Left ankle','Right ankle'],'Coordinates':'Z-up / m','Clock':report,'Scope':'Full recording','Window policy':'Independent 900-frame windows; no seam-continuity claim'}
            else:
                with h5py.File(path,'r') as f:
                    required=['camera','hip','leftArm','rightArm']+[side+joint for side in ['left','right'] for joint in ['Hand','MiddleFingerKnuckle','IndexFingerKnuckle','LittleFingerKnuckle']]
                    if 'transforms/camera' not in f:continue
                    missing=[name for name in required if 'transforms/'+name not in f]
                    if missing:raise ValueError('Missing EgoDex transforms: '+', '.join(missing))
                    frames=f['transforms/camera'].shape[0];native=frames
                    if any(f['transforms/'+name].shape!=(frames,4,4) for name in required):raise ValueError('Inconsistent transform shapes')
                    channels=list(f['transforms'].keys());attrs={str(k):str(v) for k,v in f.attrs.items()}
                if frames<3:raise ValueError('Recording shorter than 3 frames')
                page='manipulation';duration=(frames-1)/30;task=path.parent.name;family=task;fmt='EgoDex · HDF5';title=task+'/'+path.name
                key=('egodex',task,path.name,frames)
                metadata={'Task':task,'Transform channels':channels,'Reference':'Wrists and head','Robot base':'Fixed','Scope':'Full recording',**({'File attributes':attrs} if attrs else {})}
            # Prefer complete versions over partial copies with the same source ID.
            if key in records and records[key]['metadata']['Native frames']>=native:continue
            rid='record_'+hashlib.sha256(repr(key).encode()).hexdigest()[:18]
            source=path
            if page=='locomotion':
                source=storage/'inputs'/f'{rid}.json';array_json(source,dict(schema='alphamotion.sources.v1',sources=[dict(id=rid,format='soma',path=str(path),options=dict(bind_path=str(bind)))]))
            metadata['Native frames']=native
            records[key]=dict(id=rid,page=page,family=family,title=title,source_name=title,source_format=fmt,description='Full local recording · On-demand conversion',frames=frames,fps=30,duration=duration,source=str(source),version='Current pipeline · Quality not evaluated',metadata=metadata,catalog_recording=True)
        except Exception as error:unreadable.append(dict(file=str(path),reason=str(error)))
    config['clips']=samples+list(records.values());config['library_roots']=[str(p) for p in roots]
    array_json(config_path,config);array_json(storage/'library-index-report.json',dict(roots=config['library_roots'],discovered=len(files),recordings=dict(Counter(c['page'] for c in records.values())),cached_excerpts=len(samples),unreadable=unreadable))
    print(json.dumps(dict(recordings=dict(Counter(c['page'] for c in records.values())),cached_excerpts=len(samples),unreadable=len(unreadable))))

if __name__=='__main__':main()
