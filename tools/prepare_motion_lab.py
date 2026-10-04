"""Index existing local sample results for Motion Lab. No model inference here.

python tools/prepare_motion_lab.py --workspace C:/Users/Eric/Documents/Codex
Local paths and exports are ignored by Git; source data is never modified.
"""
import argparse,json
from pathlib import Path
import numpy as np
from greenwich_pipelines.viewer.export import array_json,robot_model,motion_payload
from greenwich_motion_sdk.source_batch import read_sources,sample_clip

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];ws=a.workspace.resolve();storage=repo/'.cache/motion-lab';storage.mkdir(parents=True,exist_ok=True)
    humanoids=json.loads((repo/'configs/robots.local.json').read_text(encoding='utf-8'))
    gento=json.loads((repo/'configs/robots.gento.json').read_text(encoding='utf-8'))
    rows=[]
    for r in humanoids:
        rows.append(dict(id=r['name'],label=r['label'],kind='Humanoid',pages=['locomotion','localmanipulation'],pipeline_row=r,status='ready',conversion_ready=True))
    for r in gento:
        meta_path=(repo/'configs'/r['metadata']).resolve();meta=json.loads(meta_path.read_text())
        actual=dict(r,metadata=str(meta_path),xml=str((meta_path.parent/meta['model']).resolve()))
        rows.append(dict(id=r['body'],label='Gento '+r['body'].title(),kind='Wheeled',pages=['manipulation'],pipeline_row=actual,status='ready',conversion_ready=True))
    for row in rows:
        print('ROBOT',row['id'],flush=True);robot_model(row['pipeline_row'],storage/'robots'/row['id'])
        row.update(model_url=f"/assets/robots/{row['id']}/model.glb",neutral_url=f"/assets/robots/{row['id']}/neutral.json")
    clips=[]
    def cache(clip,row,path,version):
        if not path.exists():return
        print('MOTION',clip['id'],row['id'],flush=True)
        payload=motion_payload(row['pipeline_row'],path,dict(locomotion='fullbody',localmanipulation='nymeria',manipulation='gento')[clip['page']])
        payload.update(clip_id=clip['id'],robot_id=row['id'],version=version)
        dest=storage/'clips'/clip['id']/f"{row['id']}.json";array_json(dest,payload)
        clip.setdefault('cache',{})[row['id']]=f"/assets/clips/{clip['id']}/{row['id']}.json"
    # Existing V3 output contains exactly the four source endpoint targets.
    ny=ws/'2026-09-24/ni-b/work/nymeria/v3_posture_results'
    labels={'unseen_pan':('Cooking','锅具操作'),'unseen_prep':('Preparation','准备食材'),'unseen_switch':('Transitions','操作切换'),'crouch_measure':('Reaching','下蹲取物')}
    for key,(family,title) in labels.items():
        folder=ny/key;first=folder/'unitree_g1_29dof/result.npz'
        if not first.exists():continue
        with np.load(first,allow_pickle=False) as d:
            pos=d['input_positions_yup_cm'];rot=d['target_rotation_yup'];flags=np.zeros((len(pos),4),bool)
            if 'contact_point_mask' in d:
                masks=d['contact_point_mask'];flags=np.stack((masks[:,0,0],masks[:,0,-1],masks[:,1,0],masks[:,1,-1]),-1)
            source=storage/'inputs'/f'nymeria_{key}.npz';source.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(source,position_yup_cm=pos,semantic_rotation_yup=rot,foot_contacts=flags,fps=np.array(30.))
        clip=dict(id='ny_'+key,page='localmanipulation',family=family,title=title,description='Nymeria 四端点记录 · 已有 V3 精修结果',frames=len(pos),fps=30,duration=(len(pos)-1)/30,
            source_name=key,source_format='Nymeria · 双腕 / 双踝',source=str(source),version='历史 V3 · 未整体验收')
        preview=storage/'clips'/clip['id']/'source.json';array_json(preview,dict(fps=30,frames=len(pos),targets=(pos/100).tolist(),parents=[],skeleton=[],positions=[],quaternions=[],geom_ids=[],duration=clip['duration']))
        clip['preview']=f"/assets/clips/{clip['id']}/source.json"
        for row in rows[:2]:cache(clip,row,folder/row['pipeline_row']['body']/'result.npz',clip['version'])
        clips.append(clip)
    # Gento clips retain the production source path recorded in the raw prior.
    gr=ws/'2026-09-24/ni-b/work/gento';labels={'wipe':('Cleaning','擦拭台面','arm_priority45_stable'),'fold':('Folding','折叠布料','wrist_alignment_5s')}
    for key,(family,title,version) in labels.items():
        result=gr/version/'skye'/key/'result.npz'
        if not result.exists():continue
        raw=next((x for x in [gr/'head_hands_face_raw/skye'/key/'prime_original.npz',gr/'raw/skye'/key/'prime_original.npz'] if x.exists()),None)
        if raw is None:
            candidates=list(gr.glob(f'*/skye/{key}/prime_original.npz'));raw=candidates[0] if candidates else None
        if raw is None:continue
        with np.load(raw,allow_pickle=False) as d:original=Path(str(d['source']))
        with np.load(result,allow_pickle=False) as d:T=len(d['qpos']);targets=d['targets']
        # Explicit prefix sample, preserving original transforms and timestamps.
        import h5py
        source=storage/'inputs'/f'ego_{key}.hdf5'
        with h5py.File(original,'r') as src,h5py.File(source,'w') as dst:
            def copy_dataset(name,obj):
                if isinstance(obj,h5py.Dataset):
                    v=obj[:T] if obj.ndim and obj.shape[0]>=T else obj[()];dst.create_dataset(name,data=v)
            src.visititems(copy_dataset)
        clip=dict(id='ego_'+key,page='manipulation',family=family,title=title,description='EgoDex 双腕 + 头参考 · 固定底盘上身适配',frames=T,fps=30,duration=(T-1)/30,
            source_name=original.name,source_format='EgoDex · HDF5',source=str(source),version=f'历史 {version} · 未整体验收')
        for row in rows[2:]:cache(clip,row,gr/version/row['id']/key/'result.npz',clip['version'])
        clips.append(clip)
    # A short migrated result plus several full SOMA source previews.
    source_root=ws/'2026-09-08/https-huggingface-co-datasets-bones-studio/outputs/seed-100-contact/source'
    candidates=[('walk','Walking','起步行走','baby_full_diaper_walk_ff_270_start_R_001__A461.bvh',.3),
        ('jog','Running','慢跑停步','baby_full_diaper_jog_ff_360_stop_R_002__A460.bvh',8.),
        ('turn','Walking','转向起步','baby_full_diaper_turn_walk_ff_360_start_R_001__A462.bvh',8.),
        ('crouch','Transitions','下蹲起步','crouch_ff_start_270_001__A148.bvh',8.)]
    for key,family,title,name,seconds in candidates:
        bvh=source_root/name
        if not bvh.exists():continue
        source=storage/'inputs'/f'soma_{key}.json'
        array_json(source,dict(schema='alphamotion.sources.v1',sources=[dict(id=key,format='soma',path=str(bvh),options=dict(bind_path=str(source_root/'soma_base_skel_minimal.bvh')))]))
        desc=read_sources(source)[0];motion=desc['_clip'];take,_=sample_clip(motion,30,seconds)
        pos=motion.world_position_cm[take]
        clip=dict(id='soma_'+key,page='locomotion',family=family,title=title+(' · 短样本' if key=='walk' else ''),description='SOMA BVH 全身输入 · 30 Hz',frames=len(take),fps=30,duration=(len(take)-1)/30,
            source_name=name,source_format='SOMA · BVH',source=str(source),max_seconds=seconds,version='当前统一管线 · 未做动作质量验收')
        array_json(storage/'clips'/clip['id']/'source.json',dict(fps=30,frames=len(take),duration=clip['duration'],skeleton=(pos/100).tolist(),parents=motion.parents.tolist(),targets=[],positions=[],quaternions=[],geom_ids=[]))
        clip['preview']=f"/assets/clips/{clip['id']}/source.json"
        if key=='walk':
            for row in rows[:2]:cache(clip,row,repo/'runs/fullbody_smoke/native/walk'/row['id']/'motion.npz',clip['version'])
        clips.append(clip)
    config=dict(schema='greenwich.viewer.v1',storage=str(storage),runtime=str(repo/'configs/runtime.local.json'),robots=rows,clips=clips)
    array_json(repo/'configs/viewer.local.json',config)
    print('CATALOG',len(clips),'clips',len(rows),'robots',flush=True)

if __name__=='__main__':main()
