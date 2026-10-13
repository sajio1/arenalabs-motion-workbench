"""Report the foot-placement tradeoff of stronger torso tracking, not force contact."""
from pathlib import Path
import os,sys,json,numpy as np
from scipy.ndimage import gaussian_filter1d
root=Path(os.environ.get('ARENA_EGO_WORKDIR',Path(__file__).resolve().parent));label=sys.argv[1]
row=json.loads((root/(label+'-robot.json')).read_text())[0];C=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
human=np.load(root/'smplx_motion.npz')['joints']@C.T;target=human[:,[7,8]];speed=np.linalg.norm(np.diff(target,axis=0)*30,axis=-1);low=target[:,:,1]<np.percentile(target[:,:,1],15,axis=0)+.025;mask=(speed<.1)&low[1:]&low[:-1]
def stats(x):
 x=np.asarray(x);return dict(mean=float(x.mean()),p95=float(np.percentile(x,95)),max=float(x.max())) if x.size else None
out={}
for version,folder in [('before','observed-refinement-v5'),('after','observed-refinement')]:
 d=np.load(root/folder/label/'motion.npz');names=list(d['joint_names']);ids=[names.index(row['endpoints'][s+'_ankle']) for s in ['left','right']];feet=d['world_position_cm'][:,ids]/100
 out[version]=dict(horizontal_foot_prior_error_cm=stats(np.linalg.norm((feet-target)[...,[0,2]],axis=-1)*100),ankle_speed_during_low_slow_human_frames_cm_s=stats(np.linalg.norm(np.diff(feet,axis=0)[...,[0,2]],axis=-1)[mask]*3000),low_slow_frame_pairs=int(mask.sum()),ankle_separation_min_cm=float(np.linalg.norm(feet[:,0]-feet[:,1],axis=-1).min()*100),scope='Ankle proxy, not material-contact slip; human foot positions are reconstructed priors')
(root/'observed-refinement'/label/'foot-prior-validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
