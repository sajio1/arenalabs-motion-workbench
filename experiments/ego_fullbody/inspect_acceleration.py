from pathlib import Path
import numpy as np,json
root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)))
for label in ['h2','a3']:
 for version,path in [('old',next((root/(label+'-result')).rglob('motion.npz'))),('new',root/'observed-refinement'/label/'motion.npz')]:
  d=np.load(path);a=np.diff(d['q'],n=2,axis=0)*900;names=d['joint_names'];flat=np.argsort(np.abs(a).ravel())[-5:]
  print(label,version,[(int(t+1),str(names[j]),int(s),float(a[t,j,s])) for t,j,s in [np.unravel_index(i,a.shape) for i in flat]])
  if version=='new':
   for t,j,s in [np.unravel_index(flat[-1],a.shape)]:
    print('peak q deg',np.rad2deg(d['q'][max(0,t-2):t+5,j,s]).round(2).tolist())
    print('actual knee deg',np.rad2deg(d['knee_flexion_rad'][max(0,t-2):t+5,0]).round(2).tolist())
    print('target knee deg',np.rad2deg(d['source_knee_flexion_rad'][max(0,t-2):t+5,0]).round(2).tolist())
