from pathlib import Path
import os,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(os.environ['ARENA_REVIEW_DIR'])
fig,axes=plt.subplots(4,2,figsize=(13,13),layout='constrained')
for row,(clip,robot) in enumerate([(c,r) for c in ['jacket','crouch'] for r in ['h2','a3']]):
 folder=root/clip/robot
 knees=json.loads((folder/'knee-diagnosis.json').read_text());native=json.loads((folder/'native-validation.json').read_text())
 for stage,color,label in [('smplx','#767676','SMPL-X prior'),('old_refinement','#cf8130','Before'),('new_refinement','#21916f','After')]:
  y=np.asarray(knees[stage]['timeseries_deg']);t=np.arange(len(y))/30
  axes[row,0].plot(t,y.mean(1),c=color,label=label,lw=1.5)
 axes[row,0].set(title=f'{clip} / {robot.upper()} — mean bilateral knee flexion',ylabel='Degrees')
 error=np.asarray(native['after']['wrist_errors_cm'])
 axes[row,1].plot(t,error[:,0],label='Left wrist',c='#347fbd');axes[row,1].plot(t,error[:,1],label='Right wrist',c='#d68843')
 axes[row,1].axhline(1,c='gray',ls='--',label='User tolerance: 1 cm')
 axes[row,1].set(title='Final native FK vs original observed wrists',ylabel='cm',ylim=(0,1.1))
 for ax in axes[row]:ax.grid(alpha=.2);ax.set_xlabel('Seconds');ax.legend(fontsize=8,loc='upper right')
fig.savefig(root/'tracking-review.png',dpi=150)
