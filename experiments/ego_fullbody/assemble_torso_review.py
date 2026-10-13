"""Review body-prior additions against the immutable v3 videos."""
from pathlib import Path
import os,json,subprocess
import numpy as np,cv2,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(os.environ['ARENA_REVIEW_DIR']);before=Path(os.environ['ARENA_BEFORE_DIR']);ffmpeg=os.environ['ARENA_FFMPEG']
videos={};summary={};fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for ci,clip in enumerate(['jacket','crouch']):
 summary[clip]={}
 paths=[before/clip/'h2/02-h2-wuji.mp4',root/clip/'h2/02-h2-wuji.mp4',before/clip/'a3/03-a3-wuji.mp4',root/clip/'a3/03-a3-wuji.mp4']
 args=[ffmpeg,'-y','-loglevel','error'];filters=[]
 titles=['H2 - BEFORE','H2 - WITH BODY PRIOR','A3 - BEFORE','A3 - WITH BODY PRIOR']
 for i,(p,title) in enumerate(zip(paths,titles)):
  args+=['-i',str(p)]
  filters.append(f"[{i}:v]setpts=PTS-STARTPTS,scale=960:540,setsar=1,drawbox=x=0:y=0:w=iw:h=42:color=black:t=fill,drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='{title}':fontcolor=white:fontsize=25:x=20:y=7[v{i}]")
 filters.append('[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|960_0|0_540|960_540:shortest=1[out]')
 out=root/clip/'05-before-after-torso.mp4'
 subprocess.run(args+['-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-crf','20','-preset','fast','-r','30','-movflags','+faststart',str(out)],check=True)
 for p in [paths[1],paths[3],out]:
  cap=cv2.VideoCapture(str(p));fps=cap.get(cv2.CAP_PROP_FPS);count=0;images=[]
  while True:
   ok,im=cap.read()
   if not ok:break
   if count in [0,120,240,360] and p==out:images.append(cv2.resize(im,(960,540)))
   count+=1
  cap.release();videos[str(p.relative_to(root))]=dict(frames=count,fps=fps)
  assert count==450 and abs(fps-30)<1e-6,(p,count,fps)
  if images:cv2.imwrite(str(root/clip/'review-contact-sheet.jpg'),np.concatenate(images,axis=0))
 for ri,robot in enumerate(['h2','a3']):
  r=json.loads((root/clip/robot/'torso-validation.json').read_text());native=json.loads((root/clip/robot/'native-validation.json').read_text())['after'];summary[clip][robot]={'torso':r,'native':{k:v for k,v in native.items() if k not in ['foot_floor_m','wrist_errors_cm','wrist_errors_deg']}}
  t=np.arange(len(r['source_pitch_deg']))/30;ax=axes[ci,ri]
  ax.plot(t,r['source_pitch_deg'],label='SMPL-X target',color='#666666')
  ax.plot(t,r['versions']['before']['pitch_deg'],label='Before',color='#c58632')
  ax.plot(t,r['versions']['after']['pitch_deg'],label='With body prior',color='#258469')
  ax.set(title=f'{clip} / {robot.upper()}',xlabel='Seconds',ylabel='Forward torso tilt (degrees)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(root/'torso-tracking.png',dpi=150)
(root/'video-validation.json').write_text(json.dumps(videos,indent=2))
(root/'comparison-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(videos,indent=2))
