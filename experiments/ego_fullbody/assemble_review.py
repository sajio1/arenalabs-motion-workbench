"""Make synchronized source / human / H2 / A3 reviews and decode every frame."""
from pathlib import Path
import subprocess,json,os
import cv2,numpy as np
root=Path(os.environ['ARENA_REVIEW_DIR'])
ffmpeg=os.environ['ARENA_FFMPEG']
report={}
for clip in ['jacket','crouch']:
 folder=root/clip
 paths=[folder/'00-nymeria-source.mp4',folder/'01-smplx.mp4',folder/'h2/02-h2-wuji.mp4',folder/'a3/03-a3-wuji.mp4']
 out=folder/'04-synchronized-comparison.mp4'
 filters=[]
 for i in range(4):filters.append(f'[{i}:v]setpts=PTS-STARTPTS,scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]')
 filters.append('[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|960_0|0_540|960_540:shortest=1[out]')
 args=[ffmpeg,'-y','-loglevel','error']
 for p in paths:args+=['-i',str(p)]
 subprocess.run(args+['-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-crf','20','-preset','fast','-r','30','-movflags','+faststart',str(out)],check=True)
 tiles=[]
 for p in paths+[out]:
  cap=cv2.VideoCapture(str(p));fps=cap.get(cv2.CAP_PROP_FPS);count=0;frames=[]
  while True:
   ok,im=cap.read()
   if not ok:break
   if count in [0,120,240,360]:frames.append(cv2.resize(im,(480,270)))
   count+=1
  cap.release();report[str(p.relative_to(root))]=dict(decoded_frames=count,fps=fps,duration_s=count/fps)
  if count!=450 or abs(fps-30)>1e-6:raise ValueError((p,count,fps))
  if p!=out:tiles.append(np.concatenate(frames,axis=1))
 cv2.imwrite(str(folder/'review-contact-sheet.jpg'),np.concatenate(tiles,axis=0))
(root/'video-validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
