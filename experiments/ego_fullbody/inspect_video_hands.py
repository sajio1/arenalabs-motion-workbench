"""Independent image-based finger evidence audit; never ground-truth hand labels."""
from pathlib import Path
import sys,json
import cv2,numpy as np,mediapipe as mp
root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)))
video=Path(sys.argv[1]);cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
options=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(root/'hand_landmarker.task')),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5)
points=[];pixels=[];scores=[];previews=[];frame=0
with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
 while True:
  ok,im=cap.read()
  if not ok:break
  h,w=im.shape[:2]
  result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),round(frame/fps*1000))
  p=np.full((2,21,3),np.nan);xy=np.full((2,21,2),np.nan);sc=np.zeros(2)
  for landmarks,world,categories in zip(result.hand_landmarks,result.hand_world_landmarks,result.handedness):
   side=0 if categories[0].category_name=='Left' else 1
   if categories[0].score<sc[side]:continue
   p[side]=[[q.x,q.y,q.z] for q in world];xy[side]=[[q.x,q.y] for q in landmarks];sc[side]=categories[0].score
   for q in landmarks:cv2.circle(im,(round(q.x*w),round(q.y*h)),3,(0,255,0) if side==0 else (0,180,255),-1)
   cv2.putText(im,categories[0].category_name,(round(landmarks[0].x*w),round(landmarks[0].y*h)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
  if frame%50==0 or (sc.any() and len(previews)<12):
   if len(previews)<12:previews.append(cv2.resize(im,(480,480)))
  points.append(p);pixels.append(xy);scores.append(sc);frame+=1
cap.release();points=np.array(points);scores=np.array(scores)
np.savez_compressed(root/'video_hand_landmarks.npz',points=points,image_xy=pixels,handedness_confidence=scores,fps=fps,provenance='MediaPipe monocular image estimates; not measured Nymeria finger trajectories')
report=dict(frames=len(points),fps=fps,detected_frames_per_hand=(scores>0).sum(0).tolist(),bilateral_frames=int((scores>0).all(1).sum()),provenance='RGB-derived estimate; handedness needs visual review; no interpolation of missing intervals')
(root/'video_hand_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
while len(previews)%3:previews.append(np.zeros_like(previews[0]))
cv2.imwrite(str(root/'video-hand-detections.jpg'),np.vstack([np.hstack(previews[k:k+3]) for k in range(0,len(previews),3)]))
