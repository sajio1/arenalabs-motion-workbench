import {api} from '../platform-api';
import type {LibraryClip,SceneAsset} from './types';

const escape=(s:unknown)=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
function dialog(title:string,content:string){
 const element=document.createElement('dialog');element.className='wb-data-dialog';
 element.innerHTML=`<header><h2>${title}</h2><button data-close aria-label="Close dialog">×</button></header>${content}`;
 document.body.append(element);element.showModal();element.querySelector('[data-close]')!.addEventListener('click',()=>element.close());element.onclose=()=>element.remove();return element;
}
async function response(r:Response){const data=await r.json();if(!r.ok)throw new Error(data.detail||'Request failed');return data;}

export function showRecordingUpload(onImported:(clip:LibraryClip)=>void){
 const element=dialog('Upload a recording',`<p>Upload your original video with synchronized head-camera and left/right hand or gripper trajectories. Your recordings stay separate from the preset library.</p><p>30 Hz · metres · Z-up · MP4 + observations NPZ + recording.json</p><p><a href="/api/v1/recordings/template" download>Download packaging script</a></p><p>Include observed 21-point hands for Wuji, or gripper widths for a gripper. A plain video alone does not provide calibrated operation trajectories.</p><input type="file" accept=".zip" aria-label="Recording ZIP"/><progress max="100" value="0" hidden></progress><p role="status">Choose a recording ZIP, up to 2 GiB.</p><footer><button data-submit>Validate & upload</button></footer>`);
 const file=element.querySelector('input')!,button=element.querySelector<HTMLButtonElement>('[data-submit]')!,status=element.querySelector('[role=status]')!,progress=element.querySelector('progress')!;
 button.onclick=()=>{const selected=file.files?.[0];if(!selected){status.textContent='Choose a recording ZIP first.';return;}if(selected.size>2*1024**3){status.textContent='Upload limit is 2 GiB.';return;}button.disabled=true;progress.hidden=false;
 const request=new XMLHttpRequest();request.open('POST','/api/v1/recordings/import');request.upload.onprogress=e=>{if(e.lengthComputable){progress.value=e.loaded/e.total*100;status.textContent=progress.value<100?`Uploading · ${Math.round(progress.value)}%`:'Validating coordinates, timestamps and video coverage…';}};
 request.onload=()=>{button.disabled=false;try{const data=JSON.parse(request.responseText);if(request.status>=400)throw new Error(data.detail||'Import failed');onImported(data);element.close();}catch(e){status.textContent=(e as Error).message;}};
 request.onerror=()=>{button.disabled=false;status.textContent='Upload interrupted. Your existing recordings are unchanged.';};const form=new FormData();form.append('file',selected);request.send(form);
 };
}

export async function downloadEpisode(scene:SceneAsset,robot:string){
 const element=dialog('Download dataset',`<p>${escape(scene.title)}</p><p>LeRobot v3 episode + lossless ArenaLabs extensions</p><ul><li>Whole-body joint states, action references and synchronized camera videos</li><li>Available point clouds, segmentation masks, 3D boxes and semantic labels</li><li>Original FOV and available robot FOV composite, wrist cameras and camera calibration</li><li>Available contact estimates and SMPL-H motion, with provenance and checksums</li></ul><p>Unavailable modalities remain explicitly marked. Contact estimates are not measured pressure. Actions are kinematic references.</p><p role="status">Preparing export…</p><footer><a data-download hidden>Download ZIP</a></footer>`);
 const status=element.querySelector('[role=status]')!;
 try{const selected=scene.robots.find(row=>row.id===robot);
 if(!selected?.playback_url)throw new Error('Dataset export requires a generated robot trajectory. Other completed outputs remain viewable in this scene.');
 let job=await api.start({workflow:'export',recording_id:scene.id,robot_id:robot,trajectory_scale:selected.trajectory_scale,output_modalities:scene.output_modalities??scene.generation_status?.output_modalities});
 while(element.isConnected&&(job.status==='queued'||job.status==='running')){status.textContent=job.phase||'Packaging saved results; no new inference…';await new Promise(r=>setTimeout(r,1500));job=await api.job(job.id);}
 if(!element.isConnected)return;if(job.status!=='completed')throw new Error(job.error||'Export failed');
 status.textContent=`Ready · ${(job.bytes/1024**2).toFixed(1)} MiB · ${job.frames} frames`;const link=element.querySelector<HTMLAnchorElement>('[data-download]')!;link.href=job.download_url;link.download='';link.textContent='Download complete dataset ZIP';link.hidden=false;
 }catch(e){status.textContent=(e as Error).message;}
}
