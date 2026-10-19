import {showRecordingUpload,downloadEpisode} from './dataset-io';
import {showTrajectoryHeight} from './trajectory-height';
import './feedback-dismiss.mjs';
import {demoNavigation,demoBrand} from '../demo-nav';
import {robotChoices,motionGroup} from '../state.js';
import './style.css';
import {Workspace} from './stage';
import type {SceneAsset,LibraryClip,Motion} from './types';
import {currentJointState} from './joint-states.mjs';

document.body.classList.add('workbench4d');document.title='ArenaLabs · Manipulation';
const ENCODER_POLICY='human-six-pelvis-head-hands-feet-v1';
const $=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;
const escape=(s:unknown)=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
document.getElementById('app')!.innerHTML=`
<header class="wb-header arena-header">${demoBrand()}${demoNavigation('manipulation')}<div class="wb-source"><button id="library-open">Data library <span id="library-count"></span><span>⌄</span></button><button id="recording-upload">Upload data</button><strong id="clip-name">Loading scene…</strong></div><span class="wb-mode">4D WORKSPACE</span></header>
<main class="wb-layout">
<aside class="wb-left">
 <section class="wb-panel"><header><span>Original FOV</span><div><small id="source-kind">EgoDex</small><button id="trajectory-height-open" hidden>Trajectory height</button></div></header><video id="source-video" muted playsinline preload="auto"></video></section>
 <section class="wb-panel"><header><span>Robot arm & hand view</span><small id="patch-label">Source camera</small></header><div class="wb-video-box"><video id="patch-video" muted playsinline preload="auto" hidden></video><div id="patch-empty" class="wb-unavailable">Preparing the source-camera composite</div></div></section>
 <section class="wb-panel wb-joints"><header><span>Joint output</span><div class="wb-joint-downloads"><button id="dataset-download" disabled>Download dataset ↓</button></div></header><div class="wb-state-meta"><strong id="joint-state-count">No joint states</strong><small id="joint-state-frame">—</small></div><details id="joint-states" open><summary>All joint states <span id="joint-unit">rad</span></summary><input id="joint-filter" type="search" placeholder="Filter joints…" aria-label="Filter joint states" disabled/><div class="wb-state-table"><table aria-label="Whole-body joint states"><thead><tr><th>Joint name</th><th>Motion signal</th></tr></thead><tbody id="joint-state-rows"></tbody></table><p id="joint-state-empty">Select a recording with robot output.</p></div><button id="joint-copy" aria-label="Copy current frame joint state" disabled>Copy current frame</button><p id="joint-state-signals"></p></details><details id="joint-trace"><summary>Single-joint trace</summary><select id="joint-select" aria-label="Select joint"></select><div class="wb-joint-value"><strong id="joint-value">—</strong><small>Current joint position</small></div><canvas id="joint-chart"></canvas><p id="joint-note"></p></details><details class="wb-hand-check"><summary>Finger motion verification</summary><p id="hand-audit-summary">Hand articulation check unavailable</p><select id="finger-select" aria-label="Verify finger"></select><div class="hand-legend"><span>Source bend</span><span>Wuji bend</span></div><canvas id="finger-chart"></canvas><p id="finger-stat"></p></details></section>
</aside>
<section class="wb-center"><div id="wb-viewport"></div><div id="source-preview" class="wb-source-preview" hidden><video id="preview-video" muted playsinline></video><p id="scene-generation-status" role="status">Preparing scene generation…</p><button id="scene-retry" hidden>Retry scene generation</button></div><div class="wb-scene-tools"><button id="focus-task">Task view</button><button id="focus-body">Whole robot</button><button id="focus-hands">Hands</button><button id="focus-objects">Objects</button><button id="orbit">Orbit</button><details class="wb-display"><summary>Display</summary><div>${[['cloud','Point cloud',true],['boxes','3D boxes',true],['segments','Segmentation',true],['traces','Future 20 frames',true],['robot','Robot',true]].map(([key,label,checked])=>`<label><input type="checkbox" data-layer="${key}" ${checked?'checked':''}/> ${label}</label>`).join('')}</div></details></div>
 <div class="wb-scene-label"><span class="wb-live-dot"></span><span id="scene-status">Loading scene assets</span></div><div id="cloud-status" class="wb-map-status"></div><div id="object-tags" class="wb-object-tags"></div>
 <div id="generation-feedback" class="wb-generation" role="status" aria-live="polite" hidden><span class="wb-generation-spinner" aria-hidden="true"></span><div><strong id="generation-title"></strong><p id="generation-detail"></p></div></div><div class="wb-embodiment"><button id="robot-open"><small>ROBOT</small><strong id="robot-name">Choose robot</strong><span>⌄</span></button><span class="wb-join">+</span><button id="hand-open"><small>END EFFECTOR</small><strong id="hand-name">Choose hand</strong><span>⌄</span></button><div id="embodiment-bar" hidden></div></div>
</section>
<aside class="wb-right">
 <section class="wb-panel"><header><span>End-effector error</span><small id="reference-label">PALM REFERENCE</small></header><div class="wb-errors"><div><small>LEFT</small><strong id="left-error">—</strong><span id="left-angle">—</span></div><div><small>RIGHT</small><strong id="right-error">—</strong><span id="right-angle">—</span></div></div><canvas id="error-chart"></canvas><p id="reference-note">Palm-to-source error · excludes scene and fingertip error</p><p id="fingertip-error"></p><p id="scene-depth-check"></p></section>
 <section class="wb-panel wb-touch"><header><span id="contact-title">Hand contact / tactile</span><small id="tactile-kind">NO SENSOR STREAM</small></header><label class="wb-contact-mode"><select id="contact-mode" aria-label="Finger display"><option value="proximity">Fingertip proximity</option><option value="video">Video contact estimate</option></select></label><div id="contact-hands" class="wb-hands"></div><p id="tactile-note">No tactile measurements in this recording.</p><p id="contact-evidence"></p></section>
 <section class="wb-panel"><header><span>Wrist FOV</span><small id="wrist-status">IF AVAILABLE</small></header><div id="wrist-videos" class="wb-wrist-videos"><span>No wrist-camera stream in this source</span></div></section>
 <section class="wb-panel wb-semantic"><header><span>Scene information</span><small id="semantic-origin">SOURCE LABELS</small></header><p id="task-text"></p><div id="semantic-tags"></div><div id="semantic-phases"></div><details><summary>Data & reconstruction</summary><div id="provenance"></div></details></section>
</aside>
</main>
<footer class="wb-transport"><button id="play" aria-label="Play or pause">Ⅱ</button><button id="step-back" aria-label="Previous frame">‹</button><span id="time">0.00 / 0.00</span><input id="timeline" type="range" min="0" max="1" step="0.0333333333333333" value="0" aria-label="Shared timeline"/><button id="step-next" aria-label="Next frame">›</button><select id="speed" aria-label="Playback speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><small id="frame-info">30 Hz</small><span class="wb-geometry-note">Estimated geometry · Kinematic playback</span></footer>
<aside id="library" class="wb-library" hidden><header><div><small>DATA LIBRARY</small><h2>Choose a motion</h2></div><button id="library-close" aria-label="Close library">×</button></header><div class="wb-datasets">${[['EgoDex','Ego (manipulation)'],['Nymeria','Ego (locomanipulation)'],['UMI','UMI'],['Real robot','Real robot'],['Uploaded','My uploads']].map(([value,label])=>`<button data-dataset="${value}">${label}</button>`).join('')}</div><input id="library-search" placeholder="Search tasks, files…" aria-label="Search motions"/><div id="library-list"></div><section id="library-meta"></section></aside>
<div id="wb-loading">Loading synchronized workspace…</div><div id="wb-notice" role="status" hidden></div>`;

const stage=new Workspace($('wb-viewport'));stage.segments=true;let asset:SceneAsset;let library:LibraryClip[]=[];let dataset='EgoDex';let currentRobot='';let selectedHand='';let time=0,playing=false,speed=1,last=performance.now();let loadingToken=0;let loadSerial=0;let previousChart=-1;let contact:{fps:number;likelihood:(number|null)[][][];model?:string;hand_contact?:{state:string;object:string;evidence:string}[][];observations?:{time:number;evidence:string;uncertainty:string}[]} | undefined;
const source=$<HTMLVideoElement>('source-video'),patch=$<HTMLVideoElement>('patch-video');
interface HandAudit {source_curl_deg:number[][][];robot_curl_deg:number[][][];finger_motion:{side:string;finger:string;source_curl_range_deg:number;robot_curl_range_deg:number;curl_correlation:number|null;maximum_joint_range_deg:number}[];}
let proximity:typeof contact;
let handAudit:HandAudit|undefined;
function notice(message:string){$('wb-notice').textContent=message;$('wb-notice').hidden=false;setTimeout(()=>$('wb-notice').hidden=true,7000);}
async function json<T>(url:string,options:RequestInit={}):Promise<T>{const r=await fetch(url,{cache:/((catalog|scene)\.json$|^\/api\/)/.test(url)?'no-store':'default',...options});if(!r.ok){const error=await r.json().catch(()=>({}));throw new Error(error.detail||`${r.status}: ${url}`);}return r.json();}
function chart(canvas:HTMLCanvasElement,values:number[][],frame:number,colors:string[],threshold?:number){
 const w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h||!values.length)return;const ratio=devicePixelRatio;canvas.width=w*ratio;canvas.height=h*ratio;const c=canvas.getContext('2d')!;c.scale(ratio,ratio);c.clearRect(0,0,w,h);
 const flattened=values.flat().filter(Number.isFinite);const min=Math.min(0,...flattened),max=Math.max(threshold??0,...flattened,min+.001);const y=(v:number)=>h-10-(v-min)/(max-min)*(h-20);
 c.strokeStyle='#263e30';c.lineWidth=1;for(let j=0;j<3;j++){const yy=10+j*(h-20)/2;c.beginPath();c.moveTo(0,yy);c.lineTo(w,yy);c.stroke();}
 if(threshold!==undefined){c.strokeStyle='#687945';c.setLineDash([3,4]);c.beginPath();c.moveTo(0,y(threshold));c.lineTo(w,y(threshold));c.stroke();c.setLineDash([]);}
 for(let side=0;side<values[0].length;side++){c.strokeStyle=colors[side%colors.length];c.lineWidth=1.5;c.beginPath();values.forEach((v,i)=>{const x=i/(values.length-1)*w;i?c.lineTo(x,y(v[side])):c.moveTo(x,y(v[side]));});c.stroke();}
 c.strokeStyle='#d4e4cb';c.beginPath();c.moveTo(frame/Math.max(1,values.length-1)*w,0);c.lineTo(frame/Math.max(1,values.length-1)*w,h);c.stroke();
}
let jointValueNodes:HTMLCanvasElement[]=[],jointMeters:HTMLElement[]=[],jointRanges:number[][]=[],lastJointFrame=-1;
function setupJointStates(motion?:Motion){
 lastJointFrame=-1;$<HTMLInputElement>('joint-filter').value='';$<HTMLInputElement>('joint-filter').disabled=!motion;
 for(const id of ['dataset-download','joint-copy'])$<HTMLButtonElement>(id).disabled=!motion;
 $('joint-unit').textContent=motion?.joints.unit||'—';$('joint-state-count').textContent=motion?`${motion.joints.names.length} joints · position`:'No joint states';$('joint-state-frame').textContent='—';$('joint-state-signals').textContent=motion?`4-second motion history · bars use each joint’s range in this clip. Hover for exact values; Dataset export retains native units.`:'';
 $('joint-state-empty').hidden=!!motion;$('joint-state-rows').innerHTML=motion?motion.joints.names.map((name,i)=>`<tr data-joint-row="${i}"><th><button data-joint-trace="${i}" title="${escape(name)}" aria-label="Show trace for ${escape(name)}">${escape(name)}</button></th><td><canvas class="wb-motor-signal" data-joint-position aria-label="Joint motion waveform"></canvas><div class="wb-motor-meter"><i data-joint-meter></i></div></td></tr>`).join(''):'';
 jointValueNodes=[...document.querySelectorAll<HTMLCanvasElement>('[data-joint-position]')];jointMeters=[...document.querySelectorAll<HTMLElement>('[data-joint-meter]')];jointRanges=motion?motion.joints.names.map((_,i)=>{const v=motion.joints.values.map(row=>row[i]).filter(Number.isFinite);return [Math.min(...v),Math.max(...v)];}):[];
 document.querySelectorAll<HTMLButtonElement>('[data-joint-trace]').forEach(button=>button.onclick=()=>{$<HTMLSelectElement>('joint-select').value=button.dataset.jointTrace!;$<HTMLDetailsElement>('joint-trace').open=true;previousChart=-1;});
}
$('dataset-download').onclick=()=>void downloadEpisode(asset,currentRobot);
$('recording-upload').onclick=()=>showRecordingUpload(clip=>{library.push(clip);dataset=clip.dataset;renderLibrary();previewClip(clip);});
$('joint-copy').onclick=()=>{const motion=stage.motion;if(!motion)return;const frame=Math.min(motion.frames-1,Math.floor(time*motion.fps));const state={source_id:asset.id,robot_id:currentRobot,time_basis:'clip_relative',position_unit:motion.joints.unit,...currentJointState(motion,frame)};void navigator.clipboard.writeText(JSON.stringify(state,null,2)).then(()=>notice('Current whole-body joint state copied'),()=>notice('Clipboard unavailable. Download JSON instead.'));};
$<HTMLInputElement>('joint-filter').oninput=()=>{lastJointFrame=-1;const query=$<HTMLInputElement>('joint-filter').value.toLowerCase().trim();let shown=0;document.querySelectorAll<HTMLTableRowElement>('[data-joint-row]').forEach(row=>{row.hidden=!row.textContent?.toLowerCase().includes(query);if(!row.hidden)shown++;});$('joint-state-count').textContent=stage.motion?`${query?shown+'/':''}${stage.motion.joints.names.length} joints · position`:'No joint states';};
document.querySelector('.wb-state-table')!.addEventListener('scroll',()=>lastJointFrame=-1);
$('joint-select').onchange=()=>previousChart=-1;$('joint-trace').ontoggle=()=>previousChart=-1;
async function chooseRobot(id:string){
 const row=asset.robots.find(r=>r.id===id);if(!row)return;const token=++loadingToken,resume=playing;$('scene-status').textContent='Loading robot and end effector…';playing=false;
 await stage.loadRobot(row);if(token!==loadingToken)return;currentRobot=id;$('robot-name').textContent=row.label;$('hand-name').textContent=row.hand==='none'?'No hand':row.hand==='wuji'?'Wuji':row.hand==='robotiq-2f-140'?'Robotiq 2F-140':row.hand;
 const auditUrl=asset.hand_motion_audit_urls?.[id]||asset.hand_motion_audit_url;handAudit=auditUrl?await json<HandAudit>(auditUrl):undefined;if(token!==loadingToken)return;$('finger-select').innerHTML=handAudit?handAudit.finger_motion.map((row,i)=>`<option value="${i}">${escape(row.side)} · ${escape(row.finger)}</option>`).join(''):'';
 document.querySelector<HTMLElement>('.wb-hand-check')!.hidden=row.hand!=='wuji';document.querySelector<HTMLElement>('.wb-touch')!.hidden=row.hand!=='wuji';
 const motion=stage.motion!;const housingReference=motion.tracking.reference?.startsWith('Fixed-housing')??false;const mountReference=motion.tracking.reference?.startsWith('Mounting-base')??false;const eeLabel=document.getElementById('reference-label');if(eeLabel)eeLabel.textContent=row.hand==='robotiq-2f-140'?(housingReference?'GRIPPER CENTER':mountReference?'GRIPPER BASE':'GRIPPER TCP'):row.hand==='none'?'WRIST REFERENCE':'PALM REFERENCE';$('reference-note').textContent=row.hand==='robotiq-2f-140'?`${housingReference?'Fixed-housing center':mountReference?'Mounting-base origin':'Jaw-pad midpoint'} to source reference · excludes scene and contact error`:row.hand==='none'?'Wrist-to-source error · no finger reconstruction':'Palm-to-source error · excludes scene and fingertip error';setupJointStates(motion);$('joint-select').innerHTML=motion.joints.names.map((name,i)=>`<option value="${i}">${escape(name)}</option>`).join('');$('joint-note').textContent=motion.joints.representation;
 const url=asset.patch_videos?.[id];patch.hidden=!url;$('patch-empty').hidden=!!url;if(url)patch.src=url;else{patch.pause();patch.removeAttribute('src');patch.load();}$('patch-label').textContent=url?((asset as unknown as {patch_kinds?:Record<string,string>}).patch_kinds?.[id]==='native_preview'?'ROBOT PREVIEW':((asset as unknown as {patch_kinds?:Record<string,string>}).patch_kinds?.[id]==='estimated_camera_composite'?'ESTIMATED CAMERA · SYNTHETIC':'SYNTHETIC · INPAINTED')):'UNAVAILABLE';if(!url)$('patch-empty').textContent=((asset as unknown as {patch_unavailable?:Record<string,string>}).patch_unavailable?.[id])||'No robot composite generated for this recording';
 $('scene-status').textContent=`${row.label} + ${row.hand==='none'?'No hand':row.hand==='wuji'?'Wuji':row.hand==='robotiq-2f-140'?'Robotiq 2F-140':row.hand} · Source world`;
 $('embodiment-bar').hidden=true;previousChart=-1;stage.focus('task');playing=resume;
}
async function load(url:string|SceneAsset,preservePlayback=false){
 const savedTime=time,savedPlaying=playing;
 $('generation-feedback').hidden=true;
 const request=++loadSerial;++loadingToken;
 $('wb-loading').hidden=false;setupJointStates();playing=false;time=0;$<HTMLInputElement>('timeline').value='0';
 try{const loaded=typeof url==='string'?await json<SceneAsset>(url):url;if(request!==loadSerial)return;const available=await json<RobotOption[]>('/api/4d/robots');if(request!==loadSerial)return;const allowed=new Set(available.map(r=>r.id));loaded.robots=loaded.robots.filter(r=>r.encoder_policy===ENCODER_POLICY&&(allowed.has(r.id)||allowed.has(r.registry_robot_id||''))).map(r=>({...r,label:available.find(a=>a.id===(r.registry_robot_id||r.id))?.label||r.label}));if(!asset||asset.id!==loaded.id)resetSelection();asset=loaded;dataset=asset.dataset;const ground=asset.geometry.ground as {height_m:number;estimated:boolean}|undefined;document.querySelector('.wb-geometry-note')!.textContent=ground?`Estimated floor ${ground.height_m.toFixed(3)} m · Kinematic playback`:'Estimated geometry · Kinematic playback';const depthCheck=asset.geometry.depth_scale_calibration as {after?:{median_mm:number;p95_mm:number}}|undefined;const reconstruction=asset.geometry.multiview_reconstruction as {triangulation_check?:{after:{median_mm:number;p95_mm:number}}}|undefined;const multiCheck=reconstruction?.triangulation_check?.after;$('scene-depth-check').textContent=multiCheck?`Sparse 3-view check: ${multiCheck.median_mm.toFixed(0)} mm median / ${multiCheck.p95_mm.toFixed(0)} mm p95 · limited baseline`:depthCheck?.after?`Scene depth check: ${depthCheck.after.median_mm.toFixed(0)} mm median / ${depthCheck.after.p95_mm.toFixed(0)} mm p95 (visible hand landmarks)`:'Scene depth alignment has not been measured';$('focus-objects').textContent='Objects';$('clip-name').textContent=asset.title;$('source-kind').textContent=asset.dataset;$('trajectory-height-open').hidden=false;source.src=asset.video.url;
  const task=typeof asset.semantic.task==='string'?asset.semantic.task.trim():'';$('task-text').textContent=!task||/^short visible action$/i.test(task)?'Action annotation unavailable':task;$('semantic-tags').innerHTML=asset.semantic.objects.map(o=>`<span>${escape(o)}</span>`).join('');$('tactile-note').textContent=asset.tactile.reason||'No tactile sensor stream';
  $('semantic-origin').textContent=asset.semantic.model?'QWEN · ESTIMATE':'SOURCE LABELS';
  $('semantic-phases').innerHTML=(asset.semantic.phases||[]).map((p,i)=>`<button data-phase="${i}"><small>${p.start_seconds.toFixed(1)}s</small>${escape(p.label)}</button>`).join('');
  $('semantic-phases').querySelectorAll<HTMLButtonElement>('[data-phase]').forEach(b=>b.onclick=()=>{time=asset.semantic.phases![Number(b.dataset.phase)].start_seconds;previousChart=-1;});
  const loadedContact=asset.tactile.url?await json<typeof contact>(asset.tactile.url):undefined;if(request!==loadSerial)return;contact=loadedContact;proximity=asset.tactile.proximity_url?await json<typeof proximity>(asset.tactile.proximity_url):undefined;if(request!==loadSerial)return;$<HTMLSelectElement>('contact-mode').value=proximity?'proximity':'video';handAudit=asset.hand_motion_audit_url?await json<HandAudit>(asset.hand_motion_audit_url):undefined;if(request!==loadSerial)return;$('finger-select').innerHTML=handAudit?handAudit.finger_motion.map((row,i)=>`<option value="${i}">${escape(row.side)} · ${escape(row.finger)}</option>`).join(''):'';$('hand-audit-summary').textContent=handAudit?'Actual Wuji FK compared with source finger bend. Palm movement excluded.':'Hand articulation check unavailable';$('tactile-kind').textContent=contact?(asset.tactile.status==='video_estimate'?'VIDEO ESTIMATE · 2 Hz':'VISUAL ESTIMATE'):'NO SENSOR STREAM';$('contact-title').textContent=contact?'Finger contact / tactile':'Hand contact / tactile';$('contact-evidence').textContent='';
  $('contact-hands').hidden=['UMI','Real robot'].includes(asset.dataset);$<HTMLSelectElement>('contact-mode').hidden=['UMI','Real robot'].includes(asset.dataset);$('contact-hands').innerHTML=['LEFT','RIGHT'].map((side,si)=>`<div><svg viewBox="0 0 100 115" aria-label="${side} estimated finger contact"><path data-hand-shape="${si}" d="M31 101L27 79L16 63Q12 54 20 53L31 64L30 28Q30 19 37 21L40 56L40 15Q40 6 47 9L49 54L51 12Q53 5 59 12L59 57L64 25Q67 17 73 24L69 68Q73 91 62 101Z" fill="#203d2a" stroke="#73936c" stroke-width="1.4"/>${[[22,60],[34,30],[45,16],[55,16],[68,30]].map(([x,y],finger)=>`<ellipse data-contact="${si}-${finger}" cx="${x}" cy="${y}" rx="5" ry="8" fill="#304e38"/>`).join('')}</svg><small id="contact-hand-${si}">${side}</small><div class="wb-digit-scores">${['Thumb','Index','Middle','Ring','Little'].map((finger,fi)=>`<div><span>${finger}</span><meter data-digit-meter="${si}-${fi}" min="0" max="1" value="0"></meter><b data-digit-score="${si}-${fi}">—</b></div>`).join('')}</div></div>`).join('');
  renderObjectTags();
  $('provenance').innerHTML=Object.entries(asset.geometry).map(([k,v])=>`<p><b>${escape(k.replaceAll('_',' '))}</b> ${escape(typeof v==='object'?JSON.stringify(v):v)}</p>`).join('')+`<p>${escape(asset.semantic.provenance)}</p>`;
  $('wrist-videos').innerHTML=asset.wrist_videos.length?asset.wrist_videos.map(v=>`<div><small>${escape(v.side)}</small><video muted playsinline src="${escape(v.url)}" data-start="${v.start_seconds}"></video></div>`).join(''):'<span>No wrist-camera stream in this source</span>';
  $<HTMLInputElement>('timeline').max=String(asset.duration);const preview=!asset.cloud&&!asset.robots.length;$('source-preview').hidden=!preview;$('wb-viewport').hidden=preview;document.querySelector<HTMLElement>('.wb-scene-tools')!.hidden=preview;
  const video=$<HTMLVideoElement>('preview-video');if(preview)video.src=asset.video.url;else{video.pause();video.removeAttribute('src');video.load();}
  const segmented=asset.segmented_instances;
  await stage.load({...asset,segmented_instances:segmented},typeof url==='string'?url:'',preservePlayback?savedTime:0);if(request!==loadSerial)return;
  if(currentRobot&&selectedHand&&asset.robots.some(r=>r.id===currentRobot&&r.hand===selectedHand))await chooseRobot(currentRobot);
  else{$('scene-status').textContent='Choose a robot and end effector to continue';$('robot-name').textContent='Choose robot';$('joint-select').innerHTML='';$('joint-value').textContent='—';$('joint-note').textContent='Robot output is unavailable for this recording';$('cloud-status').textContent='';for(const id of ['left-error','right-error','left-angle','right-angle'])$(id).textContent='—';for(const id of ['fingertip-error','scene-depth-check'])$(id).textContent='';for(const id of ['joint-chart','error-chart','finger-chart']){const c=$<HTMLCanvasElement>(id);c.getContext('2d')?.clearRect(0,0,c.width,c.height);}patch.pause();patch.removeAttribute('src');patch.load();patch.hidden=true;$('patch-empty').hidden=false;$('patch-empty').textContent='No robot composite generated for this recording';$('patch-label').textContent='NOT GENERATED';}
  if(request!==loadSerial)return;rememberCachedRobots(asset);time=preservePlayback?Math.min(savedTime,asset.duration):0;playing=preservePlayback?savedPlaying:true;$('wb-loading').hidden=true;renderLibrary();
 }catch(error){if(request!==loadSerial)return;$('wb-loading').textContent=`Scene unavailable: ${(error as Error).message}`;console.error(error);}
}
function renderObjectTags(){const categories=asset.segmented_instances?.categories||[];$('object-tags').innerHTML=categories.map(o=>`<span>${escape(o)} <small>Segmented cloud</small></span>`).join('');}

interface RobotOption{id:string;label:string;experimental?:boolean;kind?:string;supported?:boolean;reason?:string;conversion_ready?:boolean;report?:{message?:string};}
function sourceHand(dataset:string){return asset?.end_effector || (dataset==='Nymeria'?'none':['UMI','Real robot'].includes(dataset)?'robotiq-2f-140':'wuji');}
interface HandOption{id:string;label:string;status:string;supported:boolean;report?:{message?:string;required?:string[]};}
let robotOptions:RobotOption[]=[],handOptions:HandOption[]=[];let pickerToken=0;
async function refreshOptions(){
 const [robots,catalog,hands]=await Promise.all([json<RobotOption[]>('/api/4d/robots'),json<{robots:RobotOption[]}>('/api/catalog/localmanipulation'),json<HandOption[]>('/api/4d/hands')]);
 const merged=new Map(robots.map(r=>[r.id,r]));for(const r of catalog.robots){if(merged.has(r.id))merged.set(r.id,{...r,...merged.get(r.id),report:r.report});else merged.set(r.id,{...r,supported:false,reason:r.report?.message||'Complete physical hand frames before conversion'});}robotOptions=robotChoices([...merged.values()]);handOptions=hands;
}
function revealPicker(){const bar=$('embodiment-bar');bar.hidden=false;bar.animate([{clipPath:'inset(0 100% 0 0)'},{clipPath:'inset(0 0 0 0)'}],{duration:220,easing:'ease-out'});return bar;}
async function showRobotPicker(){
 const token=++pickerToken;const bar=revealPicker();bar.innerHTML='<p>Loading robots…</p>';
 try{await refreshOptions();if(token!==pickerToken||bar.hidden)return;
 const cached=asset.robots.filter(r=>robotOptions.some(o=>o.id===r.id||o.id===r.registry_robot_id));const all=new Map(robotOptions.map(r=>[r.id,r]));for(const r of cached){if(r.registry_robot_id&&r.registry_robot_id!==r.id)all.delete(r.registry_robot_id);all.set(r.id,{...all.get(r.id),id:r.id,label:r.label,supported:true});}
 bar.innerHTML=`<small>CHOOSE ROBOT</small><div>${[...all.values()].sort((a,b)=>Number(cached.some(r=>r.id===b.id))-Number(cached.some(r=>r.id===a.id))||Number(!!b.supported)-Number(!!a.supported)).map(r=>`<div class="wb-robot-card"><button data-result="${escape(r.id)}" class="${r.id===currentRobot?'selected':''}">${cached.some(v=>v.id===r.id)?'<span class="wb-cached">'+((asset as SceneAsset&{temporary?:boolean}).temporary?'Preview':'Cached')+'</span>':''}<strong>${escape(r.label)}</strong><small>${cached.some(v=>v.id===r.id)?'Ready':r.supported?'Generate': 'Needs setup'}</small></button><button class="wb-delete-robot" data-delete-robot="${escape(r.id)}" aria-label="Delete ${escape(r.label)}" title="Delete robot and its motion cache">×</button></div>`).join('')}<button data-upload="robot" class="wb-add"><strong>＋</strong><small>Add robot</small></button></div><p id="picker-feedback">URDF / ZIP · Select a robot for this recording</p>`;
 bar.querySelectorAll<HTMLButtonElement>('[data-result]').forEach(button=>button.onclick=()=>{const id=button.dataset.result!,row=all.get(id)!;if(!row.supported){$('picker-feedback').textContent=row.reason||'Robot setup required';return;}currentRobot=id;$('robot-name').textContent=row.label;bar.hidden=true;if(!selectedHand)void showHandPicker();else void startSelected();});
 bar.querySelectorAll<HTMLButtonElement>('[data-delete-robot]').forEach(button=>button.onclick=async()=>{
  const id=button.dataset.deleteRobot!,row=all.get(id)!;
  if(!confirm(`Delete ${row.label} and its generated motion cache? Original SMPL, source video and scene data will be kept.`))return;
  button.disabled=true;
  try{const result=await json<{deleted_robot_ids:string[]}>(`/api/robots/${encodeURIComponent(id)}`,{method:'DELETE'});const removed=new Set(result.deleted_robot_ids);asset.robots=asset.robots.filter(r=>!removed.has(r.registry_robot_id||r.id)&&!removed.has(r.id));if(removed.has(currentRobot))resetSelection();await load(asset,true);await refreshLibraryCache();await showRobotPicker();notice('Robot and motion cache deleted. Original SMPL preserved.');}
  catch(e){button.disabled=false;$('picker-feedback').textContent=(e as Error).message;}
 });
 bar.querySelector<HTMLButtonElement>('[data-upload]')!.onclick=()=>showUpload('robot');
 }catch(e){bar.innerHTML=`<p>${escape((e as Error).message)}</p><button id="add-anyway">＋ Add robot</button>`;$('add-anyway').onclick=()=>showUpload('robot');}
}
async function showHandPicker(){
 const token=++pickerToken,bar=revealPicker();bar.innerHTML='<p>Loading end effectors…</p>';
 try{await refreshOptions();if(token!==pickerToken||bar.hidden)return;
 bar.innerHTML=`<small>CHOOSE END EFFECTOR</small><div>${handOptions.filter(h=>h.id===sourceHand(asset.dataset)).map(h=>`<button data-hand="${escape(h.id)}" class="${h.id===selectedHand?'selected':''}"><strong>${escape(h.label)}</strong><small>${h.supported?'Ready':'Needs mapping'}</small></button>`).join('')}<button data-upload="hand" class="wb-add"><strong>＋</strong><small>${sourceHand(asset.dataset)==='robotiq-2f-140'?'Add gripper':'Add hand'}</small></button></div><p id="picker-feedback">${sourceHand(asset.dataset)==='none'?'This source uses wrist tracking without a hand.':'Choose an end effector compatible with this source.'}</p>`;
 bar.querySelectorAll<HTMLButtonElement>('[data-hand]').forEach(button=>button.onclick=()=>{const row=handOptions.find(h=>h.id===button.dataset.hand)!;if(row.id!==sourceHand(asset.dataset)||!row.supported){$('picker-feedback').textContent=row.id!==sourceHand(asset.dataset)?'This source requires '+sourceHand(asset.dataset):row.report?.message||'Hand setup required';return;}selectedHand=row.id;$('hand-name').textContent=row.label;bar.hidden=true;if(!currentRobot)void showRobotPicker();else void startSelected();});
 const upload=bar.querySelector<HTMLButtonElement>('[data-upload]')!;upload.hidden=sourceHand(asset.dataset)==='none';upload.onclick=()=>showUpload('hand');
 }catch(e){bar.innerHTML=`<p>${escape((e as Error).message)}</p><button id="add-anyway">＋ Add hand</button>`;$('add-anyway').onclick=()=>showUpload('hand');}
}
function showUpload(kind:'robot'|'hand'){
 ++pickerToken;const bar=revealPicker();bar.innerHTML=`<small>ADD ${kind==='robot'?'ROBOT':'HAND / GRIPPER'}</small><label class="wb-package-drop">＋ Choose URDF or ZIP<input type="file" id="package-file" accept=".urdf,.zip" aria-label="Upload ${kind} package"/></label><p>${kind==='robot'?'Robot URDF and meshes are inspected with the shared robot importer.':'One hand URDF per package. Include mesh files in the ZIP; mounting and finger mapping remain explicit.'} Up to 150 MB.</p><p id="package-status" role="status"></p>`;
 const input=$<HTMLInputElement>('package-file');const upload=async(file:File)=>{const status=$('package-status');status.textContent='Uploading and inspecting…';input.disabled=true;
 try{if(file.size>150*1024*1024)throw new Error('Package exceeds 150 MB');const form=new FormData();form.append('file',file);const result=await json<RobotOption&HandOption>(kind==='robot'?'/api/robots/localmanipulation':'/api/4d/hands',{method:'POST',body:form});status.textContent=`${result.label} · ${result.report?.message||result.status||'Saved'}`;const done=document.createElement('button');done.textContent='Back to selection';done.onclick=()=>void(kind==='robot'?showRobotPicker():showHandPicker());status.append(document.createElement('br'),done);}
 catch(e){status.textContent=(e as Error).message;input.disabled=false;}};
 input.onchange=()=>{if(input.files?.[0])void upload(input.files[0]);};const drop=bar.querySelector<HTMLElement>('.wb-package-drop')!;drop.ondragover=e=>e.preventDefault();drop.ondrop=e=>{e.preventDefault();if(e.dataTransfer?.files[0])void upload(e.dataTransfer.files[0]);};
}
function resetSelection(){currentRobot='';selectedHand='';++pickerToken;++loadingToken;$('embodiment-bar').hidden=true;$('robot-name').textContent='Choose robot';$('hand-name').textContent='Choose hand';}
async function startSelected(){
 if(!currentRobot||!selectedHand)return;
 const cached=asset.robots.find(r=>r.id===currentRobot&&r.hand===selectedHand);
 if(cached){await chooseRobot(currentRobot);return;}
 const clip=library.find(c=>c.id===asset.id);if(clip)await reconstructClip(clip);else await generateRobot(currentRobot);
}
async function generateRobot(id:string){
 if(!id||!selectedHand)return;
 const scene=asset.id,feedback=$('scene-status');let token=loadSerial,previewRevision=-1;
 try{generationFeedback({status:'running',phase:'Submitting generation request…'});let job=await json<SceneJob>('/api/4d/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scene_id:scene,robot_id:id,hand_id:selectedHand})});
 while(job.status==='queued'||job.status==='running'){if(job.robot_ready&&job.scene_url&&(job.revision??0)!==previewRevision&&token===loadSerial&&currentRobot===id){await load(job.scene_url,true);token=loadSerial;previewRevision=job.revision??0;}if(token===loadSerial&&currentRobot===id)generationFeedback(job);feedback.textContent=job.status==='queued'?`Queued · position ${job.queue_position??1}`:`Generating · ${job.phase||'Retargeting'}`;await new Promise(resolve=>setTimeout(resolve,1200));job=await json(`/api/4d/jobs/${job.id}`);}
 if(job.status!=='completed')throw new Error(job.error||job.phase||'Conversion failed');if(token===loadSerial&&currentRobot===id&&job.scene_url){await load(job.scene_url,true);}else notice('Robot conversion completed and cached.');void refreshLibraryCache();
 }catch(e){if(token===loadSerial&&currentRobot===id)generationFeedback({status:'failed',error:(e as Error).message});feedback.textContent=(e as Error).message;notice((e as Error).message);}
}
interface SceneJob{id?:string;status:string;queue_position?:number;phase?:string;error?:string;scene_url?:string;elapsed_seconds?:number;robot_ready?:boolean;revision?:number;}
function generationFeedback(job:SceneJob){
 const panel=$('generation-feedback'),busy=job.status==='running'||job.status==='queued';
 panel.hidden=false;panel.dataset.state=job.status;
 const title=job.status==='queued'?`Queued · position ${job.queue_position??1}`:job.status==='failed'?'Generation failed':job.status==='completed'?'Ready to play':job.robot_ready?'Motion ready · completing remaining views':'Generating robot motion…';
 const detail=job.status==='queued'?'Waiting for a compute slot. Your upload is saved.':job.status==='failed'?(job.error||job.phase||'Please try again.'):job.status==='completed'?'Robot motion and scene are ready.':`${job.phase||'Preparing generation'}${job.elapsed_seconds!==undefined?' · '+Math.round(job.elapsed_seconds)+' s elapsed':''}`;
 $('generation-title').textContent=title;$('generation-detail').textContent=detail;
 panel.setAttribute('aria-busy',String(busy));
 if(patch.hidden){$('patch-label').textContent=job.status==='queued'?'QUEUED':job.status==='failed'?'FAILED':job.status==='completed'?'READY':'GENERATING';$('patch-empty').hidden=false;$('patch-empty').textContent=title+' '+detail;}
}
const sceneRequests=new Map<string,Promise<void>>();
async function reconstructClip(clip:LibraryClip){
 if(!currentRobot||!selectedHand)return;
 const robotId=currentRobot,handId=selectedHand,key=clip.id+':'+robotId+':'+handId;let selectionToken=loadSerial,previewRevision=-1;
 if(sceneRequests.has(key))return sceneRequests.get(key);
 $('scene-retry').hidden=true;
 const task=(async()=>{try{generationFeedback({status:'running',phase:'Submitting generation request…'});let job=await json<SceneJob>('/api/4d/reconstruct',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scene_id:clip.id,robot_id:robotId,hand_id:handId})});
  while(job.status==='queued'||job.status==='running'){
   if(job.robot_ready&&job.scene_url&&(job.revision??0)!==previewRevision&&asset.id===clip.id&&selectionToken===loadSerial&&currentRobot===robotId&&selectedHand===handId){await load(job.scene_url,true);selectionToken=loadSerial;previewRevision=job.revision??0;}
   if(asset.id===clip.id&&selectionToken===loadSerial&&currentRobot===robotId&&selectedHand===handId){generationFeedback(job);const text=job.status==='queued'?`Queued · position ${job.queue_position??1}`:`${job.phase||'Generating recording'}${job.elapsed_seconds!==undefined?' · '+Math.round(job.elapsed_seconds)+' s':''}`;$('scene-generation-status').textContent=text;$('scene-status').textContent=text;}
   await new Promise(resolve=>setTimeout(resolve,1800));job=await json<SceneJob>(`/api/4d/jobs/${job.id}`);
  }
  if(job.status!=='completed'||!job.scene_url)throw new Error(`${job.error||job.phase||'Generation failed'}${job.id?' · Task '+job.id:''}`);
  clip.scene_url=job.scene_url;void refreshLibraryCache();renderLibrary();if(asset.id===clip.id&&selectionToken===loadSerial&&currentRobot===robotId&&selectedHand===handId){await load(job.scene_url,true);generationFeedback({status:'completed'});}else notice(`${clip.title}: scene ready`);
 }catch(e){if(asset.id===clip.id&&currentRobot===robotId&&selectedHand===handId){generationFeedback({status:'failed',error:(e as Error).message});$('scene-generation-status').textContent=(e as Error).message;$('scene-status').textContent='Generation failed';$('scene-retry').hidden=false;$('scene-retry').onclick=()=>void reconstructClip(clip);}notice((e as Error).message);}
 finally{sceneRequests.delete(key);}})();sceneRequests.set(key,task);return task;
}
function previewClip(clip:LibraryClip,generate=false){
 if(!clip.video){notice('Source video is unavailable for this recording');return;}
 const scene:SceneAsset={id:clip.id,title:clip.title,dataset:clip.dataset,end_effector:clip.end_effector,fps:30,frames:Math.round(clip.duration*30)+1,duration:clip.duration,world:'Source video only',video:clip.video,robots:[],semantic:{task:clip.family,objects:[],provenance:'Source catalog metadata; no 4D scene generated'},tactile:{status:'unavailable',reason:'No contact reconstruction generated for this recording'},wrist_videos:[],geometry:{status:'Source preview only; no reconstructed robot or scene is substituted'}};
 $('library').hidden=true;void load(scene).then(()=>{if(!generate){$('scene-generation-status').textContent='Choose a robot and end effector to continue';$('scene-status').textContent='Choose a robot and end effector to continue';}});if(generate)void reconstructClip(clip);return scene;
}

const libraryCache=new Map<string,string[]>();
let libraryCacheRefresh:Promise<void>|undefined;
function rememberCachedRobots(scene:SceneAsset){
 if((scene as SceneAsset&{temporary?:boolean}).temporary)return;
 libraryCache.set(scene.id,[...new Map(scene.robots.filter(r=>r.encoder_policy===ENCODER_POLICY&&r.model_url&&r.playback_url).map(r=>[r.id,r.label])).values()]);
}
async function refreshLibraryCache(){
 if(libraryCacheRefresh)return libraryCacheRefresh;
 libraryCacheRefresh=(async()=>{
  const catalog=await json<LibraryClip[]>('/api/4d/library');
  for(const clip of library){const latest=catalog.find(c=>c.id===clip.id);if(latest?.scene_url)clip.scene_url=latest.scene_url;}
  const pending=library.filter(c=>c.scene_url);let cursor=0;
  await Promise.all(Array.from({length:Math.min(4,pending.length)},async()=>{
   while(cursor<pending.length){const clip=pending[cursor++];try{const scene=await json<SceneAsset>(clip.scene_url!);rememberCachedRobots(scene);libraryCache.set(clip.id,libraryCache.get(scene.id)||[]);}catch{/* Keep last confirmed cache information on a transient fetch failure. */}}
  }));renderLibrary();
 })().catch(()=>{}).finally(()=>{libraryCacheRefresh=undefined;});
 return libraryCacheRefresh;
}
function libraryCacheBadge(clip:LibraryClip){
 const robots=libraryCache.get(clip.id)||[];
 return robots.length?`<span class="wb-library-cached" title="${escape(robots.join(', '))}">Cached · ${robots.length} ${robots.length===1?'robot':'robots'}</span>`:'';
}

function renderLibrary(){
 const query=$<HTMLInputElement>('library-search').value.toLowerCase();const clips=library.filter(c=>c.dataset===dataset&&`${c.title} ${c.family}`.toLowerCase().includes(query));
 const groups=new Map<string,LibraryClip[]>();for(const clip of clips){const key=`${clip.dataset} / ${motionGroup({...clip,page:clip.dataset==='Nymeria'?'localmanipulation':'manipulation'})[0]}`;groups.set(key,[...(groups.get(key)||[]),clip]);}
 $('library-list').innerHTML=!clips.length?'<p>No recordings connected for this source yet.</p>':[...groups].map(([name,rows])=>`<details ${query||groups.size===1?'open':''}><summary>${escape(name)}<span>${rows.length}</span></summary>${rows.map(c=>`<button data-motion="${escape(c.id)}" class="${c.id===asset?.id?'selected':''}"><strong>${escape(c.title)}</strong>${libraryCacheBadge(c)}<small>${c.duration.toFixed(2)} s${(libraryCache.get(c.id)?.length??0)>0?'':c.scene_url?' · Scene available':' · Generate on selection'}</small></button>`).join('')}</details>`).join('');
 $('library-list').querySelectorAll<HTMLButtonElement>('[data-motion]').forEach(button=>button.onclick=()=>{const clip=library.find(c=>c.id===button.dataset.motion)!;$('library-meta').innerHTML=`<b>${escape(clip.title)}</b><p>${escape(clip.dataset)} · ${escape(clip.family)}</p>`;
  void selectClip(clip);});
 document.querySelectorAll<HTMLButtonElement>('[data-dataset]').forEach(b=>b.classList.toggle('selected',b.dataset.dataset===dataset));
}
$('library-open').onclick=()=>{$('library').hidden=!$('library').hidden;if(!$('library').hidden)void refreshLibraryCache();};$('library-close').onclick=()=>$('library').hidden=true;$<HTMLInputElement>('library-search').oninput=renderLibrary;
document.querySelectorAll<HTMLButtonElement>('[data-dataset]').forEach(b=>b.onclick=()=>{dataset=b.dataset.dataset!;renderLibrary();});
$('robot-open').onclick=showRobotPicker;$('hand-open').onclick=()=>void showHandPicker();
document.addEventListener('pointerdown',e=>{if(!(e.target as HTMLElement).closest('.wb-embodiment'))$('embodiment-bar').hidden=true;});
$('play').onclick=()=>playing=!playing;$('step-back').onclick=()=>{playing=false;time=Math.max(0,time-1/30);};$('step-next').onclick=()=>{playing=false;time=Math.min(asset.duration,time+1/30);};
$<HTMLInputElement>('timeline').oninput=e=>{time=Number((e.target as HTMLInputElement).value);previousChart=-1;};$<HTMLSelectElement>('speed').onchange=e=>speed=Number((e.target as HTMLSelectElement).value);
function focusView(mode:'hands'|'objects'|'task'|'body'){stage.focus(mode);}
$('focus-hands').onclick=()=>focusView('hands');$('focus-objects').onclick=()=>focusView('objects');$('focus-task').onclick=()=>focusView('task');$('focus-body').onclick=()=>focusView('body');$('orbit').onclick=()=>stage.orbit=!stage.orbit;
document.querySelectorAll<HTMLInputElement>('[data-layer]').forEach(input=>input.onchange=()=>{const map:Record<string,string>={cloud:'showCloud',boxes:'showBoxes',traces:'showTraces',robot:'showRobot',segments:'segments'};(stage as unknown as Record<string,boolean>)[map[input.dataset.layer!]]=input.checked;});
document.addEventListener('keydown',e=>{if((e.target as HTMLElement).matches('input,select,textarea'))return;if(e.code==='Space'){e.preventDefault();playing=!playing;}});
function syncVideo(video:HTMLVideoElement,start:number){if(!video.src||video.hidden)return;const wanted=time+start;if(!video.seeking&&video.readyState>=1&&Math.abs(video.currentTime-wanted)>(playing?.3:.025))video.currentTime=wanted;video.playbackRate=speed;if(playing&&video.paused)void video.play().catch(()=>{});else if(!playing&&!video.paused)video.pause();}
function draw(now:number){requestAnimationFrame(draw);const dt=Math.min(.1,(now-last)/1000);last=now;if(asset&&playing){time+=dt*speed;if(time>asset.duration)time=0;}
 const rendered=stage.pose(time);if(!asset)return;const frame=stage.motion?rendered:Math.min(asset.frames-1,Math.floor(time*asset.fps));$('play').textContent=playing?'Ⅱ':'▶';$('time').textContent=`${time.toFixed(2)} / ${asset.duration.toFixed(2)}`;$<HTMLInputElement>('timeline').value=String(time);$('frame-info').textContent=`${String(frame).padStart(3,'0')} / ${asset.frames-1} · 30 Hz`;
 syncVideo(source,asset.video.start_seconds);syncVideo($<HTMLVideoElement>('preview-video'),asset.video.start_seconds);syncVideo(patch,0);document.querySelectorAll<HTMLVideoElement>('#wrist-videos video').forEach(v=>syncVideo(v,Number(v.dataset.start)||0));
 const cloud=asset.cloud;const cloudFrame=cloud?.frames[Math.min(cloud.frames.length-1,Math.floor(time*cloud.fps))];$('cloud-status').textContent=cloudFrame?`Current video frame · ${cloudFrame.count.toLocaleString()} points · estimated`:'';
 const segmented=stage.asset?.segmented_instances;if(segmented){const row=segmented.frames[Math.min(segmented.frames.length-1,Math.floor(time*segmented.fps))];const counts=new Map<string,number>();for(const item of row.instances)counts.set(item.label,(counts.get(item.label)||0)+1);$('object-tags').innerHTML=counts.size?[...counts].map(([label,count])=>`<span>${escape(label)} <small>${count} visible · segmented</small></span>`).join(''):'<span>No object mask <small>in this frame · estimate</small></span>';}
 const motion=stage.motion;if(!motion)return;for(const [side,prefix] of [[0,'left'],[1,'right']] as const){$(prefix+'-error').textContent=motion.tracking.error_mm[frame][side].toFixed(2)+' mm';$(prefix+'-angle').textContent=motion.tracking.orientation_deg[frame][side].toFixed(1)+'°';}
 const tips=motion.tracking.fingertip_error_mm?.[frame];$('fingertip-error').textContent=tips?`Source fingertip deviation: L ${(tips[0].reduce((a,b)=>a+b,0)/5).toFixed(1)} / R ${(tips[1].reduce((a,b)=>a+b,0)/5).toFixed(1)} mm`:'Source fingertip diagnostic unavailable for this result';
 if(contact){const isProximity=$<HTMLSelectElement>('contact-mode').value==='proximity'&&!!proximity;const series=isProximity?proximity!:contact;const sample=Math.min(series.likelihood.length-1,Math.floor(time*series.fps));const videoSample=Math.min(contact.likelihood.length-1,Math.floor(time*contact.fps));const v=series.likelihood[sample];const hands=contact.hand_contact?.[videoSample];$('tactile-kind').textContent=isProximity?'GEOMETRY INDEX · 15 Hz':'VIDEO ESTIMATE · 2 Hz';$('tactile-note').textContent=isProximity?'Independent fingertip proximity indices (0–1), not pressure or contact probabilities. Estimated depth can shift these values.':'Independent video contact scores (0–1); gray / — means unknown. No pressure or force measurement.';for(let si=0;si<2;si++){const hand=hands?.[si];const label=document.getElementById(`contact-hand-${si}`);if(label)label.textContent=isProximity?(si===0?'LEFT':'RIGHT'):`${si===0?'LEFT':'RIGHT'} · video: ${hand?.state??'unknown'}${hand?.object?' / '+hand.object:''}`;document.querySelector(`[data-hand-shape="${si}"]`)?.setAttribute('fill','#203d2a');}const evidence=contact.observations?.[videoSample];$('contact-evidence').textContent=isProximity?'Geometry diagnostic · source fingertips to visible object surfaces. Whole-hand video states are separate.':evidence?`${evidence.time.toFixed(1)}s sample · ${evidence.evidence} ${v.flat().filter(x=>x===null).length}/10 fingertips unknown.`:'';for(let side=0;side<2;side++)for(let finger=0;finger<5;finger++){const node=document.querySelector<SVGElement>(`[data-contact="${side}-${finger}"]`);if(node){const probability=v[side][finger];node.setAttribute('fill',probability===null?'#2c3b30':`rgb(${Math.round(55+170*probability)},${Math.round(80+130*probability)},${Math.round(60+40*probability)})`);node.setAttribute('aria-label',probability===null?'Unknown / not visible':`${isProximity?'Fingertip proximity index':'Video contact score (uncalibrated)'} ${probability.toFixed(2)}`);const meter=document.querySelector<HTMLMeterElement>(`[data-digit-meter="${side}-${finger}"]`);const value=document.querySelector<HTMLElement>(`[data-digit-score="${side}-${finger}"]`);if(meter){meter.value=probability??0;meter.classList.toggle('unknown',probability===null);}if(value)value.textContent=probability===null?'—':probability.toFixed(2);}}}
 const selectedFinger=Number($<HTMLSelectElement>('finger-select').value)||0;if(handAudit){const row=handAudit.finger_motion[selectedFinger];$('finger-stat').textContent=`Bend range ${row.source_curl_range_deg.toFixed(0)}° → ${row.robot_curl_range_deg.toFixed(0)}° · correlation ${row.curl_correlation?.toFixed(2)??'N/A'}`;chart($<HTMLCanvasElement>('finger-chart'),handAudit.source_curl_deg.map((v,i)=>[v[Math.floor(selectedFinger/5)][selectedFinger%5],handAudit!.robot_curl_deg[i][Math.floor(selectedFinger/5)][selectedFinger%5]]),frame,['#e6b778','#a3d391']);}

 if(frame!==lastJointFrame){const values=motion.joints.values[frame];jointValueNodes.forEach((canvas,i)=>{if(canvas.closest('tr')?.hidden)return;const bounds=canvas.getBoundingClientRect(),panel=canvas.closest('.wb-state-table')!.getBoundingClientRect();if(bounds.bottom<panel.top||bounds.top>panel.bottom)return;const w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;const ratio=devicePixelRatio;if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);}const ctx=canvas.getContext('2d')!;ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);const [lo,hi]=jointRanges[i],span=Math.max(hi-lo,.001),y=(v:number)=>h-3-(v-lo)/span*(h-6);ctx.strokeStyle='#274333';ctx.lineWidth=.6;for(let x=0;x<w;x+=16){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}ctx.strokeStyle='#b0d69c';ctx.lineWidth=1.5;ctx.beginPath();const window=Math.max(2,Math.round(motion.fps*4));for(let j=0;j<window;j++){const f=frame-window+1+j;if(f<0)continue;const x=j/(window-1)*w,v=motion.joints.values[f]?.[i];if(!Number.isFinite(v))continue;if(f===0||j===0)ctx.moveTo(x,y(v));else ctx.lineTo(x,y(v));}ctx.stroke();ctx.fillStyle='#d4e9c5';ctx.beginPath();ctx.arc(w-2,y(values[i]),2,0,Math.PI*2);ctx.fill();jointMeters[i].style.width=`${hi-lo<.001?50:Math.max(0,Math.min(100,(values[i]-lo)/span*100))}%`;canvas.title=`${motion.joints.names[i]} · ${values[i].toFixed(4)} ${motion.joints.unit}`;});$('joint-state-frame').textContent=`${(frame/motion.fps).toFixed(3)} s · frame ${frame}`;lastJointFrame=frame;}
 const joint=Number($<HTMLSelectElement>('joint-select').value)||0;$('joint-value').textContent=Number.isFinite(motion.joints.values[frame]?.[joint])?motion.joints.values[frame][joint].toFixed(3)+' '+motion.joints.unit:'—';
 if(frame!==previousChart){chart($<HTMLCanvasElement>('error-chart'),motion.tracking.error_mm,frame,['#eeb669','#7ad3dc'],10);chart($<HTMLCanvasElement>('joint-chart'),motion.joints.values.map(v=>[v[joint]]),frame,['#b0d69c']);previousChart=frame;}
}
requestAnimationFrame(draw);
void (async()=>{try{library=await json<LibraryClip[]>('/api/4d/library');const unique=new Map<string,LibraryClip>();for(const clip of library){const key=JSON.stringify([clip.dataset,clip.title,clip.duration,clip.video?.start_seconds||0]);if(!unique.has(key)||clip.scene_url)unique.set(key,clip);}library=[...unique.values()];$('library-count').textContent=String(library.length);renderLibrary();void refreshLibraryCache();const selected=new URLSearchParams(location.search).get('clip');const clip=library.find(c=>c.id===selected);if(clip){await selectClip(clip);return;}const candidate=new URLSearchParams(location.search).get('scene');await load(candidate&&/^\/assets\/manipulation4d\/[A-Za-z0-9/_-]+\/scene\.json$/.test(candidate)?candidate:library.find(c=>c.scene_url)!.scene_url!);}catch(e){$('wb-loading').textContent='Workspace initialization failed';console.error(e);}})();

function heightApplied(clip:LibraryClip){
 const custom=clip as LibraryClip&{temporary?:boolean};
 if(!custom.temporary&&!library.some(c=>c.id===clip.id))library.push(clip);
 resetSelection();renderLibrary();if(clip.scene_url){void load(clip.scene_url);}else previewClip(clip);
}
async function selectClip(clip:LibraryClip){
 resetSelection();++loadSerial;
 try{
  if(clip.dataset==='UMI'){
   const setup=await json<{requires_initial_height:boolean;default_scene_url?:string}>(`/api/4d/trajectory-height/${encodeURIComponent(clip.id)}`);
   if(setup.requires_initial_height){const preview=previewClip(clip,false);if(preview)await showTrajectoryHeight(preview,heightApplied,true);return;}
   if(setup.default_scene_url)clip.scene_url=setup.default_scene_url;
  }
  if(clip.scene_url){$('library').hidden=true;await load(clip.scene_url);}else previewClip(clip);
 }catch(e){notice((e as Error).message);}
}
$('trajectory-height-open').onclick=()=>{playing=false;void showTrajectoryHeight(asset,heightApplied).catch(e=>notice(e.message));};
