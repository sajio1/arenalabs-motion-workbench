import {createIcons,icons} from 'lucide';
import './style.css';
import {Stage} from './scene';
import {safeText as esc,visibleClips,activeRobot} from './state.js';
import type {Catalog,Clip,Robot,Slot,Page,Playback} from './types';

const pages:Record<Page,{subtitle:string;title:string;index:string;source:string;icon:string}>={
  locomotion:{subtitle:'SOMA → FULL-BODY',title:'Movement, reimagined.',index:'01',source:'SOMA full-body motion · Robot comparison',icon:'footprints'},
  localmanipulation:{subtitle:'NYMERIA → HUMANOID',title:'Four traces. A whole body.',index:'02',source:'Nymeria wrists and feet · Whole-body reconstruction',icon:'scan-line'},
  manipulation:{subtitle:'EGODEX → WHEELED',title:'Human intent. Robot motion.',index:'03',source:'EgoDex manipulation · Wheeled upper-body retargeting',icon:'hand'},
};
const path=location.pathname.slice(1) as Page;const page:Page=path in pages?path:'locomotion';const info=pages[page];
const icon=(name:string,cls='')=>`<i data-lucide="${name}" class="${cls}"></i>`;
const app=document.getElementById('app')!;
app.innerHTML=`<div id="viewport"></div>
 <div class="brand"><div class="brand-mark"></div><div class="brand-name">arenalab<span>MOTION LAB</span></div></div>
 <div class="pipeline-wrap"><button id="pipeline-toggle" class="pipeline-toggle" aria-expanded="false" aria-label="Switch pipeline"><span class="type-icon">${icon(info.icon)}</span><span><strong>${page}</strong><small>${info.subtitle}</small></span>${icon('chevrons-up-down','chevron')}</button>
 <nav id="pipeline-menu" class="pipeline-menu" hidden>${Object.entries(pages).map(([key,value])=>`<a href="/${key}" class="${key===page?'current':''}">${key}<small>${value.subtitle}</small></a>`).join('')}</nav></div>
 <aside class="sidebar"><div class="section-title">MOTION LIBRARY<span id="catalog-count" class="count">—</span></div>
 <label class="searchbox">${icon('search')}<input id="search" placeholder="Search motions, families…" aria-label="Search motions"/></label><div id="catalog" class="catalog"></div>
 <div class="drag-hint">${icon('mouse-pointer-2')}Drag a motion into a scene slot</div><section id="metadata" class="metadata"></section></aside>
 <div class="top-tools"><button id="orbit" class="on" aria-pressed="true" aria-label="Auto orbit">${icon('orbit')}<span>Orbit</span><kbd>O</kbd></button><button id="reset" aria-label="Reset view and time">${icon('rotate-ccw')}<span>Reset</span><kbd>R</kbd></button></div>
 <section class="display-panel"><div class="small-title">DISPLAY</div>${[['mesh','Robot mesh',true],['skeleton','Skeleton',false],['targets','Input traces',false],['wireframe','Wireframe',false]].map(([key,label,checked])=>`<div class="switch-row"><span>${label}</span><button class="switch" role="switch" aria-label="${label}" aria-checked="${checked}" data-display="${key}"></button></div>`).join('')}</section>
 <div id="slots" class="slot-overlay"></div><section id="source-videos" class="source-videos" hidden></section>
 <div class="robot-area"><div id="robot-context" class="robot-context"></div><button id="robot-toggle" class="robot-button" aria-expanded="false" aria-label="Choose robot"></button><div id="robot-tray" class="robot-tray" hidden></div><div id="robot-inspect" class="robot-inspect"></div></div>
 <div class="transport"><button id="play" class="play" aria-label="Play or pause">${icon('pause')}</button><button id="prev" class="step" aria-label="Previous frame">${icon('skip-back')}</button><span id="time" class="time">0.00 / 0.00</span><input id="timeline" class="timeline" type="range" min="0" max="1" step="0.0333333333333333" value="0" aria-label="Playback time"/><button id="next" class="step" aria-label="Next frame">${icon('skip-forward')}</button><span class="divider"></span><select id="speed" aria-label="Playback speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></div>
 <div id="frame-label" class="frame-label">30 HZ · SHARED TIMELINE · PHYSICAL SCALE</div><div class="notice"><span class="dot"></span>LOCAL WORKSPACE · Motion quality not evaluated</div>
 <div id="toast" class="toast" hidden role="status"></div><div id="modal-root"></div><div id="boot" class="loading-screen">Loading your motion workspace…</div>`;
const el=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;
function refreshIcons(){createIcons({icons});}
refreshIcons();
let catalog:Catalog;let defaultRobot:Robot;let slots:Slot[]=[];let selected:number|null=null;let selectedClip:Clip|undefined;let robotTrayOpen=false;
let playing=true,time=0,speed=1,duration=0,last=performance.now(),dragging=false,suppressClick=false;let toastTimer=0;const thumbnails=new Map<string,string>();
const stage=new Stage(el('viewport'),()=>syncOrbit());
const storageKey=`arenalab.motion-lab.${page}.v1`;
function persist(){if(!catalog)return;sessionStorage.setItem(storageKey,JSON.stringify({defaultRobot:defaultRobot.id,time,slots:slots.map(s=>({clip:s.clip?.id,robot:s.robot.id}))}));}
function toast(text:string){el('toast').textContent=text;el('toast').hidden=false;clearTimeout(toastTimer);toastTimer=window.setTimeout(()=>el('toast').hidden=true,6500);}
async function request(url:string,options?:RequestInit){const r=await fetch(url,options);if(!r.ok){const error=await r.json().catch(()=>({detail:r.statusText}));throw new Error(typeof error.detail==='string'?error.detail:'Request failed');}return r.json();}
function robotImage(robot:Robot){return thumbnails.has(robot.id)?`<img class="robot-img" src="${thumbnails.get(robot.id)}" alt="${esc(robot.label)}"/>`:`<div class="robot-img-placeholder">${icon('bot')}</div>`;}
async function warmThumbnails(robots:Robot[]){for(const robot of robots){if(!robot.model_url||thumbnails.has(robot.id))continue;try{thumbnails.set(robot.id,await stage.thumbnail(robot));renderRobot();}catch(error){console.warn('Robot thumbnail unavailable',robot.id,error);}}}
function currentRobot():Robot{return activeRobot(defaultRobot,selected,slots);}
function renderRobot(){
 const robot=currentRobot();el('robot-context').innerHTML=selected===null?`${icon('circle-dot')}Default robot for new motions`:`${icon('focus')}Slot ${String(selected+1).padStart(2,'0')} robot <button id="default-mode" aria-label="Return to default robot selection">${icon('x')}</button>`;
 el('robot-toggle').innerHTML=`${robotImage(robot)}<span><span class="micro">${robot.kind==='Wheeled'?'WHEELED ROBOT':robot.kind==='Custom URDF'?'CUSTOM URDF':'HUMANOID ROBOT'}</span><strong>${esc(robot.label.replace('Unitree ',''))}</strong><small>${robot.conversion_ready?'Choose / switch robot':'Uploaded · Metadata required'}</small></span>${icon(robotTrayOpen?'chevron-left':'chevron-right','chevron')}`;
 el('robot-toggle').setAttribute('aria-expanded',String(robotTrayOpen));el('robot-tray').hidden=!robotTrayOpen;
 el('robot-tray').innerHTML=catalog.robots.map(r=>`<button class="robot-card ${r.id===robot.id?'selected':''} ${r.id.startsWith('custom_')?'custom':''}" data-robot="${r.id}" aria-label="Choose ${esc(r.label)}">${r.id===robot.id?icon('check','check'):''}${thumbnails.has(r.id)?`<img src="${thumbnails.get(r.id)}" alt=""/>`:`<div style="height:79px;display:grid;place-items:center">${icon('bot')}</div>`}<strong>${esc(r.label.replace('Unitree ',''))}</strong><small>${r.conversion_ready?r.kind:'Metadata required'}</small></button>`).join('')+`<button class="robot-card upload-card" id="upload-open" aria-label="Upload robot URDF">${icon('plus')}<strong>Upload robot</strong><small>URDF / ZIP</small></button>`;
 el('robot-inspect').innerHTML=!robot.conversion_ready?`<button id="inspect-robot">Inspect model</button>`:'';
 el('default-mode')?.addEventListener('click',()=>{selected=null;renderRobot();renderSlots();});
 document.querySelectorAll<HTMLButtonElement>('[data-robot]').forEach(b=>b.onclick=()=>chooseRobot(b.dataset.robot!));
 el('upload-open').onclick=()=>openUpload();el('inspect-robot')?.addEventListener('click',()=>openInspection(robot));refreshIcons();
}
function chooseRobot(id:string){const robot=catalog.robots.find(r=>r.id===id)!;if(selected===null){defaultRobot=robot;toast(`New motions will use ${robot.label}`);}
 else{const slot=slots[selected];if(slot.robot.id!==id){if(slot.clip)void assign(selected,slot.clip,robot);else slot.robot=robot;}}
 renderRobot();renderSlots();persist();
}
function renderCatalog(){const matches=visibleClips(catalog.clips,el<HTMLInputElement>('search').value) as Clip[];el('catalog-count').textContent=String(matches.length).padStart(2,'0');
 const groups=new Map<string,Clip[]>();matches.forEach(c=>groups.set(c.family,[...(groups.get(c.family)||[]),c]));
 el('catalog').innerHTML=Array.from(groups).map(([family,clips])=>`<details class="family" open><summary>${icon('chevron-down')}${esc(family)}<span class="number">${clips.length}</span></summary>${clips.map(c=>`<button class="clip ${selectedClip?.id===c.id?'selected':''}" draggable="false" data-clip="${c.id}" title="${esc(c.title)}" aria-label="Motion ${esc(c.title)}"><span><strong>${esc(c.title)}</strong><small>${c.duration.toFixed(2)} s &nbsp;·&nbsp; ${c.fps} Hz</small></span>${icon('grip-vertical','grip')}</button>`).join('')}</details>`).join('')||'<div class="empty-list">No matching motions</div>';
 document.querySelectorAll<HTMLButtonElement>('[data-clip]').forEach(b=>{
   b.onclick=()=>{if(suppressClick)return;selectedClip=catalog.clips.find(c=>c.id===b.dataset.clip);renderMetadata();renderCatalog();};
   b.ondblclick=()=>{const clip=catalog.clips.find(c=>c.id===b.dataset.clip)!;const index=selected??slots.findIndex(s=>!s.clip);if(index<0){toast('All three slots are occupied. Select a slot to replace.');return;}void assign(index,clip,selected===null?defaultRobot:slots[index].robot);};
   b.ondragstart=e=>{e.dataTransfer!.setData('application/x-greenwich-clip',b.dataset.clip!);e.dataTransfer!.setData('text/plain',b.dataset.clip!);e.dataTransfer!.effectAllowed='copy';dragging=true;stage.autoRotate=false;syncOrbit();document.querySelectorAll('[data-slot]').forEach(n=>n.classList.add('drag-mode'));};
   b.ondragend=()=>{dragging=false;document.querySelectorAll('[data-slot]').forEach(n=>n.classList.remove('drag-mode','drag-over'));};
   b.onpointerdown=e=>beginDrag(e,b);
 });refreshIcons();
}
function beginDrag(down:PointerEvent,button:HTMLButtonElement){
 if(down.button!==0)return;const clip=catalog.clips.find(c=>c.id===button.dataset.clip)!;let started=false;let ghost:HTMLElement|undefined;let target:HTMLElement|null=null;
 const move=(event:PointerEvent)=>{if(event.pointerId!==down.pointerId)return;
   if(!started&&Math.hypot(event.clientX-down.clientX,event.clientY-down.clientY)<6)return;
   if(!started){started=true;dragging=true;stage.autoRotate=false;syncOrbit();document.querySelectorAll('[data-slot]').forEach(n=>n.classList.add('drag-mode'));ghost=document.createElement('div');ghost.className='drag-ghost';ghost.textContent=clip.title;document.body.append(ghost);}
   event.preventDefault();ghost!.style.left=`${event.clientX+15}px`;ghost!.style.top=`${event.clientY+15}px`;
   const hit=document.elementFromPoint(event.clientX,event.clientY)?.closest<HTMLElement>('[data-slot]')||null;
   if(hit!==target){target?.classList.remove('drag-over');target=hit;target?.classList.add('drag-over');}
 };
 const finish=(event:PointerEvent)=>{if(event.pointerId!==down.pointerId)return;window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',finish);window.removeEventListener('pointercancel',cancel);
   ghost?.remove();dragging=false;document.querySelectorAll('[data-slot]').forEach(n=>n.classList.remove('drag-mode','drag-over'));
   if(started){suppressClick=true;window.setTimeout(()=>suppressClick=false,300);if(target){const i=Number(target.dataset.slot);void assign(i,clip,slots[i].clip?slots[i].robot:defaultRobot);}}
 };
 const cancel=(event:PointerEvent)=>{target=null;finish(event);};
 window.addEventListener('pointermove',move,{passive:false});window.addEventListener('pointerup',finish);window.addEventListener('pointercancel',cancel);
}
function renderMetadata(){const c=selectedClip;const slot=selected!==null?slots[selected]:undefined;const result=slot?.clip?.id===c?.id?slot:undefined;
 const sections:Record<string,Record<string,unknown>>=c?{Source:{Family:c.family,Duration:`${c.duration.toFixed(2)} s`,Frames:c.frames,'Frame rate':`${c.fps} Hz`,Format:c.source_format,...c.metadata,File:c.source_name},...(result?{Result:{Robot:result.robot.label,Version:result.payload?.version||c.version,Frames:result.payload?.frames,'Quality status':result.payload?.quality_status||'not_evaluated'}}:{})}:{};
 el('metadata').innerHTML=`<div class="section-title">METADATA <span class="source-tag">${c?esc(c.source_format.split(' · ')[0]):'SOURCE'}</span></div>${c?`<div class="meta-title">${esc(c.title)}</div>${Object.entries(sections).map(([title,values])=>`<details class="metadata-section" open><summary>${esc(title)}</summary><dl class="metadata-fields">${Object.entries(values).filter(([,v])=>v!==undefined&&v!==null&&v!=='').map(([key,value])=>`<div><dt>${esc(key)}</dt><dd>${formatMetadata(value)}</dd></div>`).join('')}</dl></details>`).join('')}<div class="meta-note">${esc(c.description)}</div>`:'<div class="meta-note">Select a motion to inspect its source metadata.<br/>Place it in a slot to inspect its robot result.</div>'}`;
}
function formatMetadata(value:unknown):string{if(Array.isArray(value)){const text=value.map(formatMetadata).join(', ');return value.length>6?`<details class="metadata-array"><summary>${value.length} items</summary><div>${text}</div></details>`:text;}if(value!==null&&typeof value==='object')return `<div class="metadata-nested">${Object.entries(value).map(([k,v])=>`<div><span>${esc(k)}</span>: ${formatMetadata(v)}</div>`).join('')}</div>`;return esc(value);}
function renderSlots(){
 el('slots').innerHTML=slots.map(s=>`<div class="slot-label ${selected===s.index?'active':''} ${dragging?'drag-mode':''}" data-slot="${s.index}" role="button" tabindex="0" aria-label="Slot ${s.index+1}${s.clip?' '+esc(s.clip.title):' Add motion'}">
 ${s.clip?`<div class="slot-header"><span class="slot-number">0${s.index+1}</span><span class="slot-name">${esc(s.clip.title)}</span><button class="remove" data-remove="${s.index}" aria-label="Remove slot ${s.index+1}">${icon('x')}</button></div><div class="slot-subtitle">${esc(s.robot.label)} &nbsp;·&nbsp; ${s.payload?.duration.toFixed(2)||s.clip.duration.toFixed(2)} s</div>${s.payload?'<div class="quality-chip">Motion quality not evaluated</div>':''}`:`<div class="slot-add">${icon('plus')}<span>Drop a motion</span><small>SLOT 0${s.index+1}</small></div>`}
 ${!['empty','ready'].includes(s.status)?`<div class="slot-progress ${s.status==='failed'?'error':''}">${s.status==='failed'?'': '<span class="tiny-spinner"></span>'}${esc(s.message||({loading:'Loading motion',queued:'Queued for local conversion',running:'Retargeting motion',failed:'Conversion failed'} as Record<string,string>)[s.status])}</div>`:''}</div>`).join('');
 document.querySelectorAll<HTMLElement>('[data-slot]').forEach(node=>{
   const i=Number(node.dataset.slot);node.onclick=()=>selectSlot(i);node.onkeydown=e=>{if(e.key==='Enter')selectSlot(i);};
   node.ondragover=e=>{if(e.dataTransfer?.types.includes('application/x-greenwich-clip')){e.preventDefault();e.dataTransfer.dropEffect='copy';node.classList.add('drag-over');}};
   node.ondragleave=()=>node.classList.remove('drag-over');node.ondrop=e=>{e.preventDefault();node.classList.remove('drag-over');const id=e.dataTransfer?.getData('application/x-greenwich-clip');const clip=catalog.clips.find(c=>c.id===id);dragging=false;if(clip)void assign(i,clip,slots[i].clip?slots[i].robot:defaultRobot);else renderSlots();};
 });
 document.querySelectorAll<HTMLButtonElement>('[data-remove]').forEach(b=>b.onclick=e=>{e.stopPropagation();const i=Number(b.dataset.remove);slots[i].token++;slots[i].clip=undefined;slots[i].payload=undefined;slots[i].status='empty';slots[i].message=undefined;stage.remove(i);selected=null;updateDuration();renderSlots();renderRobot();renderMetadata();persist();});
 refreshIcons();positionLabels();renderVideos();
}
function renderVideos(){
 const strip=el('source-videos');strip.hidden=page==='locomotion';if(strip.hidden)return;
 const key=slots.map(s=>s.clip?.id||'').join('|');if(strip.dataset.key===key)return;strip.dataset.key=key;
 strip.innerHTML=slots.map(s=>`<button class="source-video-tile" data-video-slot="${s.index}" aria-label="Enlarge source video for slot ${s.index+1}"><header><b>0${s.index+1}</b><span>${esc(s.clip?.title||'Empty slot')}</span><small>EGO</small></header>${s.clip?.video?`<video muted playsinline preload="metadata" src="${esc(s.clip.video.url)}" data-source-slot="${s.index}"></video>`:`<div class="video-missing">${s.clip?'Source video unavailable':'Drop a motion to compare'}</div>`}</button>`).join('');
 strip.querySelectorAll<HTMLButtonElement>('[data-video-slot]').forEach(b=>b.onclick=()=>{const i=Number(b.dataset.videoSlot),s=slots[i];if(!s.clip?.video){toast('No verified source video for this clip');return;}selectSlot(i);modal(`<div class="modal-head"><h2>Slot 0${i+1} · Source Ego</h2><button id="modal-close" aria-label="Close">${icon('x')}</button></div><video class="source-video-large" muted playsinline preload="metadata" src="${esc(s.clip.video.url)}" data-source-slot="${i}"></video><p class="helper">${esc(s.clip.video.sync_method)} · Shared timeline</p>`);});
 strip.querySelectorAll('video').forEach(v=>v.addEventListener('error',()=>{v.replaceWith(Object.assign(document.createElement('div'),{className:'video-missing',textContent:'Source video could not be loaded'}));}));
}
function syncVideos(){
 document.querySelectorAll<HTMLVideoElement>('video[data-source-slot]').forEach(v=>{
   const slot=slots[Number(v.dataset.sourceSlot)],clip=slot?.clip,descriptor=clip?.video;
   if(!clip||!descriptor||!slot.payload||slot.status!=='ready'){v.pause();return;}
   const local=Math.min(time,slot.payload.duration,clip.duration),frame=Math.min(Math.floor(local*30),clip.frames-1);
   const target=descriptor.frame_times?.[frame]??descriptor.start_seconds+local;
   const bounded=Number.isFinite(v.duration)?Math.min(target,Math.max(0,v.duration-1/30)):target;
   if(v.readyState>=1&&!v.seeking&&Math.abs(v.currentTime-bounded)>(playing?.08:.012))v.currentTime=bounded;
   v.playbackRate=speed;
   const run=playing&&time<Math.min(slot.payload.duration,clip.duration);
   if(run&&v.paused)void v.play().catch(()=>{});else if(!run&&!v.paused)v.pause();
 });
}
function positionLabels(){document.querySelectorAll<HTMLElement>('[data-slot]').forEach(n=>{const i=Number(n.dataset.slot);const p=stage.screenAnchor(i);n.style.left=`${p.x}px`;n.style.top=`${Math.min(p.y+16,innerHeight-180)}px`;});}
function selectSlot(i:number){selected=i;selectedClip=slots[i].clip||selectedClip;renderSlots();renderRobot();renderMetadata();renderCatalog();}
function updateDuration(){duration=Math.max(0,...slots.map(s=>s.payload?.duration||0));el<HTMLInputElement>('timeline').max=String(duration||1);time=Math.min(time,duration);}
async function assign(i:number,clip:Clip,robot:Robot){
 const slot=slots[i],token=++slot.token,previous={clip:slot.clip,robot:slot.robot,payload:slot.payload};
 slot.clip=clip;slot.robot=robot;slot.status='loading';slot.message=undefined;selected=i;selectedClip=clip;stage.autoRotate=false;syncOrbit();renderSlots();renderRobot();renderMetadata();renderCatalog();
 try{
   let result=await request('/api/convert',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clip_id:clip.id,robot_id:robot.id})});
   while(result.status!=='completed'){
     if(token!==slot.token)return;if(result.status==='failed')throw new Error(result.error||'Conversion failed');
     slot.status=result.status;slot.message=result.status==='queued'?'Waiting for GPU · Other slots remain playable':'Running pipeline refinement';renderSlots();
     await new Promise(resolve=>setTimeout(resolve,1000));result=await request(`/api/jobs/${result.id}`);
   }
   if(token!==slot.token)return;const payload:Playback=await request(result.url);if(token!==slot.token)return;
   await stage.load(i,robot,payload,()=>token===slot.token);if(token!==slot.token)return;
   slot.payload=payload;slot.status='ready';slot.message=undefined;time=0;updateDuration();persist();renderSlots();renderMetadata();
 }catch(error){if(token!==slot.token)return;slot.status='failed';slot.message=(error as Error).message;if(previous.payload){slot.clip=previous.clip;slot.robot=previous.robot;slot.payload=previous.payload;slot.message+=' · Previous motion retained';}renderSlots();renderRobot();renderMetadata();toast(slot.message||'Loading failed');}
}
function syncOrbit(){el('orbit').classList.toggle('on',stage.autoRotate);el('orbit').setAttribute('aria-pressed',String(stage.autoRotate));}
function syncPlay(){el('pause')&&(el('pause').innerHTML=`${icon(playing?'pause':'play')}<span>${playing?'Pause':'Play'}</span><kbd>Space</kbd>`);el('play').innerHTML=icon(playing?'pause':'play');refreshIcons();}
function togglePlay(){playing=!playing;syncPlay();}
function step(direction:number){playing=false;time=Math.max(0,Math.min(duration,time+direction/30));stage.autoRotate=false;syncPlay();syncOrbit();}
function modal(content:string){el('modal-root').innerHTML=`<div class="modal-backdrop"><section class="modal" role="dialog" aria-modal="true">${content}</section></div>`;refreshIcons();el('modal-close')?.addEventListener('click',()=>el('modal-root').innerHTML='');document.querySelector('.modal-backdrop')?.addEventListener('click',e=>{if(e.target===e.currentTarget)el('modal-root').innerHTML='';});}
function openInspection(robot:Robot){modal(`<div class="modal-head"><h2>${esc(robot.label)}</h2><button id="modal-close" aria-label="Close">${icon('x')}</button></div><p>URDF inspection</p><pre>${esc(JSON.stringify(robot.report,null,2))}</pre><p class="helper">Model preview and conversion readiness are checked separately. Required mechanical metadata must be provided.</p>`);}
function openUpload(){
 modal(`<div class="modal-head"><h2>Add your robot.</h2><button id="modal-close" aria-label="Close">${icon('x')}</button></div><p>Upload a URDF, or a ZIP containing the URDF and meshes.<br/>Models stay in your local workspace and join the library after inspection.</p><label class="upload-drop" for="robot-file">${icon('upload')}<strong>Choose a file or drop a robot package here</strong><small>.urdf / .zip · Up to 150 MB</small></label><input id="robot-file" type="file" accept=".urdf,.zip"/><div id="upload-result" class="result"></div><p class="helper">Conversion requires wrist/foot/TCP metadata and explicit closed-chain constraints.</p>`);
 const file=el<HTMLInputElement>('robot-file');file.onchange=()=>{if(file.files?.[0])void uploadRobot(file.files[0]);};
 const drop=document.querySelector<HTMLElement>('.upload-drop')!;drop.ondragover=e=>e.preventDefault();drop.ondrop=e=>{e.preventDefault();if(e.dataTransfer?.files[0])void uploadRobot(e.dataTransfer.files[0]);};
}
async function uploadRobot(file:File){const result=el('upload-result');result.textContent='Checking URDF and mesh assets…';try{
 const form=new FormData();form.append('file',file);const robot:Robot=await request(`/api/robots/${page}`,{method:'POST',body:form});catalog.robots.push(robot);defaultRobot=robot;selected=null;robotTrayOpen=true;renderRobot();void warmThumbnails([robot]);
 result.innerHTML=`${esc(robot.label)} added to the robot library.<br/>${esc(robot.report?.message||'Model inspection complete')}<div class="modal-buttons"><button class="primary" id="upload-done">View robot</button></div>`;
 el('upload-done').onclick=()=>{el('modal-root').innerHTML='';persist();};
 }catch(error){result.textContent=(error as Error).message;}}
el('pipeline-toggle').onclick=()=>{const menu=el('pipeline-menu');menu.hidden=!menu.hidden;el('pipeline-toggle').setAttribute('aria-expanded',String(!menu.hidden));};
el('robot-toggle').onclick=()=>{robotTrayOpen=!robotTrayOpen;renderRobot();};
el('search').addEventListener('input',()=>renderCatalog());el('play').onclick=togglePlay;el('prev').onclick=()=>step(-1);el('next').onclick=()=>step(1);
el('orbit').onclick=()=>{stage.autoRotate=!stage.autoRotate;syncOrbit();};el('reset').onclick=()=>{stage.reset();time=0;persist();};
el<HTMLInputElement>('timeline').oninput=e=>{time=Number((e.target as HTMLInputElement).value);playing=false;stage.autoRotate=false;syncPlay();syncOrbit();};
el<HTMLSelectElement>('speed').onchange=e=>speed=Number((e.target as HTMLSelectElement).value);
document.querySelectorAll<HTMLButtonElement>('[data-display]').forEach(b=>b.onclick=()=>{const key=b.dataset.display as 'mesh'|'skeleton'|'targets'|'wireframe';stage[key]=!stage[key];b.setAttribute('aria-checked',String(stage[key]));stage.visibility();});
document.addEventListener('keydown',e=>{if((e.target as HTMLElement)?.matches('input,select,textarea'))return;if(e.code==='Space'){e.preventDefault();togglePlay();}if(e.key.toLowerCase()==='o'){stage.autoRotate=!stage.autoRotate;syncOrbit();}if(e.key.toLowerCase()==='r'){stage.reset();time=0;}if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1);if(e.key==='Escape'){el('modal-root').innerHTML='';robotTrayOpen=false;renderRobot();}});
el('viewport').addEventListener('click',e=>{const hit=stage.pick(e.clientX,e.clientY);if(hit!==null){selectSlot(hit);return;}selected=null;if(catalog){renderRobot();renderSlots();renderMetadata();}});
document.addEventListener('dragover',e=>{if(e.dataTransfer?.types.includes('application/x-greenwich-clip')){e.preventDefault();dragging=true;}});
function animate(){requestAnimationFrame(animate);const now=performance.now();const dt=Math.min((now-last)/1000,.1);last=now;if(playing&&duration>0){time+=dt*speed;if(time>duration)time=0;}
 stage.frame(time);positionLabels();syncVideos();el('time').textContent=`${time.toFixed(2)} / ${duration.toFixed(2)}`;el<HTMLInputElement>('timeline').value=String(time);el('frame-label').textContent=`FRAME ${Math.floor(time*30).toString().padStart(3,'0')} · 30 HZ · SHARED TIMELINE · PHYSICAL SCALE`;
}
async function boot(){try{catalog=await request(`/api/catalog/${page}`);defaultRobot=catalog.robots.find(r=>r.id==='g1')||catalog.robots[0];if(!defaultRobot)throw new Error('No robot configured for this pipeline');
 slots=Array.from({length:3},(_,index)=>({index,robot:defaultRobot,status:'empty',token:0}));
 let saved:any;try{saved=JSON.parse(sessionStorage.getItem(storageKey)||'null');}catch{}
 defaultRobot=catalog.robots.find(r=>r.id===saved?.defaultRobot)||defaultRobot;selectedClip=catalog.clips[0];renderCatalog();renderRobot();renderMetadata();renderSlots();el('boot').remove();animate();void warmThumbnails(catalog.robots);
 if(saved){await Promise.all(saved.slots.map(async(s:any,i:number)=>{const clip=catalog.clips.find(c=>c.id===s.clip),robot=catalog.robots.find(r=>r.id===s.robot);if(clip&&robot)await assign(i,clip,robot);}));time=Math.min(saved.time||0,duration);}
 else if(catalog.clips.length){const clip=catalog.clips.find(c=>c.cache&&Object.keys(c.cache).length>=2)||catalog.clips[0];await assign(0,clip,defaultRobot);const other=catalog.robots.find(r=>r.id!==defaultRobot.id&&clip.cache?.[r.id]);if(other)await assign(1,clip,other);}
 selected=null;renderRobot();renderSlots();renderMetadata();stage.autoRotate=true;syncOrbit();
 }catch(error){el('boot').innerHTML=`<div class="fatal">Workspace could not be loaded<br/>${esc((error as Error).message)}<br/><small>Run prepare_motion_lab.py before starting the local service.</small></div>`;}}
void boot();
