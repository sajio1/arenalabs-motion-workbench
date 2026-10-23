import {api, apiJson} from './platform-api';
import {demoNavigation,demoBrand} from './demo-nav';
import {createIcons,icons} from 'lucide';
import './style.css';
import {errorChart} from './tracking.js';
import {Stage} from './scene';
import {sampleFrame,safeText as esc,visibleClips,robotChoices,motionGroup} from './state.js';
import type {Catalog,Clip,Robot,Slot,Page,Playback} from './types';

const pages:Record<Page,{subtitle:string;title:string;index:string;source:string;icon:string}>={
  locomotion:{subtitle:'SOMA → FULL-BODY',title:'Movement, reimagined.',index:'01',source:'SOMA full-body motion · Robot comparison',icon:'footprints'},
  localmanipulation:{subtitle:'NYMERIA → HUMANOID',title:'Four traces. A whole body.',index:'02',source:'Nymeria wrists and feet · Whole-body reconstruction',icon:'scan-line'},
  manipulation:{subtitle:'EGODEX → WHEELED',title:'Human intent. Robot motion.',index:'03',source:'EgoDex manipulation · Wheeled upper-body retargeting',icon:'hand'},
};
const path=location.pathname.slice(1) as Page;const page:Page=path in pages?path:'locomotion';const info=pages[page];
const robotStatus=(r:Robot)=>r.conversion_ready?'Ready':r.status==='invalid_model'?'Model load failed':r.status==='missing_assets'?'Missing meshes':'Needs physical metadata';
const icon=(name:string,cls='')=>`<i data-lucide="${name}" class="${cls}"></i>`;
document.body.classList.add('motion-page');document.title='ArenaLabs · Locomotion';
const app=document.getElementById('app')!;
app.innerHTML=`<div id="viewport"></div>
 <header class="arena-header">${demoBrand()}${demoNavigation('locomotion')}<span class="arena-context">Locomotion</span></header>
 <aside class="sidebar"><section class="library-panel"><div class="section-title">MOTION LIBRARY<span id="catalog-count" class="count">—</span></div>
 <div class="drag-hint">${icon('mouse-pointer-2')}Drag a motion into a scene slot</div><label class="searchbox">${icon('search')}<input id="search" placeholder="Search motions, families…" aria-label="Search motions"/></label><div class="library-tabs" role="tablist" aria-label="Motion library filter"><button id="library-all" role="tab" aria-selected="true">All <span id="all-count"></span></button><button id="library-cached" role="tab" aria-selected="false">Cached <span id="cached-count"></span></button></div><div id="catalog" class="catalog"></div>
 </section><section id="metadata" class="metadata"></section></aside>
 <div class="top-tools"><button id="orbit" class="on" aria-pressed="true" aria-label="Auto orbit">${icon('orbit')}<span>Orbit</span><kbd>O</kbd></button><button id="reset" aria-label="Reset view and time">${icon('rotate-ccw')}<span>Reset</span><kbd>R</kbd></button></div>
 <section class="display-panel"><div class="small-title">DISPLAY</div>${[['mesh','Robot mesh',true],['skeleton','Skeleton',false],['targets','Input traces',false],...(page==='locomotion'?[]:[['videos','Ego videos',true]])].map(([key,label,checked])=>`<div class="switch-row"><span>${label}</span><button class="switch" role="switch" aria-label="${label}" aria-checked="${checked}" data-display="${key}"></button></div>`).join('')}</section>
 <div id="slots" class="slot-overlay"></div><section id="robot-picker" class="slot-robot-bar" hidden aria-label="Choose robot for motion"></section><section id="source-videos" class="source-videos" hidden></section>
 <div class="transport"><button id="play" class="play" aria-label="Play or pause">${icon('pause')}</button><button id="prev" class="step" aria-label="Previous frame">${icon('skip-back')}</button><span id="time" class="time">0.00 / 0.00</span><input id="timeline" class="timeline" type="range" min="0" max="1" step="0.0333333333333333" value="0" aria-label="Playback time"/><button id="next" class="step" aria-label="Next frame">${icon('skip-forward')}</button><span class="divider"></span><select id="speed" aria-label="Playback speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></div>
 <div id="frame-label" class="frame-label">30 HZ · SHARED TIMELINE · PHYSICAL SCALE</div><div class="notice"><span class="dot"></span>Kinematic preview · Not physics validated</div>
 <div id="toast" class="toast" hidden role="status"></div><div id="modal-root"></div><div id="boot" class="loading-screen">Loading your motion workspace…</div>`;
const el=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;
function refreshIcons(){createIcons({icons});}
refreshIcons();
let catalog:Catalog;let defaultRobot:Robot;let slots:Slot[]=[];let selected:number|null=null;let selectedClip:Clip|undefined;let pendingSelection:{index:number;clip:Clip}|undefined;
let showEgoVideos=true;
let libraryFilter:'all'|'cached'='all';let previousSearch='';const groupOpen=new Map<string,boolean>();
let playing=true,time=0,speed=1,duration=0,last=performance.now(),dragging=false,suppressClick=false;let toastTimer=0;const thumbnails=new Map<string,string>();
const stage=new Stage(el('viewport'),()=>syncOrbit());
// Scene state belongs to this page visit only; reloads always use the presets.
function toast(text:string){el('toast').textContent=text;el('toast').hidden=false;clearTimeout(toastTimer);toastTimer=window.setTimeout(()=>el('toast').hidden=true,6500);}
const request=apiJson;
function robotImage(robot:Robot){return thumbnails.has(robot.id)?`<img class="robot-img" src="${thumbnails.get(robot.id)}" alt="${esc(robot.label)}"/>`:`<div class="robot-img-placeholder">${icon('bot')}</div>`;}
async function warmThumbnails(robots:Robot[]){for(const robot of robots){if(!robot.model_url||thumbnails.has(robot.id))continue;try{thumbnails.set(robot.id,await stage.thumbnail(robot));renderRobot();}catch(error){console.warn('Robot thumbnail unavailable',robot.id,error);}}}
function renderRobot(){
 if(!el('robot-picker').hidden&&pendingSelection)renderPicker();
}
function openRobotPicker(index:number,clip:Clip){
 pendingSelection={index,clip};renderPicker();
 const bar=el('robot-picker');
 if(!matchMedia('(prefers-reduced-motion: reduce)').matches){
  bar.style.transformOrigin='left center';
  bar.classList.remove('unfold');void bar.offsetWidth;bar.classList.add('unfold');
 }
}
function renderPicker(){
 if(!pendingSelection)return;
 const {index,clip}=pendingSelection;
 const robots=robotChoices(catalog.robots);
 el('modal-root').innerHTML='';
 const bar=el('robot-picker');bar.hidden=false;
 bar.innerHTML=`<header><span>Slot 0${index+1} · Robot</span><span class="bar-motion">${esc(clip.title)}</span></header><div class="bar-robots">${robots.map(r=>`<div class="bar-item"><button class="bar-robot ${clip.cache?.[r.id]?'cached':''}" data-pick-robot="${r.id}" aria-label="Use ${esc(r.label)}${clip.cache?.[r.id]?' · Cached':''}" ${!r.conversion_ready?'disabled':''}>${clip.cache?.[r.id]?'<span class="cache-tag">Cached</span>':''}${robotImage(r)}<strong>${esc(r.label)}</strong>${!r.conversion_ready?`<small class="robot-unavailable">${esc(robotStatus(r))}</small>`:''}</button><button class="bar-delete-robot" data-delete-robot="${r.id}" aria-label="Delete ${esc(r.label)}" title="Delete robot and its motion cache">×</button>${r.id.startsWith('custom_')?`<button class="bar-inspect" data-inspect-robot="${r.id}" aria-label="Inspect ${esc(r.label)}">${icon('info')}</button>`:''}</div>`).join('')}<button class="bar-robot bar-upload" id="upload-open" aria-label="Upload robot URDF">${icon('plus')}<strong>Upload robot</strong><small>URDF / ZIP</small></button></div>`;
 positionPicker();
 document.querySelectorAll<HTMLButtonElement>('[data-pick-robot]').forEach(b=>b.onclick=()=>chooseRobot(b.dataset.pickRobot!));
 document.querySelectorAll<HTMLButtonElement>('[data-inspect-robot]').forEach(b=>b.onclick=()=>openInspection(catalog.robots.find(r=>r.id===b.dataset.inspectRobot)!));
 document.querySelectorAll<HTMLButtonElement>('[data-delete-robot]').forEach(b=>b.onclick=async()=>{
  const id=b.dataset.deleteRobot!,robot=catalog.robots.find(r=>r.id===id)!;
  if(!confirm(`Delete ${robot.label} and its generated motion cache? Original SMPL and source data will be kept.`))return;
  b.disabled=true;
  try{const response=await fetch(`/api/v1/robots/${encodeURIComponent(id)}`,{method:'DELETE'});const result=await response.json();if(!response.ok)throw new Error(result.detail||'Deletion failed');location.reload();}
  catch(error){b.disabled=false;alert((error as Error).message);}
 });
 el('upload-open').onclick=openUpload;
 refreshIcons();
}
function closePicker(){pendingSelection=undefined;el('robot-picker').hidden=true;el('robot-picker').innerHTML='';}
function positionPicker(){
 if(!pendingSelection||el('robot-picker').hidden)return;
 const bar=el('robot-picker'),slot=document.querySelector<HTMLElement>(`[data-slot="${pendingSelection.index}"]`);
 if(!slot)return;
 const rect=slot.getBoundingClientRect(),sidebar=document.querySelector('.sidebar')!.getBoundingClientRect();
 const leftEdge=innerWidth>760?sidebar.right+12:12;
 const width=Math.min(870,innerWidth-leftEdge-16);bar.style.width=`${Math.max(240,width)}px`;
 const height=bar.offsetHeight;
 let x=rect.right+12,y=rect.top;
 if(x+width>innerWidth-12){x=rect.left-width-12;if(x<leftEdge){x=Math.max(leftEdge,Math.min(rect.left,innerWidth-width-12));y=rect.bottom+10;}}
 if(y+height>innerHeight-12)y=Math.max(12,innerHeight-height-12);
 bar.style.left=`${Math.max(12,x)}px`;bar.style.top=`${Math.max(12,y)}px`;
}
function resolveRobot(id:string):Robot|undefined{const original=catalog.robots.find(r=>r.id===id);if(!original||original.conversion_ready||!original.report?.selection_key)return original;return robotChoices(catalog.robots).find(r=>r.report?.selection_key===original.report?.selection_key)||original;}
function chooseRobot(id:string){
 if(!pendingSelection)return;
 const robot=catalog.robots.find(r=>r.id===id),choice=pendingSelection;
 if(!robot?.conversion_ready)return;
 closePicker();el('modal-root').innerHTML='';
 void assign(choice.index,choice.clip,robot);
}
function cachedRobotIds(c:Clip){return Object.keys(c.cache||{}).filter(id=>catalog.robots.some(r=>r.id===id&&r.conversion_ready));}
function renderCatalog(){
 const query=el<HTMLInputElement>('search').value;
 if(!previousSearch)document.querySelectorAll<HTMLDetailsElement>('#catalog details[data-group]').forEach(d=>groupOpen.set(d.dataset.group!,d.open));
 previousSearch=query;
 const cached=catalog.clips.filter(c=>cachedRobotIds(c).length>0);
 el('all-count').textContent=String(catalog.clips.length);el('cached-count').textContent=String(cached.length);
 el('library-all').setAttribute('aria-selected',String(libraryFilter==='all'));el('library-cached').setAttribute('aria-selected',String(libraryFilter==='cached'));
 const matches=visibleClips(libraryFilter==='cached'?cached:catalog.clips,query) as Clip[];
 el('catalog-count').textContent=String(matches.length);
 const groups=new Map<string,Map<string,Clip[]>>();matches.forEach(c=>{const [family,subclass]=motionGroup(c);if(!groups.has(family))groups.set(family,new Map());const children=groups.get(family)!;children.set(subclass,[...(children.get(subclass)||[]),c]);});
 const clipButton=(c:Clip)=>`<button class="clip ${selectedClip?.id===c.id?'selected':''}" draggable="false" data-clip="${c.id}" title="${esc(c.title)}" aria-label="Motion ${esc(c.title)}"><span class="clip-copy"><strong>${esc(c.title)}</strong><small><span>${c.duration.toFixed(2)} s</span>${cachedRobotIds(c).length?'<span class="motion-cached">Cached</span>':''}</small></span>${icon('grip-vertical','grip')}</button>`;
 const isOpen=(key:string)=>query||groupOpen.get(key)===true?'open':'';
 el('catalog').innerHTML=Array.from(groups).sort(([a],[b])=>a.localeCompare(b)).map(([family,children])=>`<details class="family" data-group="${esc(family)}" ${isOpen(family)}><summary>${icon('chevron-down')}${esc(family)}<span class="number">${Array.from(children.values()).reduce((n,clips)=>n+clips.length,0)}</span></summary>${children.size===1?Array.from(children.values())[0].map(clipButton).join(''):Array.from(children).sort(([a],[b])=>a.localeCompare(b)).map(([subclass,clips])=>`<details class="subclass" data-group="${esc(family+' / '+subclass)}" ${isOpen(family+' / '+subclass)}><summary>${icon('chevron-down')}${esc(subclass)}<span class="number">${clips.length}</span></summary>${clips.map(clipButton).join('')}</details>`).join('')}</details>`).join('')||`<div class="empty-list">${query?'No matching motions': 'No cached motions yet'}</div>`;
 document.querySelectorAll<HTMLButtonElement>('[data-clip]').forEach(b=>{
   b.onclick=()=>{if(suppressClick)return;selectedClip=catalog.clips.find(c=>c.id===b.dataset.clip);renderMetadata();renderCatalog();};
   b.ondblclick=()=>{const clip=catalog.clips.find(c=>c.id===b.dataset.clip)!;const index=selected??slots.findIndex(s=>!s.clip);if(index<0){toast('All three slots are occupied. Select a slot to replace.');return;}openRobotPicker(index,clip);};
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
   if(started){suppressClick=true;window.setTimeout(()=>suppressClick=false,300);if(target){const i=Number(target.dataset.slot);openRobotPicker(i,clip);}}
 };
 const cancel=(event:PointerEvent)=>{target=null;finish(event);};
 window.addEventListener('pointermove',move,{passive:false});window.addEventListener('pointerup',finish);window.addEventListener('pointercancel',cancel);
}
function metadataSections(c:Clip){const slot=selected!==null?slots[selected]:undefined;const result=slot?.clip?.id===c.id?slot:undefined;
 return {Source:{Family:motionGroup(c)[0],Subclass:motionGroup(c)[1],Duration:`${c.duration.toFixed(2)} s`,Frames:c.frames,'Frame rate':`${c.fps} Hz`,Format:c.source_format,...c.metadata,File:c.source_name},...(result?{Result:{Robot:result.robot.label,Version:result.payload?.version||c.version,Frames:result.payload?.frames,'Quality status':result.payload?.quality_status||'not_evaluated'}}:{})};
}
function renderMetadata(){const c=selectedClip;
 el('metadata').innerHTML=c?`<div class="selection-heading"><span>METADATA</span><button id="motion-details">View details ${icon('arrow-up-right')}</button></div><div class="selection-name" title="${esc(c.title)}">${esc(c.title)}</div><div class="selection-summary"><span>${esc(motionGroup(c)[0])}</span><span>${c.duration.toFixed(2)} s</span><span>${esc(c.source_format.split(' · ')[0])}</span></div>`:'<div class="selection-heading"><span>METADATA</span></div><div class="selection-empty">Select a motion to inspect it</div>';
 if(c)el('motion-details').onclick=()=>openMotionDetails(c);refreshIcons();
}
function openMotionDetails(c:Clip){
 modal(`<div class="modal-head"><h2>Motion details</h2><button id="modal-close" aria-label="Close details">${icon('x')}</button></div><div class="detail-filename">${esc(c.title)}</div><div class="motion-detail-content">${Object.entries(metadataSections(c)).map(([title,values])=>`<details class="metadata-section" open><summary>${esc(title)}</summary><dl class="metadata-fields">${Object.entries(values).filter(([,v])=>v!==undefined&&v!==null&&v!=='').map(([key,value])=>`<div><dt>${esc(key)}</dt><dd>${formatMetadata(value)}</dd></div>`).join('')}</dl></details>`).join('')}${c.description?`<p class="detail-description">${esc(c.description)}</p>`:''}</div>`);
}
function formatMetadata(value:unknown):string{if(Array.isArray(value)){const text=value.map(formatMetadata).join(', ');return value.length>6?`<details class="metadata-array"><summary>${value.length} items</summary><div>${text}</div></details>`:text;}if(value!==null&&typeof value==='object')return `<div class="metadata-nested">${Object.entries(value).map(([k,v])=>`<div><span>${esc(k)}</span>: ${formatMetadata(v)}</div>`).join('')}</div>`;return esc(value);}
function renderSlots(){
 el('slots').innerHTML=slots.map(s=>`<div class="slot-label ${selected===s.index?'active':''} ${dragging?'drag-mode':''}" data-slot="${s.index}" role="button" tabindex="0" aria-label="Slot ${s.index+1}${s.clip?' '+esc(s.clip.title):' Add motion'}">
 ${s.clip?`<div class="slot-header"><span class="slot-number">0${s.index+1}</span><span class="slot-name">${esc(s.clip.title)}</span><button class="remove" data-remove="${s.index}" aria-label="Remove slot ${s.index+1}">${icon('x')}</button></div><div class="slot-subtitle"><button class="slot-robot" data-switch-robot="${s.index}" aria-label="Change robot for slot ${s.index+1}">${esc(s.robot.label)} ${icon('chevron-down')}</button> &nbsp;·&nbsp; ${s.payload?.duration.toFixed(2)||s.clip.duration.toFixed(2)} s</div>${s.payload?'<div class="quality-chip" title="Motion quality not evaluated" aria-label="Motion quality not evaluated"></div>':''}`:`<div class="slot-add">${icon('plus')}<span>Drop a motion</span><small>SLOT 0${s.index+1}</small></div>`}
 ${!['empty','ready'].includes(s.status)?`<div class="slot-progress ${s.status==='failed'?'error':''}">${s.status==='failed'?'': '<span class="tiny-spinner"></span>'}${esc(s.message||({loading:'Loading motion',queued:'Queued for local conversion',running:'Retargeting motion',failed:'Conversion failed'} as Record<string,string>)[s.status])}</div>`:''}</div>`).join('');
 document.querySelectorAll<HTMLElement>('[data-slot]').forEach(node=>{
   const i=Number(node.dataset.slot);node.onclick=()=>selectSlot(i);node.onkeydown=e=>{if(e.key==='Enter')selectSlot(i);};
   node.ondragover=e=>{if(e.dataTransfer?.types.includes('application/x-greenwich-clip')){e.preventDefault();e.dataTransfer.dropEffect='copy';node.classList.add('drag-over');}};
   node.ondragleave=()=>node.classList.remove('drag-over');node.ondrop=e=>{e.preventDefault();node.classList.remove('drag-over');const id=e.dataTransfer?.getData('application/x-greenwich-clip');const clip=catalog.clips.find(c=>c.id===id);dragging=false;if(clip)openRobotPicker(i,clip);else renderSlots();};
 });
 document.querySelectorAll<HTMLButtonElement>('[data-remove]').forEach(b=>b.onclick=e=>{e.stopPropagation();const i=Number(b.dataset.remove);slots[i].token++;slots[i].clip=undefined;slots[i].payload=undefined;slots[i].status='empty';slots[i].message=undefined;stage.remove(i);selected=null;updateDuration();renderSlots();renderRobot();renderMetadata();});
 document.querySelectorAll<HTMLButtonElement>('[data-switch-robot]').forEach(b=>b.onclick=e=>{e.stopPropagation();const i=Number(b.dataset.switchRobot);if(slots[i].clip)openRobotPicker(i,slots[i].clip!);});
 refreshIcons();positionLabels();renderVideos();
}
function trackingPanel(s:Slot){
 if(!stage.targets)return '';
 const t=s.status==='ready'?s.payload?.tracking:undefined;
 if(!t)return `<div class="tracking-unavailable">${s.status==='ready'?'Wrist tracking metrics unavailable':'Tracking metrics available after conversion'}</div>`;
 const chart=(series:number[][],threshold:number,unit:string)=>{const plot=errorChart(series,threshold);return `<div class="tracking-chart"><span>${unit}</span><svg viewBox="0 0 260 24" preserveAspectRatio="none" aria-label="${unit==='mm'?'Wrist position error in millimetres':'Wrist orientation error in degrees'}"><line class="tracking-threshold" x1="0" x2="260" y1="${plot.thresholdY}" y2="${plot.thresholdY}"/><path class="tracking-left" d="${plot.paths[0]}"/><path class="tracking-right" d="${plot.paths[1]}"/><line class="tracking-cursor" x1="0" x2="0" y1="0" y2="24"/></svg><small>${threshold}</small></div>`;};
 return `<div class="wrist-tracking" data-tracking-slot="${s.index}" title="${esc(t.reference)}. Dashed limits: 10 mm / 15°. These are measurements, not a physical-feasibility certificate."><div class="tracking-values"><span class="tracking-left-label">L <b data-position="0">—</b> mm · <b data-angle="0">—</b>°</span><span class="tracking-right-label">R <b data-position="1">—</b> mm · <b data-angle="1">—</b>°</span></div>${chart(t.position_error_mm,t.thresholds.position_mm,'mm')}${chart(t.orientation_error_deg,t.thresholds.orientation_deg,'°')}</div>`;
}
function syncTracking(){
 document.querySelectorAll<HTMLElement>('[data-tracking-slot]').forEach(panel=>{const slot=slots[Number(panel.dataset.trackingSlot)],p=slot?.payload,t=p?.tracking;if(!p||!t||slot.status!=='ready')return;
 const frame=sampleFrame(time,p.fps,p.frames);if(panel.dataset.frame===String(frame))return;panel.dataset.frame=String(frame);
 for(let hand=0;hand<2;hand++){const position=panel.querySelector<HTMLElement>(`[data-position="${hand}"]`)!,angle=panel.querySelector<HTMLElement>(`[data-angle="${hand}"]`)!;const mm=t.position_error_mm[frame][hand],deg=t.orientation_error_deg[frame][hand];position.textContent=mm.toFixed(1);angle.textContent=deg.toFixed(1);position.classList.toggle('over-limit',mm>t.thresholds.position_mm);angle.classList.toggle('over-limit',deg>t.thresholds.orientation_deg);}
 panel.querySelectorAll('.tracking-cursor').forEach(cursor=>cursor.setAttribute('transform',`translate(${frame*260/Math.max(1,p.frames-1)} 0)`));
 });
}
function renderVideos(){
 const strip=el('source-videos');strip.hidden=page==='locomotion'||!showEgoVideos;if(strip.hidden){strip.querySelectorAll('video').forEach(v=>v.pause());return;}
 const key=String(stage.targets)+'|'+slots.map(s=>[s.clip?.id,s.robot.id,s.status,s.token,Boolean(s.payload?.tracking)].join(':')).join('|');if(strip.dataset.key===key)return;strip.dataset.key=key;
 strip.innerHTML=slots.map(s=>`<button class="source-video-tile" data-video-slot="${s.index}" aria-label="Enlarge source video for slot ${s.index+1}"><header><b>0${s.index+1}</b><span>${esc(s.clip?.title||'Empty slot')}</span><small>EGO</small></header>${s.clip?.video?`<video muted playsinline preload="metadata" src="${esc(s.clip.video.url)}" data-source-slot="${s.index}"></video>`:`<div class="video-missing">${s.clip?'Source video unavailable':'Drop a motion to compare'}</div>`}${trackingPanel(s)}</button>`).join('');
 strip.querySelectorAll<HTMLButtonElement>('[data-video-slot]').forEach(b=>b.onclick=()=>{const i=Number(b.dataset.videoSlot),s=slots[i];if(!s.clip?.video){toast('No verified source video for this clip');return;}selectSlot(i);modal(`<div class="modal-head"><h2>Slot 0${i+1} · Source Ego</h2><button id="modal-close" aria-label="Close">${icon('x')}</button></div><video class="source-video-large" muted playsinline preload="metadata" src="${esc(s.clip.video.url)}" data-source-slot="${i}"></video><p class="helper">${esc(s.clip.video.sync_method)} · Shared timeline</p>`);});
 strip.querySelectorAll('video').forEach(v=>v.addEventListener('error',()=>{v.replaceWith(Object.assign(document.createElement('div'),{className:'video-missing',textContent:'Source video could not be loaded'}));}));
}
function syncVideos(){
 document.querySelectorAll<HTMLVideoElement>('video[data-source-slot]').forEach(v=>{
   if(v.closest('#source-videos')&&!showEgoVideos){v.pause();return;}
   const slot=slots[Number(v.dataset.sourceSlot)],clip=slot?.clip,descriptor=clip?.video;
   if(!clip||!descriptor||!slot.payload||slot.status!=='ready'){v.pause();return;}
   const local=Math.min(time,slot.payload.duration,clip.duration),frame=sampleFrame(local,clip.fps,clip.frames);
   const target=descriptor.frame_times?.[frame]??descriptor.start_seconds+local;
   const bounded=Number.isFinite(v.duration)?Math.min(target,Math.max(0,v.duration-1/30)):target;
   if(v.readyState>=1&&!v.seeking&&Math.abs(v.currentTime-bounded)>(playing?.08:.012))v.currentTime=bounded;
   v.playbackRate=speed;
   const run=playing&&time<Math.min(slot.payload.duration,clip.duration);
   if(run&&v.paused)void v.play().catch(()=>{});else if(!run&&!v.paused)v.pause();
 });
}
function positionLabels(){const videos=document.querySelector<HTMLElement>('.source-videos');const limit=videos&&videos.getBoundingClientRect().height>0?videos.getBoundingClientRect().top-94:innerHeight-180;document.querySelectorAll<HTMLElement>('[data-slot]').forEach(n=>{const i=Number(n.dataset.slot);const p=stage.screenAnchor(i);n.style.left=`${p.x}px`;n.style.top=`${Math.min(p.y+16,limit)}px`;});}
function selectSlot(i:number){selected=i;selectedClip=slots[i].clip||selectedClip;renderSlots();renderRobot();renderMetadata();renderCatalog();}
function updateDuration(){duration=Math.max(0,...slots.map(s=>s.payload?.duration||0));el<HTMLInputElement>('timeline').max=String(duration||1);time=Math.min(time,duration);}
async function assign(i:number,clip:Clip,robot:Robot){
 const slot=slots[i],token=++slot.token,previous={clip:slot.clip,robot:slot.robot,payload:slot.payload};
 slot.clip=clip;slot.robot=robot;slot.status='loading';slot.message=undefined;selected=i;selectedClip=clip;stage.autoRotate=false;syncOrbit();renderSlots();renderRobot();renderMetadata();renderCatalog();
 try{
   let result=await api.start({workflow:'locomotion',recording_id:clip.id,robot_id:robot.id});
   while(result.status!=='completed'){
     if(token!==slot.token)return;if(result.status==='failed')throw new Error(result.error||'Conversion failed');
     slot.status=result.status;slot.message=result.status==='queued'?`Queued · Position ${result.queue_position||1} · Waiting for GPU`:'Generating · GPU running';renderSlots();
     await new Promise(resolve=>setTimeout(resolve,1000));result=await api.job(result.id);
   }
   if(token!==slot.token)return;const payload:Playback=await request(result.url);if(token!==slot.token)return;
   await stage.load(i,robot,payload,()=>token===slot.token);if(token!==slot.token)return;
   clip.cache={...clip.cache,[robot.id]:result.url};slot.payload=payload;slot.status='ready';slot.message=undefined;time=0;updateDuration();renderSlots();renderMetadata();renderCatalog();
 }catch(error){if(token!==slot.token)return;slot.status='failed';slot.message=(error as Error).message;if(previous.payload){slot.clip=previous.clip;slot.robot=previous.robot;slot.payload=previous.payload;slot.message+=' · Previous motion retained';}renderSlots();renderRobot();renderMetadata();toast(slot.message||'Loading failed');}
}
function syncOrbit(){el('orbit').classList.toggle('on',stage.autoRotate);el('orbit').setAttribute('aria-pressed',String(stage.autoRotate));}
function syncPlay(){el('pause')&&(el('pause').innerHTML=`${icon(playing?'pause':'play')}<span>${playing?'Pause':'Play'}</span><kbd>Space</kbd>`);el('play').innerHTML=icon(playing?'pause':'play');refreshIcons();}
function togglePlay(){playing=!playing;syncPlay();}
function step(direction:number){playing=false;time=Math.max(0,Math.min(duration,time+direction/30));stage.autoRotate=false;syncPlay();syncOrbit();}
function modal(content:string){el('modal-root').innerHTML=`<div class="modal-backdrop"><section class="modal" role="dialog" aria-modal="true">${content}</section></div>`;refreshIcons();el('modal-close')?.addEventListener('click',()=>el('modal-root').innerHTML='');document.querySelector('.modal-backdrop')?.addEventListener('click',e=>{if(e.target===e.currentTarget)el('modal-root').innerHTML='';});}
function openInspection(robot:Robot){
 const report=robot.report as any;const editable=robot.id.startsWith('custom_')&&robot.conversion_ready&&report?.endpoints;
 el('robot-picker').hidden=true;modal(`<div class="modal-head"><h2>${esc(robot.label)}</h2><button id="modal-close" aria-label="Close">${icon('x')}</button></div><p>${esc(report?.message||robotStatus(robot))}</p><pre>${esc(JSON.stringify(report,null,2))}</pre>${editable?`<details><summary>Physical endpoint / wrist frames</summary><p class="helper">Axes are fixed in each vendor wrist-link frame. Editing invalidates cached results; rerun the motion afterward.</p><textarea id="physical-metadata" aria-label="Physical robot metadata" style="width:100%;height:220px;background:#181818;color:#ddd">${esc(JSON.stringify({endpoints:report.endpoints,palm_frames:report.palm_frames},null,2))}</textarea><button id="save-physical" class="primary">Save physical metadata</button><div id="physical-result" role="status"></div></details>`:''}`);
 if(pendingSelection){const back=document.createElement('button');back.className='picker-back';back.textContent='Back to robot selection';back.onclick=renderPicker;document.querySelector('.modal')!.append(back);}
 el('save-physical')?.addEventListener('click',async()=>{const result=el('physical-result');try{const updated:Robot=await request(`/api/v1/robots/${robot.id}/metadata`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(JSON.parse(el<HTMLTextAreaElement>('physical-metadata').value))});Object.assign(robot,updated);result.textContent='Saved. Rerun motions to use the new frames.';}catch(error){result.textContent=(error as Error).message;}});
}
function openUpload(){
 el('robot-picker').hidden=true;modal(`<div class="modal-head"><h2>Add your robot.</h2><button id="modal-close" aria-label="Close">${icon('x')}</button></div><p>Upload a URDF, or a ZIP containing the URDF and meshes.<br/>Models stay in your local workspace and join the library after inspection.</p><label class="upload-drop" for="robot-file">${icon('upload')}<strong>Choose a file or drop a robot package here</strong><small>.urdf / .zip · Up to 150 MB</small></label><input id="robot-file" type="file" accept=".urdf,.zip"/><div id="upload-result" class="result"></div><p class="helper">Humanoid topology, endpoints and neutral wrist frames are inferred automatically. Unsupported or ambiguous models receive a specific inspection report.</p>`);
 if(pendingSelection){const back=document.createElement('button');back.className='picker-back';back.textContent='Back to robot selection';back.onclick=renderPicker;document.querySelector('.modal')!.append(back);}
 const file=el<HTMLInputElement>('robot-file');file.onchange=()=>{if(file.files?.[0])void uploadRobot(file.files[0]);};
 const drop=document.querySelector<HTMLElement>('.upload-drop')!;drop.ondragover=e=>e.preventDefault();drop.ondrop=e=>{e.preventDefault();if(e.dataTransfer?.files[0])void uploadRobot(e.dataTransfer.files[0]);};
}
async function uploadRobot(file:File){const result=el('upload-result');result.textContent='Checking URDF and mesh assets…';try{
 const form=new FormData();form.append('file',file);const robot:Robot=await request(`/api/v1/robots/import/${page}`,{method:'POST',body:form});catalog.robots.push(robot);void warmThumbnails([robot]);
 result.innerHTML=`${esc(robot.label)} · ${esc(robotStatus(robot))}.<br/>${esc(robot.report?.message||'Model inspection complete')}<div class="modal-buttons"><button class="primary" id="upload-done">View robot</button></div>`;
 el('upload-done').textContent='Back to robot selection';el('upload-done').onclick=()=>renderPicker();
 }catch(error){result.textContent=(error as Error).message;}}

el('library-all').onclick=()=>{libraryFilter='all';renderCatalog();};el('library-cached').onclick=()=>{libraryFilter='cached';renderCatalog();};
el('search').addEventListener('input',()=>renderCatalog());el('play').onclick=togglePlay;el('prev').onclick=()=>step(-1);el('next').onclick=()=>step(1);
el('orbit').onclick=()=>{stage.autoRotate=!stage.autoRotate;syncOrbit();};el('reset').onclick=()=>{stage.reset();time=0;};
el<HTMLInputElement>('timeline').oninput=e=>{time=Number((e.target as HTMLInputElement).value);playing=false;stage.autoRotate=false;syncPlay();syncOrbit();};
el<HTMLSelectElement>('speed').onchange=e=>speed=Number((e.target as HTMLSelectElement).value);
document.querySelectorAll<HTMLButtonElement>('[data-display]').forEach(b=>b.onclick=()=>{if(b.dataset.display==='videos'){showEgoVideos=!showEgoVideos;b.setAttribute('aria-checked',String(showEgoVideos));renderVideos();return;}const key=b.dataset.display as 'mesh'|'skeleton'|'targets';stage[key]=!stage[key];b.setAttribute('aria-checked',String(stage[key]));stage.visibility();if(key==='targets')renderVideos();});
document.addEventListener('pointerdown',e=>{const bar=el('robot-picker');if(!bar.hidden&&!bar.contains(e.target as Node))closePicker();});
document.addEventListener('keydown',e=>{if((e.target as HTMLElement)?.matches('input,select,textarea'))return;if(e.code==='Space'){e.preventDefault();togglePlay();}if(e.key.toLowerCase()==='o'){stage.autoRotate=!stage.autoRotate;syncOrbit();}if(e.key.toLowerCase()==='r'){stage.reset();time=0;}if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1);if(e.key==='Escape'){el('modal-root').innerHTML='';closePicker();}});
el('viewport').addEventListener('click',e=>{const hit=stage.pick(e.clientX,e.clientY);if(hit!==null){selectSlot(hit);return;}selected=null;if(catalog){renderRobot();renderSlots();renderMetadata();}});
document.addEventListener('dragover',e=>{if(e.dataTransfer?.types.includes('application/x-greenwich-clip')){e.preventDefault();dragging=true;}});
function animate(){requestAnimationFrame(animate);const now=performance.now();const dt=Math.min((now-last)/1000,.1);last=now;if(playing&&duration>0){time+=dt*speed;if(time>duration)time=0;}
 stage.frame(time);positionLabels();positionPicker();syncVideos();syncTracking();el('time').textContent=`${time.toFixed(2)} / ${duration.toFixed(2)}`;el<HTMLInputElement>('timeline').value=String(time);el('frame-label').textContent=`FRAME ${Math.floor(time*30).toString().padStart(3,'0')} · 30 HZ`;
}
async function boot(){try{await api.session();catalog=await request(`/api/v1/catalogs/${page}`);defaultRobot=catalog.robots.find(r=>r.id==='g1')||catalog.robots[0];if(!defaultRobot)throw new Error('No robot configured for this pipeline');
 slots=Array.from({length:3},(_,index)=>({index,robot:defaultRobot,status:'empty',token:0}));
 selectedClip=catalog.clips[0];renderCatalog();renderRobot();renderMetadata();renderSlots();el('boot').remove();animate();void warmThumbnails(catalog.robots);
 if(catalog.presets?.length){await Promise.all(catalog.presets.slice(0,3).map(async(s,i)=>{const clip=catalog.clips.find(c=>c.id===s.clip_id),robot=resolveRobot(s.robot_id);if(clip&&robot)await assign(i,clip,robot);}));}
 else if(catalog.clips.length){const clip=catalog.clips.find(c=>c.cache&&Object.keys(c.cache).length>=2)||catalog.clips[0];await assign(0,clip,defaultRobot);const other=catalog.robots.find(r=>r.id!==defaultRobot.id&&clip.cache?.[r.id]);if(other)await assign(1,clip,other);}
 time=0;
 setInterval(async()=>{try{const fresh:Catalog=await request(`/api/v1/catalogs/${page}`);if(dragging)return;let changed=false;for(const clip of catalog.clips){const cache=fresh.clips.find(c=>c.id===clip.id)?.cache;if(JSON.stringify(clip.cache)!==JSON.stringify(cache)){clip.cache=cache;changed=true;}}if(changed)renderCatalog();}catch{}},15000);
 selected=null;renderRobot();renderSlots();renderMetadata();stage.autoRotate=true;syncOrbit();
 }catch(error){el('boot').innerHTML=`<div class="fatal">Workspace could not be loaded<br/>${esc((error as Error).message)}<br/><small>Run prepare_motion_lab.py before starting the local service.</small></div>`;}}
void boot();
