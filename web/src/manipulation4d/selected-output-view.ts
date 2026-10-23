import type {SceneAsset} from './types';

/** FOV is a deliverable even when its internal trajectory was not requested. */
export function bindPatchVideo(asset:Pick<SceneAsset,'patch_videos'|'patch_kinds'|'patch_unavailable'|'generation_status'>, preferredId:string, allowedIds:ReadonlySet<string>, video:HTMLVideoElement, empty:HTMLElement, label:HTMLElement):string|undefined {
 const ids=[preferredId,asset.generation_status?.robot_id,...Object.keys(asset.patch_videos??{})];
 const id=ids.find((candidate):candidate is string=>!!candidate&&allowedIds.has(candidate)&&!!asset.patch_videos?.[candidate]);
 const url=id?asset.patch_videos?.[id]:undefined;
 video.hidden=!url;empty.hidden=!!url;
 if(url)video.src=url;else{video.pause();video.removeAttribute('src');video.load();}
 const kind=id?asset.patch_kinds?.[id]:undefined;
 label.textContent=url?(kind==='native_preview'?'ROBOT PREVIEW':kind==='estimated_camera_composite'?'ESTIMATED CAMERA · SYNTHETIC':'SYNTHETIC · INPAINTED'):'NOT GENERATED';
 if(!url)empty.textContent=asset.patch_unavailable?.[preferredId]||'No robot composite generated for this recording';
 return id;
}

export interface ContactSeries {fps:number;likelihood:(number|null)[][][];model?:string;hand_contact?:{state:string;object:string;evidence:string}[][];observations?:{time:number;evidence:string;uncertainty:string}[];}
/** Source contact updates are independent of native robot playback. */
export function drawContactPanel(contact:ContactSeries|undefined,proximity:ContactSeries|undefined,renderTime:number,document:Document):void {
 const $=<T extends HTMLElement=HTMLElement>(id:string)=>document.getElementById(id) as T;
 if(contact?.likelihood.length){const isProximity=$<HTMLSelectElement>('contact-mode').value==='proximity'&&!!proximity;const series=isProximity?proximity!:contact;const sample=Math.min(series.likelihood.length-1,Math.floor(renderTime*series.fps));const videoSample=Math.min(contact.likelihood.length-1,Math.floor(renderTime*contact.fps));const v=series.likelihood[sample];const hands=contact.hand_contact?.[videoSample];$('tactile-kind').textContent=isProximity?'GEOMETRY INDEX · 15 Hz':'VIDEO ESTIMATE · 2 Hz';$('tactile-note').textContent=isProximity?'Independent fingertip proximity indices (0–1), not pressure or contact probabilities. Estimated depth can shift these values.':'Independent video contact scores (0–1); gray / — means unknown. No pressure or force measurement.';for(let si=0;si<2;si++){const hand=hands?.[si];const label=document.getElementById(`contact-hand-${si}`);if(label)label.textContent=isProximity?(si===0?'LEFT':'RIGHT'):`${si===0?'LEFT':'RIGHT'} · video: ${hand?.state??'unknown'}${hand?.object?' / '+hand.object:''}`;document.querySelector(`[data-hand-shape="${si}"]`)?.setAttribute('fill','#203d2a');}const evidence=contact.observations?.[videoSample];$('contact-evidence').textContent=isProximity?'Geometry diagnostic · source fingertips to visible object surfaces. Whole-hand video states are separate.':evidence?`${evidence.time.toFixed(1)}s sample · ${evidence.evidence} ${v.flat().filter(x=>x===null).length}/10 fingertips unknown.`:'';for(let side=0;side<2;side++)for(let finger=0;finger<5;finger++){const node=document.querySelector<SVGElement>(`[data-contact="${side}-${finger}"]`);if(node){const probability=v[side][finger];node.setAttribute('fill',probability===null?'#2c3b30':`rgb(${Math.round(55+170*probability)},${Math.round(80+130*probability)},${Math.round(60+40*probability)})`);node.setAttribute('aria-label',probability===null?'Unknown / not visible':`${isProximity?'Fingertip proximity index':'Video contact score (uncalibrated)'} ${probability.toFixed(2)}`);const meter=document.querySelector<HTMLMeterElement>(`[data-digit-meter="${side}-${finger}"]`);const value=document.querySelector<HTMLElement>(`[data-digit-score="${side}-${finger}"]`);if(meter){meter.value=probability??0;meter.classList.toggle('unknown',probability===null);}if(value)value.textContent=probability===null?'—':probability.toFixed(2);}}}
}
