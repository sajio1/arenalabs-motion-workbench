/** Shared browser client. UI modules never submit to a worker-specific endpoint. */
export type TrajectoryScale='preserve'|'match_robot';
export type OutputModality='trajectory'|'point_cloud'|'contact'|'inpaint_fov';
export type OutputPlan={requested:OutputModality[]|null;effective:OutputModality[]|null;available:{id:OutputModality;available:boolean;reason?:string|null}[];dependencies:Record<string,unknown>;error?:string|null;unavailable:OutputModality[];defaulted:boolean};
export type RunSpec={workflow:'manipulation'|'locomotion'|'export';recording_id:string;robot_id:string;hand_id?:string;trajectory_scale?:TrajectoryScale|null;output_modalities?:OutputModality[]|null};
export type RunPlan={executable:boolean;required_choice?:'trajectory_scale'|null;choices:string[];outputs:OutputPlan;policy:{trajectory_scale?:TrajectoryScale|null;scale_choice_required?:boolean;scale_estimate_unavailable?:boolean;source_policy?:string;source_policy_version?:string|null;morphology?:{available:boolean;source_height_m?:number;target_height_m?:number;scale_factor?:number;large_mismatch?:boolean;reason?:string}}};
type Job={id:string;status:string;result:Record<string,any>;error?:{detail?:string;code?:string}|null};
export type UiJob={id:string;status:'queued'|'running'|'completed'|'failed';error?:string;robot_ready?:boolean;scene_ready?:boolean;contact_ready?:boolean;fov_ready?:boolean;completed_outputs?:string[];[key:string]:any};
export function hasPublishedOutputs(job:{robot_ready?:boolean;scene_ready?:boolean;contact_ready?:boolean;fov_ready?:boolean;completed_outputs?:string[]}){
 return !!(job.robot_ready||job.scene_ready||job.contact_ready||job.fov_ready||job.completed_outputs?.length);
}

export function mediaUrl(url:string):string{
 if(url.startsWith('/assets/'))return '/api/v1/media/'+url.slice('/assets/'.length);
 if(url.startsWith('/api/source-video/'))return '/api/v1/recordings/'+url.slice('/api/source-video/'.length)+'/source-video';
 return url;
}
export function canonicalAssetUrl(url:string):string{
 return url.startsWith('/api/v1/media/')?'/assets/'+url.slice('/api/v1/media/'.length):url;
}
export function normalizeUrls<T>(value:T):T{
 // Motion buffers are large numeric arrays. Only walk metadata-bearing objects.
 if(typeof value==='string')return mediaUrl(value) as T;
 if(Array.isArray(value)){
  if(!value.length||Array.isArray(value[0])||typeof value[0]==='number')return value;
  return value.map(normalizeUrls) as T;
 }
 if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([k,v])=>[k,normalizeUrls(v)])) as T;
 return value;
}
export class ApiError extends Error{
 constructor(public status:number,public problem:any){super(typeof problem.detail==='string'?problem.detail:problem.detail?.message||problem.field_errors?.map((e:any)=>e.msg).join('; ')||problem.title||'Request failed');}
}
export async function apiJson<T=any>(url:string,options:RequestInit={}):Promise<T>{
 const r=await fetch(mediaUrl(url),{credentials:'same-origin',cache:'no-store',...options});
 const data=await r.json().catch(()=>({detail:`${r.status}: ${r.statusText}`}));
 if(!r.ok)throw new ApiError(r.status,data);
 return normalizeUrls(data);
}
function uiJob(job:Job):UiJob{
 return {...job.result,id:job.id,status:job.status==='succeeded'?'completed':job.status==='dispatching'?'queued':
  ['interrupted','cancelled','outcome_unknown'].includes(job.status)?'failed':job.status as UiJob['status'],
  error:job.error?.detail||job.error?.code};
}
let sessionPromise:Promise<unknown>|undefined;
const pending=new Map<string,{key:string;promise?:Promise<UiJob>}>();
function submissionIdentity(spec:RunSpec){
 // Property order must not change a retry key; scale choices must change it.
 return JSON.stringify(Object.fromEntries(Object.entries(spec).filter(([,v])=>v!==undefined&&v!==null).map(([k,v])=>[k,k==='output_modalities'&&Array.isArray(v)?[...v].sort():v]).sort(([a],[b])=>String(a).localeCompare(String(b)))));
}
export const api={
 session(){return sessionPromise??=(apiJson('/api/v1/sessions',{method:'POST'}).catch(e=>{sessionPromise=undefined;throw e;}));},
 async recordings<T>():Promise<T[]>{
  await this.session();const items:T[]=[];let offset=0;
  do{const page=await apiJson<{items:T[];total:number}>(`/api/v1/recordings?limit=500&offset=${offset}`);
   items.push(...page.items);offset+=page.items.length;if(offset>=page.total||!page.items.length)break;
  }while(true);return items;
 },
 async plan(spec:RunSpec):Promise<RunPlan>{await this.session();return apiJson('/api/v1/plans',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(spec)});},
 async start(spec:RunSpec):Promise<UiJob>{
  await this.session();const identity=submissionIdentity(spec);
  let submission=pending.get(identity);
  if(submission?.promise)return submission.promise;
  // Retain the key after an ambiguous network failure, including a page reload.
  // Deliberate retry of a confirmed failed job gets a new key.
  const storageKey='arena-api-submit:'+identity;
  let saved:string|null=null;try{saved=sessionStorage.getItem(storageKey);}catch{}
  if(!submission){submission={key:saved||crypto.randomUUID()};pending.set(identity,submission);}
  try{sessionStorage.setItem(storageKey,submission.key);}catch{}
  const current=submission;
  current.promise=apiJson<Job>('/api/v1/runs',{method:'POST',headers:{'Content-Type':'application/json','Idempotency-Key':current.key},body:JSON.stringify(spec)})
   .then(job=>{if(job.status!=='outcome_unknown'){pending.delete(identity);try{sessionStorage.removeItem(storageKey);}catch{}}return uiJob(job);})
   .catch(error=>{
    // A structured 4xx rejection is known not to have dispatched. Keep keys for
    // network/5xx ambiguity, but permit corrected preflight inputs to be retried.
    if(error instanceof ApiError&&error.status>=400&&error.status<500){pending.delete(identity);try{sessionStorage.removeItem(storageKey);}catch{}if(error.status===401)sessionPromise=undefined;}
    throw error;
   })
   .finally(()=>{current.promise=undefined;});
  return current.promise;
 },
 async job(id:string):Promise<UiJob>{return uiJob(await apiJson<Job>('/api/v1/jobs/'+encodeURIComponent(id)));},
};
