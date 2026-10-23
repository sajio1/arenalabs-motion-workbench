/** Shared browser client. UI modules never submit to a worker-specific endpoint. */
export type RunSpec={workflow:'manipulation'|'locomotion'|'export';recording_id:string;robot_id:string;hand_id?:string};
type Job={id:string;status:string;result:Record<string,any>;error?:{detail?:string;code?:string}|null};
export type UiJob={id:string;status:'queued'|'running'|'completed'|'failed';error?:string;[key:string]:any};

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
export async function apiJson<T=any>(url:string,options:RequestInit={}):Promise<T>{
 const r=await fetch(mediaUrl(url),{credentials:'same-origin',cache:'no-store',...options});
 const data=await r.json().catch(()=>({detail:`${r.status}: ${r.statusText}`}));
 if(!r.ok)throw new Error(typeof data.detail==='string'?data.detail:data.field_errors?.map((e:any)=>e.msg).join('; ')||data.title||'Request failed');
 return normalizeUrls(data);
}
function uiJob(job:Job):UiJob{
 return {...job.result,id:job.id,status:job.status==='succeeded'?'completed':job.status==='dispatching'?'queued':
  ['interrupted','cancelled','outcome_unknown'].includes(job.status)?'failed':job.status as UiJob['status'],
  error:job.error?.detail||job.error?.code};
}
let sessionPromise:Promise<unknown>|undefined;
const pending=new Map<string,{key:string;promise?:Promise<UiJob>}>();
export const api={
 session(){return sessionPromise??=(apiJson('/api/v1/sessions',{method:'POST'}).catch(e=>{sessionPromise=undefined;throw e;}));},
 async recordings<T>():Promise<T[]>{
  await this.session();const items:T[]=[];let offset=0;
  do{const page=await apiJson<{items:T[];total:number}>(`/api/v1/recordings?limit=500&offset=${offset}`);
   items.push(...page.items);offset+=page.items.length;if(offset>=page.total||!page.items.length)break;
  }while(true);return items;
 },
 async start(spec:RunSpec):Promise<UiJob>{
  await this.session();const identity=JSON.stringify(spec);
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
   .finally(()=>{current.promise=undefined;});
  return current.promise;
 },
 async job(id:string):Promise<UiJob>{return uiJob(await apiJson<Job>('/api/v1/jobs/'+encodeURIComponent(id)));},
};
