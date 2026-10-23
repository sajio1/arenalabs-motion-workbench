import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import ts from 'typescript';

let sequence=0;
async function client(t,handler){
 const previous={fetch:globalThis.fetch,sessionStorage:globalThis.sessionStorage};
 const values=new Map();
 globalThis.sessionStorage={getItem:k=>values.get(k)??null,setItem:(k,v)=>values.set(k,v),removeItem:k=>values.delete(k)};
 globalThis.fetch=handler;
 t.after(()=>{globalThis.fetch=previous.fetch;if(previous.sessionStorage===undefined)delete globalThis.sessionStorage;else globalThis.sessionStorage=previous.sessionStorage;});
 const text=await readFile(new URL('../src/platform-api.ts',import.meta.url),'utf8');
 const code=ts.transpileModule(text,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
 return import('data:text/javascript;base64,'+Buffer.from(code).toString('base64')+'#'+sequence++);
}
const response=value=>new Response(JSON.stringify(value),{headers:{'Content-Type':'application/json'}});
const spec={workflow:'manipulation',recording_id:'clip',robot_id:'robot',hand_id:'none'};

test('simultaneous submissions share a session, a request and its task identity',async t=>{
 const calls=[];let finish;
 const {api}=await client(t,async(url,options)=>{
  calls.push([url,options]);
  if(url.endsWith('/sessions'))return response({});
  return new Promise(resolve=>{finish=()=>resolve(response({id:'one',status:'queued',result:{revision:3}}));});
 });
 const first=api.start(spec),second=api.start({...spec});
 while(!finish)await new Promise(resolve=>setImmediate(resolve));finish();
 const [a,b]=await Promise.all([first,second]);
 assert.equal(calls.filter(([url])=>url.endsWith('/sessions')).length,1);
 assert.equal(calls.filter(([url])=>url.endsWith('/runs')).length,1);
 assert.equal(a.id,b.id);assert.equal(a.revision,3);
 assert.equal(JSON.parse(calls[1][1].body).hand_id,'none');
});

test('ambiguous network failure retains the same submission key for retry',async t=>{
 const keys=[];
 const {api}=await client(t,async(url,options)=>{
  if(url.endsWith('/sessions'))return response({});
  keys.push(options.headers['Idempotency-Key']);
  if(keys.length===1)throw new TypeError('Network unavailable');
  return response({id:'same-server-task',status:'succeeded',result:{}});
 });
 await assert.rejects(api.start(spec),/Network unavailable/);
 assert.equal((await api.start(spec)).status,'completed');
 assert.equal(keys[0],keys[1]);
});

test('media transport preserves numeric motion buffers and canonical geometry references',async t=>{
 const {normalizeUrls,canonicalAssetUrl}=await client(t,()=>{});
 const points=[[1,2,3],[4,5,6]],value=normalizeUrls({points,url:'/assets/scene/playback.json'});
 assert.equal(value.points,points);
 assert.equal(value.url,'/api/v1/media/scene/playback.json');
 assert.equal(canonicalAssetUrl('/api/v1/media/scene/scene.json'),'/assets/scene/scene.json');
 assert.equal(canonicalAssetUrl('/assets/scene/geometry/index.json'),'/assets/scene/geometry/index.json');
});

test('interrupted and unknown tasks are not presented as completed',async t=>{
 let status='interrupted';
 const {api}=await client(t,async()=>response({id:'job',status,error:{detail:'Review task state'},result:{robot_ready:true,revision:7}}));
 for(status of ['interrupted','outcome_unknown']){
  const job=await api.job('job');assert.equal(job.status,'failed');assert.equal(job.error,'Review task state');assert.equal(job.revision,7);
 }
});
