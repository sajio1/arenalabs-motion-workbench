import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import ts from 'typescript';

const source=await readFile(new URL('../src/manipulation4d/source-variants.ts',import.meta.url),'utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {sourceVariantChoices,selectSourceVariant,prepareSourcePreview}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
const variants=[{id:'original-human',label:'Human'},{id:'separate-g1',label:'G1',source_robot_id:'g1'},{id:'separate-h2',label:'H2',source_robot_id:'h2'}];
const human={id:'original-human',source_group_id:'original-human',source_variant_id:'human',source_variants:variants,scene_url:'/human/result.json'};
const g1={id:'separate-g1',source_group_id:'original-human',source_variant_id:'g1',source_variants:variants,scene_url:'/g1/result.json'};

test('variant controls use exact recording IDs, not labels, role suffixes or inferred dataset names',()=>{
 const choices=sourceVariantChoices({...human,source_variant_id:'g1'});
 assert.deepEqual(choices.map(v=>[v.id,v.selected,v.available]),[['original-human',true,true],['separate-g1',false,true],['separate-h2',false,true]]);
 assert.equal(sourceVariantChoices(g1)[1].selected,true);
 assert.deepEqual(sourceVariantChoices({id:'unrelated',source_variants:variants}),[]);
 assert.deepEqual(sourceVariantChoices({...human,source_variants:[variants[0]]}),[]);
 assert.deepEqual(sourceVariantChoices({...human,id:'not-in-group'}),[]);
});

test('duplicate IDs are ignored and unavailable variants keep their reason',()=>{
 const choices=sourceVariantChoices({...human,source_variants:[...variants,variants[1],{id:'',label:'invalid'},{id:'missing',label:'Other',available:false,reason:'Source not paired'}]});
 assert.equal(choices.length,4);assert.equal(choices[3].available,false);assert.equal(choices[3].reason,'Source not paired');
});

test('switch fetches and selects the target record with its own cache without changing source history',async()=>{
 const original=structuredClone(human),calls=[];
 assert.equal(await selectSourceVariant(human,'separate-g1',{
  fetchRecording:async id=>{calls.push(['get',id]);return g1;},isCurrent:()=>true,
  selectRecording:async clip=>calls.push(['select',clip.id,clip.scene_url]),
 }),true);
 assert.deepEqual(calls,[['get','separate-g1'],['select','separate-g1','/g1/result.json']]);
 assert.deepEqual(human,original);
});

test('same variant is a no-op and undeclared or unavailable variants never fetch',async()=>{
 const actions={fetchRecording:async()=>assert.fail('must not fetch'),isCurrent:()=>true,selectRecording:async()=>assert.fail('must not select')};
 assert.equal(await selectSourceVariant(human,human.id,actions),false);
 await assert.rejects(selectSourceVariant(human,'forged',actions),/not declared/);
 await assert.rejects(selectSourceVariant({...human,source_variants:[variants[0],{...variants[1],available:false,reason:'Not paired'}]},'separate-g1',actions),/Not paired/);
});

test('stale target replies cannot override a newer recording selection',async()=>{
 assert.equal(await selectSourceVariant(human,g1.id,{fetchRecording:async()=>g1,isCurrent:()=>false,selectRecording:async()=>assert.fail('stale selection')}),false);
});

test('mismatched response identity or group never substitutes an input',async()=>{
 for(const target of [{...g1,id:'other'},{...g1,source_group_id:'other'}]){
  await assert.rejects(selectSourceVariant(human,g1.id,{fetchRecording:async()=>target,isCurrent:()=>true,selectRecording:async()=>assert.fail('incorrect input')}),/different recording group/);
 }
});

test('lazy preview is opt-in and never replaces a bound result or existing raw preview',async()=>{
 const fetch=async()=>assert.fail('unexpected preview preparation');
 for(const clip of [human,{...human,scene_url:undefined},{...human,source_preview_available:true},{...human,scene_url:undefined,source_preview_available:true,source_motion:{model_url:'/original.glb'}}]){
  assert.equal(await prepareSourcePreview(clip,fetch,()=>true),clip);
 }
});

test('lazy source preview attaches only to its own immutable recording copy',async()=>{
 const clip={...g1,scene_url:undefined,source_preview_available:true},original=structuredClone(clip),calls=[];
 const motion={source_family:'robot',source_robot_id:'g1',model_url:'/g1.glb',playback_url:'/g1.json'};
 const prepared=await prepareSourcePreview(clip,async id=>{calls.push(id);return {recording_id:id,source_motion:motion,cache_hit:true};},()=>true);
 assert.deepEqual(calls,[g1.id]);assert.equal(prepared.source_motion,motion);assert.equal(prepared.id,g1.id);assert.deepEqual(clip,original);
});

test('late preview replies are discarded and identity mismatches fail explicitly',async()=>{
 const clip={...g1,scene_url:undefined,source_preview_available:true};
 assert.equal(await prepareSourcePreview(clip,async()=>({recording_id:g1.id,source_motion:{}}),()=>false),undefined);
 await assert.rejects(prepareSourcePreview(clip,async()=>({recording_id:'other',source_motion:{}}),()=>true),/did not match/);
 await assert.rejects(prepareSourcePreview(clip,async()=>{throw new Error('Preview assets unavailable');},()=>true),/Preview assets unavailable/);
});
