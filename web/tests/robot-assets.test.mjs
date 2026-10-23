import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import ts from 'typescript';
import * as THREE from 'three';
const text=await readFile(new URL('../src/manipulation4d/robot-assets.ts',import.meta.url),'utf8');
const code=ts.transpileModule(text,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText
 .replace(/from ['"]three['"]/,`from ${JSON.stringify(new URL('../node_modules/three/build/three.module.js',import.meta.url).href)}`);
const {RobotAssets}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
function fixture(t,capacity=2){
 const old=globalThis.fetch;let fetches=0,loads=0;const models=[];
 globalThis.fetch=async()=>{fetches++;return Response.json({frames:2});};t.after(()=>globalThis.fetch=old);
 const loader={async loadAsync(){loads++;const scene=new THREE.Group();scene.add(new THREE.Mesh(new THREE.BoxGeometry(),new THREE.MeshStandardMaterial()));models.push(scene);return {scene};}};
 return {cache:new RobotAssets(loader,capacity),models,counts:()=>({fetches,loads})};
}
const row=(key='a')=>({model_url:`/${key}.glb`,playback_url:`/${key}.json`});
test('prefetch and simultaneous scenes share one parse; transforms are independent',async t=>{
 const {cache,counts}=fixture(t);cache.prefetch(row());
 const [a,b]=await Promise.all([cache.acquire(row()),cache.acquire(row())]);
 assert.deepEqual(counts(),{fetches:1,loads:1});assert.notEqual(a.node,b.node);
 assert.equal(a.node.children[0].geometry,b.node.children[0].geometry);
 a.node.children[0].position.x=8;assert.equal(b.node.children[0].position.x,0);a.release();b.release();
 const c=await cache.acquire(row());assert.deepEqual(counts(),{fetches:1,loads:1});c.release();
});
test('LRU never disposes a displayed mesh and eventually releases inactive resources',async t=>{
 const {cache,models,counts}=fixture(t,1);const a=await cache.acquire(row('a'));let disposed=0;
 models[0].children[0].geometry.addEventListener('dispose',()=>disposed++);
 const b=await cache.acquire(row('b'));assert.equal(disposed,0);a.release();assert.equal(disposed,1);b.release();
 const again=await cache.acquire(row('a'));assert.equal(counts().loads,3);again.release();
});
test('a failed motion request can retry without poisoning the parsed-model cache',async t=>{
 const {cache,counts}=fixture(t);globalThis.fetch=async()=>new Response('missing',{status:404});
 await assert.rejects(cache.acquire(row()),/motion unavailable/);
 globalThis.fetch=async()=>Response.json({frames:2});const result=await cache.acquire(row());
 assert.equal(counts().loads,1);result.release();
});
