import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import ts from 'typescript';
import * as THREE from 'three';

const source=await readFile(new URL('../src/manipulation4d/source-body-layer.ts',import.meta.url),'utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText
 .replace(/from ['"]three['"]/,`from ${JSON.stringify(new URL('../node_modules/three/build/three.module.js',import.meta.url).href)}`);
const {SourceBodyLayer,sourceMotionDisplay}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
const body={vertices_url:'/vertices',faces_url:'/faces',frames:2,vertex_count:3,face_count:1,world_up:'z',fps:2};
const basis=[[1,0,0],[0,0,1],[0,-1,0]];
const vertices=new Float32Array([0,0,0,1,0,0,0,0,1,2,0,0,3,0,0,2,0,1]);

test('source robot provenance never presents a robot mesh as SMPL',()=>{
 assert.equal(sourceMotionDisplay({...body,source_family:'robot',source_robot_id:'g1'}).label,'Source robot · G1');
 assert.equal(sourceMotionDisplay({...body,source_family:'robot'}).label,'Source robot');
 for(const family of ['smpl','smplh','smplx'])assert.equal(sourceMotionDisplay({...body,source_family:family}).label,'SMPL motion');
 assert.equal(sourceMotionDisplay(body).label,'Source motion');
});

test('recorded skeleton proxy surface is explicitly distinguished from fitted SMPL',()=>{
 const display=sourceMotionDisplay({...body,source_family:'smplx',source_format:'recorded_skeleton_surface'});
 assert.equal(display.label,'Recorded skeleton · proxy surface');
 assert.match(display.title,/approximate skinned surface/);
 assert.match(display.title,/not original subject SMPL geometry/);
});

function fixture(t,delay){
 const previous=globalThis.fetch,calls=[];
 globalThis.fetch=async(url,options)=>{calls.push(url);if(delay)await delay();return new Response(url==='/vertices'?vertices.buffer.slice(0):new Uint32Array([0,1,2]).buffer);};
 t.after(()=>globalThis.fetch=previous);
 const parent=new THREE.Group(),layer=new SourceBodyLayer(parent,body,30,basis);
 t.after(()=>layer.dispose());return {parent,layer,calls};
}

test('SMPL is opt-in, uses its own clock and preserves the shared source similarity',async t=>{
 const {parent,layer,calls}=fixture(t);parent.scale.setScalar(.5);parent.position.set(1,0,2);
 layer.pose(.7);await layer.setVisible(false,.7);assert.equal(calls.length,0);assert.equal(parent.children.length,0);
 await layer.setVisible(true,.7);assert.deepEqual(calls,['/vertices','/faces']);assert.equal(layer.mesh.visible,true);
 const positions=layer.mesh.geometry.getAttribute('position');assert.deepEqual(Array.from(positions.array),[2,0,0,3,0,0,2,1,0]);
 parent.updateMatrixWorld(true);assert.deepEqual(layer.mesh.localToWorld(new THREE.Vector3(2,0,0)).toArray(),[2,0,2]);
 await layer.setVisible(false,0);assert.equal(layer.mesh.visible,false);
 await layer.setVisible(true,0);assert.equal(calls.length,2);assert.equal(positions.array[0],0);
 assert.deepEqual(Array.from(vertices.slice(0,3)),[0,0,0]);
});

test('unchecking while loading never makes the body visible on completion',async t=>{
 let finish;const pending=new Promise(resolve=>finish=resolve);
 const {layer,calls}=fixture(t,()=>pending);const first=layer.setVisible(true,0);
 await layer.setVisible(false,.7);finish();await first;
 assert.equal(layer.mesh.visible,false);await layer.setVisible(true,.7);
 assert.equal(calls.length,2);assert.equal(layer.mesh.geometry.getAttribute('position').array[0],2);
});

test('switching scenes disposes a pending mesh without attaching stale geometry',async t=>{
 let finish;const pending=new Promise(resolve=>finish=resolve);
 const {parent,layer}=fixture(t,()=>pending);const loading=layer.setVisible(true,0);
 layer.dispose();finish();await loading;assert.equal(parent.children.length,0);assert.equal(layer.mesh,undefined);
});

test('missing geometry reports an error and can retry the existing cache',async t=>{
 const {layer,calls}=fixture(t);const valid=globalThis.fetch;
 globalThis.fetch=async()=>new Response('',{status:404});
 await assert.rejects(layer.setVisible(true,0),/cache unavailable/);assert.equal(layer.mesh,undefined);
 globalThis.fetch=valid;await layer.setVisible(true,0);assert.equal(layer.mesh.visible,true);assert.equal(calls.length,2);
});
