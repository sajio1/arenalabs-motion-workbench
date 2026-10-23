import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import ts from 'typescript';
import * as THREE from 'three';
const source=await readFile(new URL('../src/manipulation4d/source-robot-layer.ts',import.meta.url),'utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText
 .replace(/from ['"]three['"]/,`from ${JSON.stringify(new URL('../node_modules/three/build/three.module.js',import.meta.url).href)}`);
const {SourceRobotLayer}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
const metadata={source_family:'robot',source_robot_id:'g1',world_up:'z',model_url:'/g1.glb',playback_url:'/g1.json',frames:2,fps:30};
const motion={fps:30,frames:2,positions:[[[1,2,3]],[[4,5,6]]],quaternions:[[[0,0,0,1]],[[0,0,0,1]]]};
function fixture(t,delay){
 const parent=new THREE.Group(),scene=new THREE.Group(),node=new THREE.Mesh(new THREE.BoxGeometry(),new THREE.MeshStandardMaterial());node.name='geom_0';scene.add(node);
 let loads=0,requests=0;const old=globalThis.fetch;globalThis.fetch=async()=>{requests++;return Response.json(motion);};t.after(()=>globalThis.fetch=old);
 const loader={async loadAsync(){loads++;if(delay)await delay();return {scene};}};
 const layer=new SourceRobotLayer(parent,metadata,[[1,0,0],[0,0,1],[0,-1,0]],loader);t.after(()=>layer.dispose());
 return {parent,node,layer,counts:()=>[requests,loads]};
}
test('original source robot uses shared clock and similarity without target robot state',async t=>{
 const {parent,node,layer,counts}=fixture(t);parent.scale.setScalar(.5);parent.position.set(1,0,0);
 await layer.setVisible(false,0);assert.deepEqual(counts(),[0,0]);
 await layer.setVisible(true,1/30);parent.updateMatrixWorld(true);
 assert.ok(node.getWorldPosition(new THREE.Vector3()).distanceTo(new THREE.Vector3(3,3,-2.5))<1e-12);
 assert.equal(node.material.transparent,false);assert.equal(node.material.opacity,1);await layer.setVisible(false,0);assert.equal(layer.mesh.visible,false);
 await layer.setVisible(true,0);assert.deepEqual(node.position.toArray(),[1,2,3]);assert.deepEqual(counts(),[1,1]);
});
test('scene change releases late source robot geometry',async t=>{
 let finish;const pending=new Promise(resolve=>finish=resolve);const {parent,node,layer,counts}=fixture(t,()=>pending);let disposed=false;node.geometry.addEventListener('dispose',()=>disposed=true);
 const loading=layer.setVisible(true,0);while(counts()[1]===0)await new Promise(resolve=>setImmediate(resolve));
 layer.dispose();finish();await loading;assert.equal(parent.children.length,0);assert.equal(disposed,true);
});
test('a source robot with the wrong clock fails instead of showing unrelated motion',async t=>{
 const {layer,counts}=fixture(t);globalThis.fetch=async()=>Response.json({...motion,frames:3});
 await assert.rejects(layer.setVisible(true,0),/clock or geometry mismatch/);assert.equal(counts()[1],0);
});
test('source STL geometry without exported normals receives lighting normals',async t=>{
 const {node,layer}=fixture(t);node.geometry.deleteAttribute('normal');
 await layer.setVisible(true,0);const normal=node.geometry.getAttribute('normal');
 assert.ok(normal);assert.ok([...normal.array].every(Number.isFinite));assert.equal(node.material.vertexColors,false);
});
