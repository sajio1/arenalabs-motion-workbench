import test from 'node:test';
import assert from 'node:assert/strict';
import {observationDepth} from '../src/manipulation4d/observation-depth.mjs';
const identity=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const k=[[10,0,5],[0,10,5],[0,0,1]];
test('foreground occludes historical background regardless of point order',()=>{
 for(const order of [false,true]){
  const points=order?[0,0,1,0,0,2]:[0,0,2,0,0,1],labels=order?[1,0]:[0,1];
  const r=observationDepth(new Float32Array(points),new Uint8Array(labels),identity,k,10,10,10);
  const p=(5*10+5)*4;assert.equal(r.data[p],1);assert.equal(r.data[p+1],0);
 }
});
test('out-of-FOV, behind-camera and invalid points never activate memory',()=>{
 const r=observationDepth(new Float32Array([20,0,1,0,0,-1,NaN,0,1]),new Uint8Array([0,0,0]),identity,k,10,10,10);
 assert.ok(r.data.every(v=>v===0));
});
test('visibility follows the original source camera rather than viewer orbit',()=>{
 const translated=identity.slice();translated[12]=-1;
 const r=observationDepth(new Float32Array([1,0,2]),new Uint8Array([0]),translated,k,10,10,10);
 const p=(5*10+5)*4;assert.equal(r.data[p],2);assert.equal(r.data[p+1],1);
});
