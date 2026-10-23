import test from 'node:test';
import assert from 'node:assert/strict';
import {sourceDisplayBasis,rawSourceToDisplay,sourceDisplayTransform,transformSourcePoint} from '../src/manipulation4d/source-transform.mjs';
import * as THREE from 'three';

test('source similarity commutes with the manipulation raw-Z-up to display basis',()=>{
 const anchor=[3,-2,.4],raw=[4,1,1.7],scale=.65;
 const transform=sourceDisplayTransform({scale,anchor_xyz_m:anchor,world_up:'z',applied:false});
 const display=[raw[0],raw[2],-raw[1]];
 const actual=transformSourcePoint(display,transform);
 const normalized=raw.map((v,i)=>anchor[i]+scale*(v-anchor[i]));
 const expected=[normalized[0],normalized[2],-normalized[1]];
 actual.forEach((v,i)=>assert.ok(Math.abs(v-expected[i])<1e-12));
 const fixed=transformSourcePoint([anchor[0],anchor[2],-anchor[1]],transform);
 fixed.forEach((v,i)=>assert.ok(Math.abs(v-[anchor[0],anchor[2],-anchor[1]][i])<1e-12));
 assert.deepEqual(raw,[4,1,1.7]);
});

for(const [name,basis] of [['manipulation',[[1,0,0],[0,0,1],[0,-1,0]]],['gripper',[[0,1,0],[0,0,1],[1,0,0]]]])test(`${name}: declared basis aligns source body/object and camera with numerical world`,()=>{
 const b=sourceDisplayBasis(basis),anchor=[3,-2,.4],scale=.65;
 const tr=sourceDisplayTransform({scale,anchor_xyz_m:anchor,world_up:'z'},b);
 const sourceRoot=new THREE.Group();sourceRoot.scale.setScalar(tr.scale);sourceRoot.position.fromArray(tr.offset);
 const rawWorld=new THREE.Group();rawWorld.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(...b[0],0,...b[1],0,...b[2],0,0,0,0,1));sourceRoot.add(rawWorld);
 const object=new THREE.Object3D();object.position.set(2,-1,.7);object.quaternion.setFromEuler(new THREE.Euler(.2,-.4,.9));rawWorld.add(object);sourceRoot.updateMatrixWorld(true);
 const local=new THREE.Vector3(.3,-.2,.1),raw=local.clone().applyQuaternion(object.quaternion).add(object.position).toArray();
 // Independent raw-world scaling followed by actual row equations for each exporter.
 const scaled=raw.map((v,i)=>anchor[i]+scale*(v-anchor[i]));
 const expected=name==='gripper'?[scaled[1],scaled[2],scaled[0]]:[scaled[0],scaled[2],-scaled[1]];
 object.localToWorld(local.clone()).toArray().forEach((v,i)=>assert.ok(Math.abs(v-expected[i])<1e-12));
 transformSourcePoint(rawSourceToDisplay(raw,b),tr).forEach((v,i)=>assert.ok(Math.abs(v-expected[i])<1e-12));
 const n=new THREE.Vector3(0,0,1).applyQuaternion(object.quaternion).toArray(),normal=new THREE.Vector3(0,0,1).transformDirection(object.matrixWorld).toArray();
 normal.forEach((v,i)=>assert.ok(Math.abs(v-rawSourceToDisplay(n,b)[i])<1e-12));
});

test('invalid display basis is rejected instead of silently rotating the source incorrectly',()=>{
 for(const basis of [[[1,0,0],[0,1,0],[0,0,1]],[[1,0,0],[0,0,1],[0,1,0]],[[2,0,0],[0,0,2],[0,-2,0]],[[NaN,0,0],[0,0,1],[0,-1,0]]])assert.throws(()=>sourceDisplayBasis(basis),/Invalid/);
});

test('already transformed source is never scaled twice and invalid transforms fail explicitly',()=>{
 const p=[1,2,3];
 for(const value of [undefined,{scale:.5,anchor_xyz_m:[0,0,0],world_up:'z',applied:true}])assert.deepEqual(transformSourcePoint(p,sourceDisplayTransform(value)),p);
 for(const scale of [0,-1,NaN,Infinity])assert.throws(()=>sourceDisplayTransform({scale,anchor_xyz_m:[0,0,0],world_up:'z'}),/Invalid/);
 assert.throws(()=>sourceDisplayTransform({scale:.5,anchor_xyz_m:[0,0,0],world_up:'y'}),/Invalid/);
});

test('original object-local axes, camera position and body share one similarity; robot stays outside it',()=>{
 const scale=.7,anchor=[.4,-.8,.1],tr=sourceDisplayTransform({scale,anchor_xyz_m:anchor,world_up:'z'});
 const sourceRoot=new THREE.Group();sourceRoot.scale.setScalar(tr.scale);sourceRoot.position.fromArray(tr.offset);
 const rawWorld=new THREE.Group();rawWorld.rotation.x=-Math.PI/2;sourceRoot.add(rawWorld);
 const object=new THREE.Object3D();object.position.set(2,-1,.7);object.quaternion.setFromEuler(new THREE.Euler(.2,-.4,.9));rawWorld.add(object);sourceRoot.updateMatrixWorld(true);
 const local=new THREE.Vector3(.3,-.2,.1),raw=local.clone().applyQuaternion(object.quaternion).add(object.position);
 const expected=transformSourcePoint([raw.x,raw.z,-raw.y],tr);
 const actual=object.localToWorld(local.clone()).toArray();actual.forEach((v,i)=>assert.ok(Math.abs(v-expected[i])<1e-12));
 const robot=new THREE.Object3D();robot.position.fromArray(expected);robot.updateMatrixWorld(true);
 assert.deepEqual(robot.getWorldPosition(new THREE.Vector3()).toArray(),expected);
 // Directions remain rotations: object normal is not translated or scaled.
 const normal=new THREE.Vector3(0,0,1).transformDirection(object.matrixWorld);
 const rawNormal=new THREE.Vector3(0,0,1).applyQuaternion(object.quaternion);
 assert.ok(normal.distanceTo(new THREE.Vector3(rawNormal.x,rawNormal.z,-rawNormal.y))<1e-12);
});
