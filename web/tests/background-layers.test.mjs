import test from 'node:test';
import assert from 'node:assert/strict';
import {backgroundLayers} from '../src/manipulation4d/background-layers.mjs';

test('background has one owner, including toggles and unavailable fusion',()=>{
 for(const cloud of [false,true])for(const memory of [false,true])for(const loaded of [false,true]){
  const result=backgroundLayers(cloud,memory,loaded);
  assert.equal(Number(result.fused)+Number(result.frameBackground),cloud?1:0);
 }
 assert.deepEqual(backgroundLayers(true,true,true),{fused:true,frameBackground:false});
 assert.deepEqual(backgroundLayers(true,false,true),{fused:false,frameBackground:true});
 assert.deepEqual(backgroundLayers(true,true,false),{fused:false,frameBackground:true});
});
