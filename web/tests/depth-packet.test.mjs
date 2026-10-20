import test from 'node:test';
import assert from 'node:assert/strict';
import {decodeDepthPacket} from '../src/manipulation4d/depth-packet.mjs';

function packet(){
 const value={shape:[2,2],intrinsic:[[2,0,0],[0,2,0],[0,0,1]],camera_to_world:[[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]],source_frame:42,depth_scale:.001};
 let header=new TextEncoder().encode(JSON.stringify(value));const length=Math.ceil(header.length/4)*4;
 const buffer=new ArrayBuffer(4+length+4*6);new DataView(buffer).setUint32(0,length,true);
 const bytes=new Uint8Array(buffer);bytes.fill(32,4,4+length);bytes.set(header,4);
 new Uint16Array(buffer,4+length,4).set([1000,0,2000,1000]);
 bytes.fill(128,4+length+8,4+length+20);bytes.set([1,2,3,4],4+length+20);return buffer;
}
test('browser restores exact calibrated points, excludes invalid depth, preserves source clock',()=>{
 const value=decodeDepthPacket(packet());
 assert.equal(value.sourceFrame,42);assert.equal(value.count,3);
 assert.deepEqual([...value.xyz],[1,2,4,1,3,5,1.5,2.5,4]);
 assert.deepEqual([...value.labels],[1,3,4]);
});
test('truncated packet cannot be presented as a valid observation',()=>{
 assert.throws(()=>decodeDepthPacket(packet().slice(0,-1)),/coverage/);
});

test('source human cloud uses the same calibrated transform and remains a separate layer',()=>{
 const value={shape:[1,1],intrinsic:[[1,0,0],[0,1,0],[0,0,1]],camera_to_world:[[.8,0,0,.1],[0,.8,0,.2],[0,0,.8,-.1],[0,0,0,1]],source_frame:4,depth_scale:.001,human_depth_bytes:2};
 const header=new TextEncoder().encode(JSON.stringify(value)),length=Math.ceil(header.length/4)*4;
 const buffer=new ArrayBuffer(4+length+8),view=new DataView(buffer),bytes=new Uint8Array(buffer);view.setUint32(0,length,true);bytes.fill(32,4,4+length);bytes.set(header,4);
 view.setUint16(4+length,0,true);bytes.fill(128,4+length+2,4+length+5);bytes[4+length+5]=0;view.setUint16(4+length+6,1000,true);
 const decoded=decodeDepthPacket(buffer);assert.equal(decoded.count,0);assert.equal(decoded.human.count,1);assert.equal(decoded.sourceFrame,4);
 for(const [i,want] of [.1,.2,.7].entries())assert.ok(Math.abs(decoded.human.xyz[i]-want)<1e-6);
 assert.throws(()=>decodeDepthPacket(buffer.slice(0,-1)),/coverage/);
});
