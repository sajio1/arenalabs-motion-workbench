import test from 'node:test';
import assert from 'node:assert/strict';
import {DepthStream} from '../src/manipulation4d/depth-stream.mjs';

const tick=()=>new Promise(resolve=>setImmediate(resolve));
function deferredLoader(){
 const calls=[];
 return {calls,load:(frame,signal)=>new Promise((resolve,reject)=>calls.push({frame,signal,resolve,reject}))};
}

test('far seek prioritizes the new window and discards late aborted responses',async()=>{
 const loader=deferredLoader(),stream=new DepthStream(200,loader.load,{concurrency:2,lookahead:2});
 stream.want(0);await tick();assert.deepEqual(loader.calls.map(c=>c.frame),[0,1]);
 stream.want(100);assert.ok(loader.calls.every(c=>c.signal.aborted));
 for(const call of loader.calls)call.resolve({sourceFrame:call.frame});
 await tick();assert.deepEqual(loader.calls.slice(2).map(c=>c.frame),[100,101]);
 assert.equal(stream.frames.size,0);assert.ok(stream.requests.size<=2);
 loader.calls[2].resolve({sourceFrame:100});await tick();
 assert.equal(stream.want(100).sourceFrame,100);assert.ok(!stream.frames.has(0));
 stream.dispose();
});

test('repeated scrubbing deduplicates requests and keeps the current frame in a bounded cache',async()=>{
 const loader=deferredLoader(),stream=new DepthStream(20,loader.load,{concurrency:3,capacity:3,lookahead:2});
 for(let i=0;i<20;i++)stream.want(0);
 await tick();assert.equal(loader.calls.length,3);
 for(const call of loader.calls)call.resolve({sourceFrame:call.frame});await tick();
 stream.want(3);await tick();
 for(const call of loader.calls.slice(3))call.resolve({sourceFrame:call.frame});await tick();
 assert.equal(stream.frames.size,3);assert.equal(stream.want(3).sourceFrame,3);
 assert.deepEqual([...stream.frames.keys()].sort((a,b)=>a-b),[3,4,5]);stream.dispose();
});

test('a failed frame waits for explicit retry instead of issuing an animation-frame retry storm',async()=>{
 let calls=0,fail=true;
 const stream=new DepthStream(1,async()=>{calls++;if(fail)throw new Error('network');return {sourceFrame:0};});
 assert.equal(await stream.initial(0),undefined);assert.equal(stream.errors.get(0).message,'network');
 for(let i=0;i<20;i++)stream.want(0);await tick();assert.equal(calls,1);
 fail=false;stream.retry();await tick();assert.equal(stream.want(0).sourceFrame,0);assert.equal(calls,2);stream.dispose();
});

test('changing recordings cancels in-flight work and cannot refill the disposed cache',async()=>{
 const loader=deferredLoader(),stream=new DepthStream(100,loader.load);
 stream.want(0);await tick();stream.dispose();
 for(const call of loader.calls){assert.ok(call.signal.aborted);call.resolve({sourceFrame:call.frame});}
 await tick();assert.equal(stream.frames.size,0);assert.equal(stream.requests.size,0);assert.equal(loader.calls.length,6);
});

test('playback fills a runway before starting and after starvation; paused seeks stay immediate',async()=>{
 const loader=deferredLoader(),stream=new DepthStream(20,loader.load,{concurrency:6,lookahead:5,resumeFrames:3});
 assert.equal(stream.ready(0),false);await tick();
 loader.calls[0].resolve({sourceFrame:0});await tick();
 assert.equal(stream.ready(0,false),true);assert.equal(stream.ready(0),false);
 loader.calls[1].resolve({sourceFrame:1});loader.calls[2].resolve({sourceFrame:2});await tick();
 assert.equal(stream.ready(0),true);assert.equal(stream.ready(1),true);assert.equal(stream.ready(2),true);
 assert.equal(stream.ready(3),false);loader.calls.find(c=>c.frame===3).resolve({sourceFrame:3});await tick();
 assert.equal(stream.ready(3),false,'one late frame must not resume and immediately stall again');
 loader.calls.find(c=>c.frame===4).resolve({sourceFrame:4});loader.calls.find(c=>c.frame===5).resolve({sourceFrame:5});await tick();
 assert.equal(stream.ready(3),true);stream.dispose();
});

test('dense clouds shrink the prefetch horizon to the byte budget without refetch loops',async()=>{
 const counts=new Map();
 const stream=new DepthStream(100,async i=>{counts.set(i,(counts.get(i)||0)+1);return {xyz:new Float32Array(10),sourceFrame:i};},{concurrency:2,maxBytes:120,lookahead:20,resumeFrames:18});
 stream.ready(0);for(let i=0;i<12;i++)await tick();
 assert.ok(stream.bytes<=120);assert.equal(stream.horizon(),2);assert.equal(stream.ready(0),true);
 assert.ok([...counts.values()].every(n=>n===1));assert.ok(counts.size<=4);stream.dispose();
});

test('a short tail can play without waiting for a full initial runway',async()=>{
 const stream=new DepthStream(2,async i=>({sourceFrame:i}));
 stream.want(0);await tick();assert.equal(stream.ready(0),true);assert.equal(stream.ready(1),true);stream.dispose();
});

test('the tail prefetches the next loop before playback wraps to zero',async()=>{
 const stream=new DepthStream(100,async i=>({sourceFrame:i}),{lookahead:5,resumeFrames:3});
 stream.want(98);await tick();
 assert.ok([98,99,0,1,2,3].every(i=>stream.frames.has(i)));
 assert.equal(stream.ready(98),true);assert.equal(stream.ready(0),true);stream.dispose();
});
