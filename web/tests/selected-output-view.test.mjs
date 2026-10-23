import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import ts from 'typescript';

const source=await readFile(new URL('../src/manipulation4d/selected-output-view.ts',import.meta.url),'utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {bindPatchVideo,drawContactPanel}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));

function node(){return {hidden:false,textContent:'',value:0,attributes:{},classList:{toggle(){}},setAttribute(k,v){this.attributes[k]=v;}};}
function video(){return {...node(),src:'old.mp4',pause(){this.paused=true;},removeAttribute(k){delete this[k];},load(){this.loaded=true;}};}

test('FOV-only public scene binds video with no robot trajectory and never needs its model/playback',()=>{
 const scene={robots:[],output_modalities:['inpaint_fov'],generation_status:{robot_id:'g1'},patch_videos:{g1:'/selected/composite.mp4'},patch_kinds:{g1:'estimated_camera_composite'}};
 const v=video(),empty=node(),label=node();
 assert.equal(bindPatchVideo(scene,'',new Set(['g1']),v,empty,label),'g1');
 assert.equal(v.src,'/selected/composite.mp4');assert.equal(v.hidden,false);assert.equal(empty.hidden,true);
 assert.equal(label.textContent,'ESTIMATED CAMERA · SYNTHETIC');assert.deepEqual(scene.robots,[]);
});

test('point-cloud/contact-only snapshot clears stale FOV and ignores unpublished/unknown robot patches',()=>{
 const v=video(),empty=node(),label=node();
 assert.equal(bindPatchVideo({robots:[],patch_videos:{unregistered:'/hidden.mp4'}},'g1',new Set(['g1']),v,empty,label),undefined);
 assert.equal(v.hidden,true);assert.equal(v.src,undefined);assert.equal(v.paused,true);assert.equal(empty.hidden,false);
 assert.equal(label.textContent,'NOT GENERATED');
});

test('contact-only panel advances source-clock samples without robot playback or geometry',()=>{
 const nodes=new Map();const get=k=>{if(!nodes.has(k))nodes.set(k,node());return nodes.get(k);};
 const document={getElementById:get,querySelector:get};get('contact-mode').value='video';
 const series={fps:2,likelihood:[[[.1,.2,.3,.4,.5],[.5,.4,.3,.2,.1]],[[null,.7,.8,.9,1],[1,.9,.8,.7,null]]],hand_contact:[[{state:'open',object:''},{state:'open',object:''}],[{state:'touch',object:'cup'},{state:'unknown',object:''}]]};
 drawContactPanel(series,undefined,0,document);
 assert.equal(get('[data-digit-score="0-0"]').textContent,'0.10');
 drawContactPanel(series,undefined,.75,document);
 assert.equal(get('[data-digit-score="0-0"]').textContent,'—');assert.equal(get('[data-digit-score="0-1"]').textContent,'0.70');
 assert.equal(get('[data-contact="0-0"]').attributes['aria-label'],'Unknown / not visible');
 assert.equal(get('contact-hand-0').textContent,'LEFT · video: touch / cup');
 assert.match(get('tactile-note').textContent,/No pressure or force/);
});
