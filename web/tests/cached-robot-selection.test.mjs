import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import ts from 'typescript';

const source=await readFile(new URL('../src/manipulation4d/cached-robot-selection.ts',import.meta.url),'utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {cachedRobotSelection}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
const robot=(id='a3',hand='wuji')=>({id,hand,label:id,model_url:'/model.glb',playback_url:'/playback.json'});

test('fresh completed scene restores its published robot and hand with no top-level end effector',()=>{
 const row=robot();
 const scene={end_effector:null,robots:[robot('other','none'),row],generation_status:{robot_id:'a3',hand_id:'wuji'},output_modalities:['trajectory','point_cloud','contact','inpaint_fov']};
 assert.equal(cachedRobotSelection(scene),row);
});

test('legacy end-effector string or object cannot override the published hand, including explicit none',()=>{
 for(const end_effector of ['wuji',{id:'wuji'},null,undefined]){
  const row=robot('g1','none');
  assert.equal(cachedRobotSelection({robots:[row],end_effector}),row);
 }
});

test('generation metadata resolves registry aliases but never substitutes a mismatched hand',()=>{
 const row={...robot('a3-variant'),registry_robot_id:'a3'};
 assert.equal(cachedRobotSelection({robots:[row],generation_status:{robot_id:'a3',hand_id:'wuji'}}),row);
 assert.equal(cachedRobotSelection({robots:[row],generation_status:{robot_id:'a3',hand_id:'none'}}),undefined);
});

test('explicit nontrajectory snapshots never load a stale internal robot',()=>{
 for(const outputs of [[],['point_cloud'],['contact'],['inpaint_fov']]){
  assert.equal(cachedRobotSelection({robots:[robot()],output_modalities:outputs}),undefined);
  assert.equal(cachedRobotSelection({robots:[robot()],generation_status:{output_modalities:outputs}}),undefined);
 }
});

test('valid user selection is preserved and an unavailable selection is not replaced',()=>{
 const rows=[robot(),robot('x2','none')];const scene={robots:rows,generation_status:{robot_id:'a3',hand_id:'wuji'}};
 assert.equal(cachedRobotSelection(scene,'x2','none'),rows[1]);
 assert.equal(cachedRobotSelection(scene,'missing','none'),undefined);
 assert.equal(cachedRobotSelection(scene,'x2',''),undefined);
});

test('ambiguous legacy scenes and missing trajectory assets stay unselected',()=>{
 assert.equal(cachedRobotSelection({robots:[robot(),robot('x2','none')]}),undefined);
 assert.equal(cachedRobotSelection({robots:[{...robot(),playback_url:''}]}),undefined);
 assert.equal(cachedRobotSelection({robots:[]}),undefined);
});
