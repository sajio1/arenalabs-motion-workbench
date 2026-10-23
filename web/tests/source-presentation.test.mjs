import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import ts from 'typescript';
const code=ts.transpileModule(await readFile(new URL('../src/manipulation4d/source-presentation.ts',import.meta.url),'utf8'),{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {withSourcePresentation,initialSourceVisibility}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
test('native source supersedes human preview video and geometry without changing target results',()=>{
 const scene={video:{url:'human.mp4'},source_body:{vertices_url:'human.bin'},cloud:{url:'human-cloud'},source_camera:{},segmented_instances:{},patch_videos:{a3:'old-composite'},wrist_videos:[],robots:[{id:'a3'}],source_transform:{scale:.7}};
 const native={source_robot_id:'g1',video_role:'preview-only',objects:[{id:'original-slope'}]};
 const output=withSourcePresentation(scene,native);
 assert.equal(output.video,null);assert.equal(output.source_body,undefined);assert.equal(output.cloud,undefined);assert.equal(output.source_camera,undefined);assert.equal(output.patch_videos,undefined);
 assert.equal(output.source_robot,native);assert.equal(output.robots,scene.robots);assert.equal(output.source_transform,scene.source_transform);
 assert.equal(scene.video.url,'human.mp4');assert.equal(scene.source_body.vertices_url,'human.bin');
});
test('supplied SMPL and actual robot camera streams remain available',()=>{
 const scene={video:{url:'recorded.mp4'},cloud:{url:'measured'},source_body:{vertices_url:'smplh.bin'},
  source_camera:{url:'camera'},segmented_instances:{url:'instances'},patch_videos:{a3:'fov.mp4'},
  wrist_videos:[{url:'wrist.mp4'}]};
 assert.equal(withSourcePresentation(scene),scene);
 const output=withSourcePresentation(scene,{source_robot_id:'robot',video_role:'recorded'});
 for(const key of ['video','cloud','source_camera','segmented_instances','patch_videos','wrist_videos']){
  assert.equal(output[key],scene[key]);
 }
 assert.equal(output.source_body,undefined);
});
test('unspecified source robot video provenance never erases requested CV results',()=>{
 const scene={video:{url:'camera.mp4'},cloud:{url:'cloud'},patch_videos:{a3:'fov.mp4'}};
 for(const role of [undefined,'unspecified']){
  const output=withSourcePresentation(scene,{source_robot_id:'robot',video_role:role});
  assert.equal(output.video,scene.video);assert.equal(output.cloud,scene.cloud);
  assert.equal(output.patch_videos,scene.patch_videos);
 }
});
test('native preview defaults on, user visibility persists on reload, human remains opt-in',()=>{
 assert.equal(initialSourceVisibility(true,false,true,false),true);
 assert.equal(initialSourceVisibility(true,true,false,false),false);
 assert.equal(initialSourceVisibility(false,true,true,true),false);
 assert.equal(initialSourceVisibility(false,false,true,false),false);
 assert.equal(initialSourceVisibility(false,false,false,true),true);
});

const {recordingSourcePresentation}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
test('identically named GRAIL recordings describe their selected source, not the dataset',()=>{
 const shared={dataset:'GRAIL',title:'Terrain curbs — curb 073 — 000',family:'terrain'};
 assert.equal(recordingSourcePresentation({...shared,source_body_family:'smplx',source_evidence:'reconstructed'}).label,'SMPL-X · reconstructed');
 assert.equal(recordingSourcePresentation({...shared,source_body_family:'robot',source_robot_id:'g1',source_stage:'post_rl',source_evidence:'simulation_fk'}).label,'G1 · post-RL simulation');
 assert.equal(recordingSourcePresentation(shared).label,'Source type unspecified');
});
test('robot preview metadata is a fallback and does not turn all recordings into simulations',()=>{
 assert.equal(recordingSourcePresentation({source_motion:{source_family:'robot',source_robot_id:'g1'}}).label,'G1 · joint-state motion');
 assert.equal(recordingSourcePresentation({source_family:'robot',source_robot_id:'g1',source_evidence:'recorded'}).label,'G1 · recorded');
 assert.equal(recordingSourcePresentation({source_family:'robot',source_robot_id:'g1',source_stage:'post-rl-simulation'}).label,'G1 · post-RL simulation');
 assert.equal(recordingSourcePresentation({source_family:'robot',source_robot_id:'g1',source_stage:'unknown',source_evidence:'simulation_fk'}).label,'G1 · simulation FK');
});
test('explicit catalog source metadata wins over stale preview metadata',()=>{
 const source={source_body_family:'smplh',source_evidence:'recorded',source_motion:{source_family:'robot',source_robot_id:'g1',source_evidence:'simulation_fk'},metadata:{source_body_family:'smplx'}};
 assert.equal(recordingSourcePresentation(source).label,'SMPL-H · recorded');
 assert.equal(recordingSourcePresentation({...source,source_evidence:undefined}).label,'SMPL-H');
});
test('observed inputs stay observed even when downstream generation uses SMPL-H',()=>{
 const source={motion_source_policy:'three-point-egoallo',generated_human_model:'smplh',dataset:'EgoDex'};
 assert.equal(recordingSourcePresentation(source).label,'Observed head + wrists');
 assert.equal(recordingSourcePresentation({metadata:{source_format:'head-wrists-v1'}}).label,'Observed head + wrists');
 assert.equal(recordingSourcePresentation({dataset:'EgoDex',generated_human_model:'smplh'}).label,'Source type unspecified');
});
test('body source labels preserve declared family and evidence without exposing paths',()=>{
 for(const [family,label] of [['smpl','SMPL'],['smplh','SMPL-H'],['smplx','SMPL-X']]){
  assert.equal(recordingSourcePresentation({metadata:{source_body_family:family,source_evidence:'reconstructed',source:'/private/model.npz'}}).label,label+' · reconstructed');
 }
 const row={source_body_family:'smplx',source_evidence:'/private/unspecified.npz'};
 assert.equal(recordingSourcePresentation(row).label,'SMPL-X');
 assert.equal(JSON.stringify(recordingSourcePresentation(row)).includes('/private'),false);
});
test('recorded skeleton format takes precedence over SMPL-X joint semantics',()=>{
 const source={source_format:'recorded_skeleton_surface',source_body_family:'smplx',source_evidence:'recorded'};
 const result=recordingSourcePresentation(source);
 assert.equal(result.label,'Recorded skeleton · recorded');
 assert.ok(result.detail.includes('SMPL-X joint semantics'));
 assert.ok(result.detail.includes('derived surface'));
 assert.equal(recordingSourcePresentation({...source,source_format:'smpl_parameters',source_evidence:'fitted'}).label,'SMPL-X · fitted');
 assert.equal(recordingSourcePresentation({...source,source_format:'smpl_parameters',source_evidence:'reconstructed'}).label,'SMPL-X · reconstructed');
 assert.equal(recordingSourcePresentation({metadata:source}).label,result.label);
});
test('explicit observation schema, evidence and policy identify head and wrist input',()=>{
 for(const source of [
  {source_family:'observations',source_format:'observed_head_wrists',source_evidence:'observed-head-wrists'},
  {source_family:'observations',source_policy:'observed-head-wrists'},
  {source_family:'observations',source_policy:'three-point-egoallo'},
  {source_evidence:'observed-head-wrists'},
  {metadata:{source_format:'observed_head_wrists'}}
 ])assert.equal(recordingSourcePresentation(source).label,'Observed head + wrists');
 assert.equal(recordingSourcePresentation({source_family:'observations'}).label,'Source type unspecified');
});
