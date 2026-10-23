import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import ts from 'typescript';
const code=ts.transpileModule(await readFile(new URL('../src/manipulation4d/source-presentation.ts',import.meta.url),'utf8'),{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {withSourcePresentation,initialSourceVisibility}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
test('catalog source can only be attached to an unbound raw preview',()=>{
 const scene={video:{url:'auxiliary.mp4'},cloud:{url:'preview-cloud'},source_camera:{},segmented_instances:{},patch_videos:{a3:'old-composite'},wrist_videos:[],robots:[]};
 const native={source_robot_id:'g1',video_role:'preview-only',objects:[{id:'original-slope'}]};
 const output=withSourcePresentation(scene,native);
 assert.equal(output.video,null);assert.equal(output.cloud,undefined);assert.equal(output.source_camera,undefined);assert.equal(output.patch_videos,undefined);
 assert.equal(output.source_robot,native);assert.equal(scene.video.url,'auxiliary.mp4');
 for(const bound of [{source_body:{vertices_url:'human.bin'}},{source_robot:{source_robot_id:'g1',model_url:'bound.glb'}},{robots:[{id:'a3'}]},{output_modalities:['contact']},{generation_status:{status:'running'}}]){
  const saved={...scene,...bound};assert.equal(withSourcePresentation(saved,native),saved);
 }
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
 assert.equal(output,scene);
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

const {sceneSourceMetadata,recordingSceneCacheKey}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
test('old SMPL scene remains bound to 301 human frames after catalog becomes 299 G1 frames',()=>{
 const human={source_family:'smplx',source_evidence:'reconstructed',source_sha256:'human',frames:301,fps:30,vertices_url:'historical.bin'};
 const scene={id:'same',frames:301,fps:30,source_body:human,robots:[{id:'x2'}]};
 const recording={id:'same',source_body_family:'robot',source_robot_id:'g1',source_stage:'post_rl',source_motion:{source_family:'robot',source_robot_id:'g1',source_sha256:'robot',frames:299,fps:30}};
 assert.equal(withSourcePresentation(scene,recording.source_motion),scene);
 assert.equal(recordingSourcePresentation(sceneSourceMetadata(scene,recording)).label,'SMPL-X · reconstructed');
 assert.equal(sceneSourceMetadata({id:'same',robots:[]},recording).source_motion,undefined);
});
test('bound source, objects, scene transform and requested CV survive current catalog changes',()=>{
 const bound={source_family:'robot',source_robot_id:'g1',source_sha256:'original',frames:299,fps:30,model_url:'bound.glb',playback_url:'bound.json',objects:[{id:'bound-object'}],video_role:'preview-only'};
 const scene={frames:299,fps:30,source_robot:bound,source_transform:{scale:.7,anchor_xyz_m:[1,2,0],applied:false},cloud:{url:'requested-cloud'},video:{url:'requested.mp4'},robots:[{id:'x2'}]};
 const current={source_body_family:'robot',source_stage:'post_rl',source_motion:{...bound,source_sha256:'different',playback_url:'raw.json'}};
 assert.equal(withSourcePresentation(scene,current.source_motion),scene);
 assert.equal(scene.source_robot,bound);assert.equal(scene.source_transform.scale,.7);assert.equal(scene.cloud.url,'requested-cloud');
 assert.equal(sceneSourceMetadata(scene,current).source_stage,undefined);
 assert.equal(sceneSourceMetadata(scene,{...current,source_motion:{...current.source_motion,source_sha256:'original'}}).source_stage,'post_rl');
 for(const mismatch of [{frames:301},{fps:24},{source_robot_id:'other'}]){
  assert.equal(sceneSourceMetadata(scene,{...current,source_motion:{...bound,...mismatch}}).source_stage,undefined);
 }
});
test('cached badges use the current scene URL, never recording ID alone',()=>{
 const recording={id:'same',scene_url:'/assets/manipulation4d/current/scene.json'};
 assert.equal(recordingSceneCacheKey(recording,'/assets/manipulation4d/historical/scene.json'),undefined);
 assert.equal(recordingSceneCacheKey(recording,'/api/v1/media/manipulation4d/current/scene.json'),recordingSceneCacheKey(recording));
 assert.equal(recordingSceneCacheKey({id:'same'}),undefined);
});
