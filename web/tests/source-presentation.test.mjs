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
 const scene={video:{url:'recorded.mp4'},cloud:{url:'measured'},source_body:{vertices_url:'smplh.bin'}};
 assert.equal(withSourcePresentation(scene),scene);
 const output=withSourcePresentation(scene,{source_robot_id:'robot',video_role:'recorded'});
 assert.equal(output.video,scene.video);assert.equal(output.cloud,scene.cloud);
});
test('native preview defaults on, user visibility persists on reload, human remains opt-in',()=>{
 assert.equal(initialSourceVisibility(true,false,true,false),true);
 assert.equal(initialSourceVisibility(true,true,false,false),false);
 assert.equal(initialSourceVisibility(false,true,true,true),false);
 assert.equal(initialSourceVisibility(false,false,true,false),false);
 assert.equal(initialSourceVisibility(false,false,false,true),true);
});
