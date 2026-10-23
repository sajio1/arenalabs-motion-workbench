import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import ts from 'typescript';

const source=await readFile(new URL('../src/manipulation4d/library-recordings.ts',import.meta.url),'utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {uniqueRecordings}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));

test('distinct HiPhi recordings remain available when title and clock match',()=>{
 const common={dataset:'HiPhi',title:'Keep both hands on the box and give it clear forward-and-back correcting nudges to square its position, repeatedly.',duration:60.8,video:null};
 const clips=[{...common,id:'hiphi-adjusting-adjust-0093'},{...common,id:'hiphi-adjusting-adjust-0098'}];
 assert.deepEqual(uniqueRecordings(clips),clips);
});

test('explicit source catalog alias merges with its source and keeps a ready scene',()=>{
 const source={id:'record_fc85d256536ad128bc',title:'add_remove_lid/1.hdf5'};
 const alias={id:'egodex_add_remove_lid_1',source_catalog_id:source.id,title:source.title,scene_url:'/assets/example/scene.json'};
 assert.deepEqual(uniqueRecordings([source,alias]),[alias]);
 assert.deepEqual(uniqueRecordings([alias,source]),[alias]);
});

test('a missing alias target remains visible and repeated exact IDs are deduplicated',()=>{
 const missing={id:'alias',source_catalog_id:'not-in-this-page',title:'Same title'};
 const first={id:'other',title:'Same title'};
 const ready={...first,scene_url:'/assets/other/scene.json'};
 assert.deepEqual(uniqueRecordings([missing,first,ready]),[missing,ready]);
 assert.deepEqual(uniqueRecordings([missing,ready,first]),[missing,ready]);
});
