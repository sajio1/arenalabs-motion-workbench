import test from 'node:test';
import assert from 'node:assert/strict';
import {sampleFrame,activeRobot,visibleClips,safeText,robotChoices,motionGroup} from '../src/state.js';
test('one seconds clock clamps each clip without stretching or independent looping',()=>{
  assert.equal(sampleFrame(2,30,31),30);assert.equal(sampleFrame(2,30,91),60);assert.equal(sampleFrame(-1,30,31),0);assert.equal(sampleFrame(1/30,30,31),1);
});
test('package duplicates prefer usable model; equal labels do not merge distinct models',()=>{
 const robots=[{id:'old',label:'A3',report:{selection_key:'one'},conversion_ready:false},{id:'new',label:'A3',report:{selection_key:'one'},conversion_ready:true},{id:'different',label:'A3',report:{selection_key:'two'},conversion_ready:true},{id:'unknown',label:'A3'}];
 assert.deepEqual(robotChoices(robots).map(r=>r.id),['new','different','unknown']);
});
test('categories use dataset activity metadata without changing filenames',()=>{
 const clip={title:'xdata.npz [00:00–00:15]',family:'record_id',metadata:{'Activity category':'Clothing & textiles','Activity subclass':'Laundry session excerpts'}};
 assert.deepEqual(motionGroup(clip),['Clothing & textiles','Laundry session excerpts']);
 assert.equal(visibleClips([clip],'laundry').length,1);
 assert.equal(clip.title,'xdata.npz [00:00–00:15]');
 assert.deepEqual(motionGroup({page:'manipulation',family:'ego_007_screw_unscrew_bottle_cap'}),['Assembly & tools','Screw Unscrew Bottle Cap']);
});
test('global robot selection and selected slot remain independent',()=>{
  const slots=[{robot:{id:'g1'}},{robot:{id:'h2'}}],defaultRobot={id:'skye'};
  assert.equal(activeRobot(defaultRobot,null,slots).id,'skye');assert.equal(activeRobot(defaultRobot,1,slots).id,'h2');assert.equal(slots[0].robot.id,'g1');
});
test('family and filenames searchable; metadata cannot inject markup',()=>{
  const clips=[{id:'walk',title:'行走',family:'Walking',description:'SOMA',source_name:'walk.bvh'}];
  assert.equal(visibleClips(clips,'walking').length,1);assert.equal(visibleClips(clips,'跳跃').length,0);
  assert.equal(safeText('<img src=x onerror="a">'),'&lt;img src=x onerror=&quot;a&quot;&gt;');
});
