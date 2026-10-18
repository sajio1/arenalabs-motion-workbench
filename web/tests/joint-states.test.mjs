import test from 'node:test';
import assert from 'node:assert/strict';
import {jointStateExport,jointStateCSV,currentJointState} from '../src/manipulation4d/joint-states.mjs';
const motion={fps:30,frames:2,joints:{names:['left_hip','finger,"tip"'],unit:'rad',values:[[0,-.234567891],[1.5,.9]]}};
test('whole-clip export preserves native order, precision and relative timestamps',()=>{
 const before=JSON.stringify(motion),result=jointStateExport(motion,{robot_id:'test'});
 assert.deepEqual(result.name,motion.joints.names);assert.deepEqual(result.position,motion.joints.values);
 assert.deepEqual(result.time_seconds,[0,1/30]);assert.deepEqual(result.unavailable_signals,['velocity','effort']);assert.equal(JSON.stringify(motion),before);
 assert.equal(jointStateCSV(motion),'time_seconds,"left_hip","finger,""tip"""\r\n0,0,-0.234567891\r\n0.03333333333333333,1.5,0.9\r\n');
});
test('current frame reads both joints and leaves unavailable signals empty',()=>{
 assert.deepEqual(currentJointState(motion,1),{time_seconds:1/30,name:motion.joints.names,position:[1.5,.9],velocity:[],effort:[]});
});
test('incomplete and nonfinite state arrays cannot be downloaded as valid states',()=>{
 assert.throws(()=>jointStateExport({...motion,joints:{...motion.joints,values:[[0],[1]]}}));
 assert.throws(()=>jointStateExport({...motion,joints:{...motion.joints,values:[[0,NaN],[1,2]]}}));
});
test('mobile robot export distinguishes slide meters from hinge radians',()=>{
 const mixed={...motion,joints:{...motion.joints,unit:'mixed',units:['m','rad']}};
 assert.deepEqual(jointStateExport(mixed).position_units,['m','rad']);
 assert.throws(()=>jointStateExport({...mixed,joints:{...mixed.joints,units:['rad']}}));
});
