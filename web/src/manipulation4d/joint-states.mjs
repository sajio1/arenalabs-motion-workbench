/** Keep exported native coordinates in their original order and precision. */
export function validateJointStates(motion) {
 const {names,values,velocity,effort,units}=motion.joints;
 if(units&&(units.length!==names.length||units.some(u=>!['rad','m'].includes(u))))throw new Error('Invalid per-joint units');
 if(!Number.isFinite(motion.fps)||motion.fps<=0||values.length!==motion.frames)throw new Error('Invalid joint-state timebase');
 if(!names.length||new Set(names).size!==names.length)throw new Error('Joint names must be nonempty and unique');
 for(const signal of [values,velocity,effort].filter(Boolean)){
  if(signal.length!==motion.frames||signal.some(row=>row.length!==names.length||row.some(v=>!Number.isFinite(v))))throw new Error('Joint-state dimensions or values are invalid');
 }
}
export function currentJointState(motion,frame) {
 const index=Math.max(0,Math.min(motion.frames-1,Math.floor(frame)));
 return {time_seconds:index/motion.fps,name:motion.joints.names,position:motion.joints.values[index],...(motion.joints.units?{position_units:motion.joints.units}:{}),velocity:motion.joints.velocity?.[index]??[],effort:motion.joints.effort?.[index]??[]};
}
export function jointStateExport(motion,context={}) {
 validateJointStates(motion);
 const {names,values,unit,units,velocity,effort}=motion.joints;
 return {schema:'arenalab.joint_states.v1',...context,provenance:'Retargeted kinematic output',time_basis:'clip_relative',fps:motion.fps,frames:motion.frames,position_unit:unit,...(motion.base?{base:motion.base}:{}),...(units?{position_units:units}:{}),name:names,time_seconds:values.map((_,i)=>i/motion.fps),position:values,...(velocity?{velocity}:{}),...(effort?{effort}:{}),unavailable_signals:[...(!velocity?['velocity']:[]),...(!effort?['effort']:[])]};
}
export function jointStateCSV(motion) {
 validateJointStates(motion);
 const quote=value=>'"'+String(value).replaceAll('"','""')+'"';
 return ['time_seconds,'+motion.joints.names.map((name,i)=>quote(motion.joints.units?name+' ['+motion.joints.units[i]+']':name)).join(','),...motion.joints.values.map((row,i)=>[i/motion.fps,...row].join(','))].join('\r\n')+'\r\n';
}
