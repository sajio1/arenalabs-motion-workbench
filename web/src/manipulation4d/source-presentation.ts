import type {SceneAsset,RecordingSourceMetadata} from './types';

/** Describe the selected input, never infer it from a dataset or action title. */
export function recordingSourcePresentation(source:RecordingSourceMetadata){
 const metadata=source.metadata??{};
 const field=(key:keyof RecordingSourceMetadata)=>{
  const value=source[key]??metadata[key];return typeof value==='string'?value.trim():'';
 };
 const declaredFamily=(field('source_body_family')||field('source_family')).toLowerCase();
 const preview=source.source_motion;
 const native=(!declaredFamily||declaredFamily==='robot')&&(!field('source_robot_id')||field('source_robot_id')===preview?.source_robot_id)?preview:undefined;
 const family=declaredFamily||native?.source_family||'';
 const evidence=(field('source_evidence')||native?.source_evidence||'').toLowerCase();
 const stage=(field('source_stage')||native?.source_stage||'').toLowerCase();
 const robot=field('source_robot_id')||native?.source_robot_id;
 const postRL=['post_rl','post-rl','post-rl-simulation','post_rl_simulation'].includes(stage);
 const provenance=postRL?'post-RL simulation':({reconstructed:'reconstructed',recorded:'recorded',mocap:'motion capture',simulation_fk:'simulation FK',simulation:'simulation',fitted:'fitted'} as Record<string,string>)[evidence];
 const format=field('source_format');
 const body=({smpl:'SMPL',smplh:'SMPL-H',smplx:'SMPL-X','smpl-h':'SMPL-H','smpl-x':'SMPL-X'} as Record<string,string>)[family];
 if(format==='recorded_skeleton_surface'){
  const label=['Recorded skeleton',provenance].filter(Boolean).join(' · ');
  return {label,detail:'Input: '+label+'. Recorded joint positions'+(body?' with '+body+' joint semantics':'')+' and a derived surface.'};
 }
 const observed={label:'Observed head + wrists',detail:'Input: observed head and wrists. Full-body motion is reconstructed before retargeting.'};
 if(format==='observed_head_wrists'||evidence==='observed-head-wrists')return observed;
 if(family==='robot'||(!family&&robot)){
  const label=[robot?.toUpperCase()||'Robot',provenance||'joint-state motion'].join(' · ');
  return {label,detail:'Input: '+label+'. Supplied robot motion; no human body reconstruction.'};
 }
 if(body){
  const label=[body,provenance].filter(Boolean).join(' · ');
  return {label,detail:'Input: '+label+'. Supplied human motion; decoded for retargeting.'};
 }
 const policy=field('motion_source_policy')||field('source_policy');
 if(['three-point-egoallo','head-wrists-v1','observed-head-wrists'].includes(policy)||field('body_adapter')==='head-wrists-v1'||format==='head-wrists-v1'){
  return observed;
 }
 return {label:'Source type unspecified',detail:'The recording does not declare its motion source type.'};
}

/** Recorded robot state is authoritative; legacy human previews are not source RGB. */
export function withSourcePresentation(scene:SceneAsset,source=scene.source_robot):SceneAsset{
 if(!source)return scene;
 const result={...scene,source_robot:source,source_body:undefined};
 if(source.video_role==='preview-only'){
  // Old reconstruction previews can contain a human and a different scene. Keep
  // target robot results, but only render the supplied native source geometry.
  Object.assign(result,{video:null,cloud:undefined,source_camera:undefined,segmented_instances:undefined,
   patch_videos:undefined,patch_kinds:undefined,wrist_videos:[]});
 }
 return result;
}

export function initialSourceVisibility(nativeRobot:boolean,previousNativeRobot:boolean,changedRecording:boolean,checked:boolean){
 if(nativeRobot)return changedRecording||checked;
 return previousNativeRobot?false:checked;
}
