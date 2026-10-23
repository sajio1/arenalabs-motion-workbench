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
 const provenance=postRL?'post-RL simulation':({reconstructed:'reconstructed',recorded:'recorded',mocap:'motion capture',simulation_fk:'simulation FK',simulation:'simulation',fitted:'fitted',retargeted:'retargeted reference'} as Record<string,string>)[evidence];
 const format=field('source_format');
 const body=({smpl:'SMPL',smplh:'SMPL-H',smplx:'SMPL-X','smpl-h':'SMPL-H','smpl-x':'SMPL-X'} as Record<string,string>)[family];
 if(format==='recorded_skeleton_surface'){
  const label=['Recorded skeleton',provenance].filter(Boolean).join(' · ');
  return {label,detail:'Input: '+label+'. Recorded joint positions'+(body?' with '+body+' joint semantics':'')+' and a derived surface.'};
 }
 const observed={label:'Observed head + wrists',detail:'Input: observed head and wrists. Full-body motion is reconstructed before retargeting.'};
 if(format==='observed_head_wrists'||evidence==='observed-head-wrists')return observed;
 if(family==='robot'||(!family&&robot)){
  const label=[field('source_robot_label')||native?.source_robot_label?.trim()||robot?.toUpperCase()||'Robot',provenance||'joint-state motion'].join(' · ');
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

/** A display name may be borrowed only for the exact same source robot identity. */
export function sourceRobotDisplayLabel(source:{source_robot_id?:string;source_robot_label?:string}|undefined,recording?:RecordingSourceMetadata){
 const explicit=source?.source_robot_label?.trim();if(explicit)return explicit;
 const field=(key:'source_robot_id'|'source_robot_label')=>{
  const value=recording?.[key]??recording?.metadata?.[key];return typeof value==='string'?value.trim():'';
 };
 if(source?.source_robot_id&&source.source_robot_id===field('source_robot_id')&&field('source_robot_label'))return field('source_robot_label');
 return source?.source_robot_id?.toUpperCase()||'Robot';
}

/** Scene-bound inputs, including morphology transforms and CV, are immutable. */
export function withSourcePresentation(scene:SceneAsset,source=scene.source_robot):SceneAsset{
 if(!source||scene.source_robot||scene.source_body||scene.robots?.length||scene.output_modalities||scene.generation_status)return scene;
 const result={...scene,source_robot:source,source_body:undefined};
 if(source.video_role==='preview-only'){
  // Only a raw catalog preview may discard its non-source auxiliary video.
  Object.assign(result,{video:null,cloud:undefined,source_camera:undefined,segmented_instances:undefined,
   patch_videos:undefined,patch_kinds:undefined,wrist_videos:[]});
 }
 return result;
}

/** Read the historical scene's input; an equal recording ID is not provenance. */
export function sceneSourceMetadata(scene:SceneAsset,recording?:RecordingSourceMetadata):RecordingSourceMetadata{
 const bound=scene.source_robot||scene.source_body;
 if(!bound)return {};
 const current=recording?.source_motion;
 const sameNativeSource=!!(scene.source_robot&&current&&bound.source_sha256&&
  bound.source_sha256===current.source_sha256&&bound.source_robot_id===current.source_robot_id&&
  bound.frames===current.frames&&bound.fps===current.fps&&scene.frames===bound.frames&&scene.fps===bound.fps);
 const metadata:RecordingSourceMetadata=sameNativeSource?{...recording}:{};
 for(const key of ['source_family','source_evidence','source_policy','source_robot_id','source_robot_label','source_format'] as const){
  if(bound[key])metadata[key]=bound[key];
 }
 metadata.source_body_family=bound.source_family;
 if(bound.source_robot_id)metadata.source_robot_label=sourceRobotDisplayLabel(bound,recording);
 if(scene.source_robot){metadata.source_motion=scene.source_robot;if(scene.source_robot.source_stage)metadata.source_stage=scene.source_robot.source_stage;}
 return metadata;
}

/** Raw recording metadata and a saved scene's working source are different layers. */
export function sourceInputBadges(recording?:RecordingSourceMetadata,scene?:SceneAsset){
 const raw=recordingSourcePresentation(recording??{});
 const bound=scene&&recordingSourcePresentation(sceneSourceMetadata(scene,recording));
 const separate=bound&&bound.label!=='Source type unspecified'&&bound.label!==raw.label;
 return {
  raw:{text:'Raw input: '+raw.label,detail:raw.detail.replace(/^Input:/,'Raw input:')},
  scene:separate?{text:'Scene input: '+bound.label,detail:'Source bound to this saved scene; this may be an intermediate reconstruction or a historical source.'}:undefined,
 };
}

/** A historical URL must not populate the current recording's cached badge. */
export function recordingSceneCacheKey(recording:{id:string;scene_url?:string},sceneUrl=recording.scene_url){
 const canonical=(url:string)=>url.replace(/^\/api\/v1\/media\//,'/assets/');
 if(!recording.scene_url||!sceneUrl||canonical(recording.scene_url)!==canonical(sceneUrl))return undefined;
 return recording.id+'\n'+canonical(sceneUrl);
}

export function initialSourceVisibility(nativeRobot:boolean,previousNativeRobot:boolean,changedRecording:boolean,checked:boolean){
 if(nativeRobot)return changedRecording||checked;
 return previousNativeRobot?false:checked;
}
