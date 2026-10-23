export type Vec3=[number,number,number];
export interface SourceObject {id:string;mesh_url:string;poses_url:string;frames:number;world_up:'z';}
export interface CloudFrame {source_frame:number;time:number;count:number;human_count?:number;positions:number;colors:number;labels:number;}
export interface SceneAsset {
 output_modalities?:import('../platform-api').OutputModality[];
 generation_status?:{robot_id?:string;hand_id?:string;output_modalities?:import('../platform-api').OutputModality[]};
 id:string;title:string;dataset:string;end_effector?:string;fps:number;frames:number;duration:number;world:string;
 video?:{url:string;start_seconds:number}|null;cloud?:{url?:string;format?:string;geometry_url?:string;source_human?:boolean;frames:CloudFrame[];fps:number};source_hands_url?:string;
 source_transform?:{scale:number;anchor_xyz_m:Vec3;world_up:'z';applied?:boolean};
 source_world_to_display?:[Vec3,Vec3,Vec3];
 source_robot?:{source_family:'robot';source_robot_id:string;model_url:string;playback_url:string;frames:number;fps:number;world_up:'z';video_role?:'preview-only'|'recorded';objects?:SourceObject[];source_sha256?:string;source_evidence?:string;source_stage?:string;source_policy?:string;source_format?:string};
 source_body?:{vertices_url:string;faces_url:string;frames:number;vertex_count:number;face_count:number;world_up:'z';fps?:number;source_family?:string;source_robot_id?:string;source_format?:string;source_surface?:string;source_sha256?:string;source_evidence?:string;source_policy?:string;
  objects?:SourceObject[]};
 robots:{id:string;encoder_policy?:string;trajectory_scale?:'preserve'|'match_robot';source_policy?:string;source_policy_version?:string;experimental?:boolean;registry_robot_id?:string;label:string;hand:string;model_url:string;playback_url:string;reference:string;quality?:{mean_mm:number;max_mm:number}}[];
 segmented_instances?:{fps:number;categories?:string[];audit_url?:string;frames:{time:number;source_frame:number;instances:{id:string;label:string;semantic_label:number;bounds:[Vec3,Vec3];points?:number}[]}[]};
 semantic:{task:string;objects:string[];provenance:string;model?:string;instruction?:string;phases?:{start_seconds:number;end_seconds:number;label:string}[]};tactile:{status:string;reason?:string;url?:string;proximity_url?:string};
 wrist_videos:{side:string;url:string;start_seconds:number}[];geometry:Record<string,unknown>;
 patch_videos?:Record<string,string>;patch_kinds?:Record<string,string>;patch_unavailable?:Record<string,string>;camera_poses?:number[][][];
 source_camera?:{poses:number[][][];intrinsic:number[][];width:number;height:number;fps:number};hand_motion_audit_url?:string;hand_motion_audit_urls?:Record<string,string>;
}
export interface Motion {
 fps:number;frames:number;positions:number[][][];quaternions:number[][][];
 tracking:{actual:number[][][];target:number[][][];error_mm:number[][];orientation_deg:number[][];reference:string;fingertip_error_mm?:number[][][]};
 joints:{names:string[];values:number[][];velocity?:number[][];effort?:number[][];unit:string;units?:string[];representation:string};
}
export interface RecordingSourceMetadata {
 source_body_family?:string;source_family?:string;source_evidence?:string;source_stage?:string;source_robot_id?:string;
 source_format?:string;source_policy?:string;motion_source_policy?:string;body_adapter?:string;has_supplied_body?:boolean;
 source_motion?:(SceneAsset['source_robot'])&{source_evidence?:string;source_stage?:string};metadata?:Record<string,unknown>;
}
export interface LibraryClip extends RecordingSourceMetadata {id:string;source_catalog_id?:string;title:string;dataset:string;end_effector?:string;uploaded?:boolean;family:string;duration:number;frames?:number;fps?:number;scene_url?:string;video?:{url:string;start_seconds:number};}
