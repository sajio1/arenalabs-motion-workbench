export type Vec3=[number,number,number];
export interface CloudFrame {source_frame:number;time:number;count:number;positions:number;colors:number;labels:number;}
export interface ObjectFrame {time:number;source_frame:number;status:string;matrix?:number[][];bounds?:[Vec3,Vec3];surface_rmse_m?:number;}
export interface SurfaceFrame {time:number;source_frame:number;vertices:number;faces:number;positions:number;colors:number;indices:number;bounds:[Vec3,Vec3]|null;boxes?:{id:number;bounds:[Vec3,Vec3]}[];}
export interface SceneAsset {
 id:string;title:string;dataset:string;end_effector?:string;fps:number;frames:number;duration:number;world:string;
 video:{url:string;start_seconds:number};cloud?:{url:string;frames:CloudFrame[];fps:number};source_hands_url?:string;
 robots:{id:string;encoder_policy?:string;experimental?:boolean;registry_robot_id?:string;label:string;hand:string;model_url:string;playback_url:string;reference:string;quality?:{mean_mm:number;max_mm:number}}[];
 objects:{id:string;label:string;mesh_url:string;track_url:string;inferred:boolean;semantic_label:number;quality:Record<string,unknown>;observed_surface?:{url:string;fps:number;frames:SurfaceFrame[];provenance:string;all_instances?:boolean}}[];
 segmented_instances?:{fps:number;categories?:string[];audit_url?:string;frames:{time:number;source_frame:number;instances:{id:string;label:string;semantic_label:number;bounds:[Vec3,Vec3];points?:number}[]}[]};
 semantic:{task:string;objects:string[];provenance:string;model?:string;instruction?:string;phases?:{start_seconds:number;end_seconds:number;label:string}[]};tactile:{status:string;reason?:string;url?:string;proximity_url?:string};
 wrist_videos:{side:string;url:string;start_seconds:number}[];geometry:Record<string,unknown>;
 patch_videos?:Record<string,string>;camera_poses?:number[][][];
 spatial_memory?:{url:string;count:number;positions:number;colors:number;first_frames:number;last_frames:number;source_frame_ids:number[];active_points:number[];voxel_m:number;mode?:string};
 completed_assets?:{kind:string;model_url:string;full_asset_url:string;track_url:string;fps:number;estimated:boolean}[];completion_default?:boolean;completion_available?:boolean;source_camera?:{poses:number[][][];intrinsic:number[][];width:number;height:number;fps:number};hand_motion_audit_url?:string;hand_motion_audit_urls?:Record<string,string>;
}
export interface Motion {
 fps:number;frames:number;positions:number[][][];quaternions:number[][][];
 tracking:{actual:number[][][];target:number[][][];error_mm:number[][];orientation_deg:number[][];reference:string;fingertip_error_mm?:number[][][]};
 joints:{names:string[];values:number[][];velocity?:number[][];effort?:number[][];unit:string;units?:string[];representation:string};
}
export interface LibraryClip {id:string;title:string;dataset:string;end_effector?:string;uploaded?:boolean;family:string;duration:number;scene_url?:string;video?:{url:string;start_seconds:number};metadata?:Record<string,unknown>;}

export interface CompleteFrame{source_frame:number;time:number;instances:{id:number;matrix:number[][];bounds:[Vec3,Vec3];status:string;observed:boolean}[];}
