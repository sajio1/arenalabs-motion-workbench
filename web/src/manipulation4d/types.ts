export type Vec3=[number,number,number];
export interface CloudFrame {source_frame:number;time:number;count:number;positions:number;colors:number;labels:number;}
export interface SceneAsset {
 id:string;title:string;dataset:string;end_effector?:string;fps:number;frames:number;duration:number;world:string;
 video:{url:string;start_seconds:number};cloud?:{url:string;frames:CloudFrame[];fps:number};source_hands_url?:string;
 robots:{id:string;encoder_policy?:string;experimental?:boolean;registry_robot_id?:string;label:string;hand:string;model_url:string;playback_url:string;reference:string;quality?:{mean_mm:number;max_mm:number}}[];
 segmented_instances?:{fps:number;categories?:string[];audit_url?:string;frames:{time:number;source_frame:number;instances:{id:string;label:string;semantic_label:number;bounds:[Vec3,Vec3];points?:number}[]}[]};
 semantic:{task:string;objects:string[];provenance:string;model?:string;instruction?:string;phases?:{start_seconds:number;end_seconds:number;label:string}[]};tactile:{status:string;reason?:string;url?:string;proximity_url?:string};
 wrist_videos:{side:string;url:string;start_seconds:number}[];geometry:Record<string,unknown>;
 patch_videos?:Record<string,string>;camera_poses?:number[][][];
 source_camera?:{poses:number[][][];intrinsic:number[][];width:number;height:number;fps:number};hand_motion_audit_url?:string;hand_motion_audit_urls?:Record<string,string>;
}
export interface Motion {
 fps:number;frames:number;positions:number[][][];quaternions:number[][][];
 tracking:{actual:number[][][];target:number[][][];error_mm:number[][];orientation_deg:number[][];reference:string;fingertip_error_mm?:number[][][]};
 joints:{names:string[];values:number[][];velocity?:number[][];effort?:number[][];unit:string;units?:string[];representation:string};
}
export interface LibraryClip {id:string;title:string;dataset:string;end_effector?:string;uploaded?:boolean;family:string;duration:number;scene_url?:string;video?:{url:string;start_seconds:number};metadata?:Record<string,unknown>;}
