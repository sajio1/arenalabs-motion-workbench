export type Page = 'locomotion' | 'localmanipulation' | 'manipulation';
export interface Robot {id:string;label:string;kind:string;pages:Page[];status:string;conversion_ready:boolean;model_url?:string;neutral_url?:string;report?:{message:string;links:number;joints:number;missing_meshes:string[]};}
export interface Clip {id:string;page:Page;title:string;family:string;description:string;frames:number;fps:number;duration:number;source_name:string;source_format:string;preview?:string;cache?:Record<string,string>;version:string;metadata?:Record<string,unknown>;}
export interface Catalog {page:Page;robots:Robot[];clips:Clip[];}
export interface Playback {fps:number;frames:number;duration:number;geom_ids:number[];positions:number[][][];quaternions:number[][][];skeleton:number[][][];parents:number[];targets:number[][][];clip_id?:string;robot_id?:string;version?:string;quality_status?:string;}
export interface Neutral {geom_ids:number[];positions:number[][];quaternions:number[][];body_positions:number[][];parents:number[];}
export interface Slot {index:number;robot:Robot;clip?:Clip;status:'empty'|'loading'|'queued'|'running'|'ready'|'failed';message?:string;payload?:Playback;token:number;}
