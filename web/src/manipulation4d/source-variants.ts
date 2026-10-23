import type {LibraryClip} from './types';

type VariantRecording=Pick<LibraryClip,'id'|'source_group_id'|'source_variants'>;
export interface SourcePreviewResponse {recording_id:string;source_motion:NonNullable<LibraryClip['source_motion']>;cache_hit?:boolean;}

/** Variants are distinct recordings, not catalog aliases or inferred robot names. */
export function sourceVariantChoices(recording?:VariantRecording){
 if(!recording?.source_group_id)return [];
 const seen=new Set<string>();
 const choices=(recording.source_variants??[]).filter(variant=>{
  if(typeof variant.id!=='string'||!variant.id||typeof variant.label!=='string'||!variant.label.trim()||seen.has(variant.id))return false;
  seen.add(variant.id);return true;
 }).map(variant=>({...variant,available:variant.available!==false,selected:variant.id===recording.id}));
 return choices.length>1&&choices.some(variant=>variant.selected)?choices:[];
}

/** Fetch the target's own scene/cache; never reuse the originating scene or generate motion. */
export async function selectSourceVariant(recording:VariantRecording,id:string,actions:{
 fetchRecording:(id:string)=>Promise<LibraryClip>;
 isCurrent:()=>boolean;
 selectRecording:(recording:LibraryClip)=>Promise<void>;
}){
 const choice=sourceVariantChoices(recording).find(variant=>variant.id===id);
 if(!choice)throw new Error('This input variant is not declared for the recording.');
 if(!choice.available)throw new Error(choice.reason||'This input variant is unavailable.');
 if(choice.selected)return false;
 const target=await actions.fetchRecording(choice.id);
 if(!actions.isCurrent())return false;
 if(target.id!==choice.id||target.source_group_id!==recording.source_group_id)throw new Error('The requested input belongs to a different recording group.');
 await actions.selectRecording(target);
 return true;
}

/** Materialize only explicitly offered source geometry, never a target-robot run. */
export async function prepareSourcePreview(recording:LibraryClip,fetchPreview:(id:string)=>Promise<SourcePreviewResponse>,isCurrent:()=>boolean){
 if(recording.scene_url||recording.source_motion||recording.source_preview_available!==true)return recording;
 const preview=await fetchPreview(recording.id);
 if(!isCurrent())return undefined;
 if(preview.recording_id!==recording.id||!preview.source_motion)throw new Error('Source preview did not match the selected recording.');
 return {...recording,source_motion:preview.source_motion};
}
