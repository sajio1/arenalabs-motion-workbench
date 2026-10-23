interface RecordingIdentity {id:string;source_catalog_id?:string;scene_url?:string;}

/** Only an explicit catalog alias establishes that two rows are one recording. */
export function uniqueRecordings<T extends RecordingIdentity>(recordings:readonly T[]):T[]{
 const unique=new Map<string,T>();
 for(const clip of recordings){
  const key=clip.source_catalog_id||clip.id;
  const previous=unique.get(key);
  if(!previous||(!previous.scene_url&&clip.scene_url))unique.set(key,clip);
 }
 return [...unique.values()];
}
