import type {SceneAsset} from './types';

/** Restore only published trajectory assets; this never requests generation. */
export function cachedRobotSelection(scene:Pick<SceneAsset,'robots'|'generation_status'|'output_modalities'>,robotId='',handId=''){
 const outputs=scene.output_modalities??scene.generation_status?.output_modalities;
 if(outputs&&!outputs.includes('trajectory'))return undefined;
 const candidates=scene.robots.filter(r=>r.model_url&&r.playback_url&&r.hand);
 const matches=(row:SceneAsset['robots'][number],id:string)=>row.id===id||row.registry_robot_id===id;
 // Do not replace an active user selection with a different cached result.
 if(robotId||handId)return candidates.find(r=>r.id===robotId&&r.hand===handId);
 const generated=scene.generation_status;
 const requested=candidates.filter(r=>(!generated?.robot_id||matches(r,generated.robot_id))&&(!generated?.hand_id||r.hand===generated.hand_id));
 return requested.length===1?requested[0]:undefined;
}
