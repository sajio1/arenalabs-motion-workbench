import type {SceneAsset} from './types';

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
