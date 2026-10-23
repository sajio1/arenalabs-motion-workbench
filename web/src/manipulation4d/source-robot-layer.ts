import * as THREE from 'three';
import type {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import type {SceneAsset} from './types';

/** Original rigid-link playback, independent of the selected target robot. */
export class SourceRobotLayer {
 mesh?:THREE.Group;
 private pending?:Promise<void>;
 private controller=new AbortController();
 private disposed=false;private visible=false;private time=0;
 private motion?:{fps:number;frames:number;positions:number[][][];quaternions:number[][][]};
 private nodes=new Map<number,THREE.Object3D>();
 constructor(private parent:THREE.Group,private source:NonNullable<SceneAsset['source_robot']>,private basis:number[][],private loader:GLTFLoader){}
 async setVisible(visible:boolean,time:number){
  this.visible=visible;this.time=time;if(this.mesh)this.mesh.visible=visible;
  if(!visible||this.disposed)return;
  if(!this.mesh){this.pending??=this.load().catch(error=>{this.pending=undefined;throw error;});await this.pending;}
  this.pose(this.time);
 }
 private release(node:THREE.Object3D){node.traverse(o=>{if(o instanceof THREE.Mesh){o.geometry.dispose();for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose();}});}
 private async load(){
  const s=this.source;
  if(s.world_up!=='z'||!Number.isSafeInteger(s.frames)||s.frames<1||!Number.isFinite(s.fps)||s.fps<=0)throw new Error('Invalid source robot clock');
  const response=await fetch(s.playback_url,{signal:this.controller.signal});if(!response.ok)throw new Error('Source robot motion unavailable');
  const motion=await response.json(),count=motion.positions?.[0]?.length;
  if(motion.frames!==s.frames||motion.fps!==s.fps||!Number.isInteger(count)||count<1||motion.positions?.length!==s.frames||motion.quaternions?.length!==s.frames)throw new Error('Source robot clock or geometry mismatch');
  for(let f=0;f<s.frames;f++){
   if(motion.positions[f].length!==count||motion.quaternions[f].length!==count)throw new Error('Source robot geometry count changed');
   for(let g=0;g<count;g++)for(const [v,n] of [[motion.positions[f][g],3],[motion.quaternions[f][g],4]] as [number[],number][]){if(v?.length!==n||!v.every(Number.isFinite))throw new Error('Invalid source robot pose');}
  }
  if(this.disposed)return;
  const gltf=await this.loader.loadAsync(s.model_url);
  if(this.disposed){this.release(gltf.scene);return;}
  const nodes=new Map<number,THREE.Object3D>();gltf.scene.traverse(o=>{if(/^geom_\d+$/.test(o.name))nodes.set(Number(o.name.slice(5)),o);});
  if(nodes.size!==count||[...nodes.keys()].some(i=>i>=count)){this.release(gltf.scene);throw new Error('Source robot mesh does not match its motion');}
  const b=this.basis,group=new THREE.Group();group.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(b[0][0],b[0][1],b[0][2],0,b[1][0],b[1][1],b[1][2],0,b[2][0],b[2][1],b[2][2],0,0,0,0,1));
  gltf.scene.traverse(o=>{if(o instanceof THREE.Mesh){for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose();o.material=new THREE.MeshStandardMaterial({color:0xe8e9e6,roughness:.72,metalness:.05,side:THREE.DoubleSide});}});
  group.add(gltf.scene);group.visible=this.visible;this.parent.add(group);this.nodes=nodes;this.motion=motion;this.mesh=group;
 }
 pose(time:number){
  this.time=time;if(!this.mesh||!this.motion||this.disposed)return;this.mesh.visible=this.visible;if(!this.visible)return;
  const frame=Math.min(this.motion.frames-1,Math.max(0,Math.floor(time*this.motion.fps)));
  for(const [i,node] of this.nodes){node.position.fromArray(this.motion.positions[frame][i]);node.quaternion.fromArray(this.motion.quaternions[frame][i]);}
 }
 dispose(){this.disposed=true;this.controller.abort();if(this.mesh){this.parent.remove(this.mesh);this.release(this.mesh);}this.mesh=undefined;this.motion=undefined;this.nodes.clear();}
}
