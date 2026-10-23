import * as THREE from 'three';
import type {SceneAsset} from './types';

/** Source identity comes from provenance, never from dataset labels or target robot. */
export function sourceMotionDisplay(body:SceneAsset['source_body']|SceneAsset['source_robot']){
 if(body?.source_family==='robot'||body?.source_robot_id)return {
  label:'Source robot'+(body.source_robot_id?' · '+body.source_robot_id.toUpperCase():''),
  title:'Show original source robot motion, synchronized with the target robot',
 };
 if(['smpl','smplh','smplx'].includes(body?.source_family??''))return {
  label:'SMPL motion',title:'Show cached source SMPL / SMPL-H / SMPL-X motion',
 };
 return {label:'Source motion',title:'Show cached source motion'};
}

/** Optional source motion; loading this layer never starts body inference. */
export class SourceBodyLayer {
 mesh?:THREE.Mesh<THREE.BufferGeometry,THREE.MeshStandardMaterial>;
 private vertices?:Float32Array;
 private pending?:Promise<void>;
 private controller=new AbortController();
 private disposed=false;
 private visible=false;
 private frame=-1;
 private time=0;
 constructor(private parent:THREE.Group,private body:NonNullable<SceneAsset['source_body']>,private fps:number,private basis:number[][]){}

 async setVisible(visible:boolean,time:number){
  this.visible=visible;this.time=time;
  if(this.mesh)this.mesh.visible=visible;
  if(!visible||this.disposed)return;
  if(!this.mesh){
   this.pending??=this.load().catch(error=>{this.pending=undefined;throw error;});
   await this.pending;
  }
  this.pose(this.time);
 }

 private async load(){
  const body=this.body,rate=body.fps??this.fps;
  if(body.world_up!=='z'||![body.frames,body.vertex_count,body.face_count].every(v=>Number.isSafeInteger(v)&&v>0)||!Number.isFinite(rate)||rate<=0)throw new Error('Invalid source motion metadata');
  const buffers=await Promise.all([body.vertices_url,body.faces_url].map(async url=>{
   const response=await fetch(url,{signal:this.controller.signal});
   if(!response.ok)throw new Error('Source motion cache unavailable');
   return response.arrayBuffer();
  }));
  if(this.disposed)return;
  if(buffers[0].byteLength!==body.frames*body.vertex_count*3*4||buffers[1].byteLength!==body.face_count*3*4)throw new Error('Source motion cache size mismatch');
  const faces=new Uint32Array(buffers[1]),vertices=new Float32Array(buffers[0]);
  if(faces.some(v=>v>=body.vertex_count)||vertices.some(v=>!Number.isFinite(v)))throw new Error('Invalid source motion geometry');
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(body.vertex_count*3),3));
  geometry.setIndex(new THREE.BufferAttribute(faces,1));
  this.vertices=vertices;
  this.mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:0xa6cbd7,transparent:true,opacity:.28,depthWrite:false,side:THREE.DoubleSide,roughness:.7}));
  this.mesh.frustumCulled=false;this.mesh.visible=this.visible;this.parent.add(this.mesh);
 }

 pose(time:number){
  this.time=time;
  if(!this.mesh||!this.vertices||this.disposed)return;
  this.mesh.visible=this.visible;
  if(!this.visible)return;
  const frame=Math.min(this.body.frames-1,Math.max(0,Math.floor(time*(this.body.fps??this.fps))));
  if(frame===this.frame)return;
  const position=this.mesh.geometry.getAttribute('position') as THREE.BufferAttribute;
  const count=this.body.vertex_count*3,input=this.vertices.subarray(frame*count,(frame+1)*count),output=position.array,b=this.basis;
  for(let i=0;i<count;i+=3)for(let j=0;j<3;j++)output[i+j]=b[j][0]*input[i]+b[j][1]*input[i+1]+b[j][2]*input[i+2];
  position.needsUpdate=true;this.mesh.geometry.computeVertexNormals();this.frame=frame;
 }

 dispose(){
  this.disposed=true;this.controller.abort();
  if(this.mesh){this.parent.remove(this.mesh);this.mesh.geometry.dispose();this.mesh.material.dispose();}
  this.mesh=undefined;this.vertices=undefined;
 }
}
