import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import type {Motion,SceneAsset} from './types';

type Model={pending:Promise<THREE.Group>;refs:number;used:number;ready?:THREE.Group};
/** Cache owns immutable geometry/materials; scene instances own transforms only. */
export class RobotAssets {
 private models=new Map<string,Model>();
 private motions=new Map<string,{pending:Promise<Motion>;expires:number}>();
 constructor(private loader:GLTFLoader,private capacity=3){}
 private model(url:string){
  let entry=this.models.get(url);
  if(!entry){
   entry={refs:0,used:performance.now(),pending:this.loader.loadAsync(url).then(gltf=>{
    gltf.scene.traverse(o=>{if(o instanceof THREE.Mesh)for(const material of Array.isArray(o.material)?o.material:[o.material]){
     if(material instanceof THREE.MeshStandardMaterial){material.metalness=.08;material.roughness=.65;}
    }});
    entry!.ready=gltf.scene;this.trim();return gltf.scene;
   }).catch(error=>{this.models.delete(url);throw error;})};
   this.models.set(url,entry);
  }
  entry.used=performance.now();return entry;
 }
 private motion(url:string){
  let item=this.motions.get(url);
  if(!item||item.expires<performance.now()){
   item={expires:Infinity,pending:fetch(url).then(async r=>{if(!r.ok)throw new Error('Robot motion unavailable');return r.json() as Promise<Motion>;}).then(value=>{item!.expires=performance.now()+2000;return value;}).catch(error=>{this.motions.delete(url);throw error;})};
   this.motions.set(url,item);
   for(const key of this.motions.keys())if(this.motions.size>3&&key!==url)this.motions.delete(key);
  }
  return item.pending;
 }
 prefetch(row:SceneAsset['robots'][number]){void Promise.all([this.model(row.model_url).pending,this.motion(row.playback_url)]).catch(()=>{});}
 async acquire(row:SceneAsset['robots'][number]){
  const entry=this.model(row.model_url);entry.refs++;
  const release=()=>{entry.refs--;this.trim();};
  try{const [model,motion]=await Promise.all([entry.pending,this.motion(row.playback_url)]);this.trim();return {node:model.clone(true),motion,release};}
  catch(error){release();throw error;}
 }
 private trim(){
  for(const [key,entry] of [...this.models].sort((a,b)=>a[1].used-b[1].used)){
   if(this.models.size<=this.capacity)break;
   if(entry.refs||!entry.ready)continue;
   const geometries=new Set<THREE.BufferGeometry>(),materials=new Set<THREE.Material>();
   entry.ready.traverse(o=>{if(o instanceof THREE.Mesh){geometries.add(o.geometry);for(const m of Array.isArray(o.material)?o.material:[o.material])materials.add(m);}});
   geometries.forEach(g=>g.dispose());materials.forEach(m=>{for(const value of Object.values(m))if(value instanceof THREE.Texture)value.dispose();m.dispose();});
   this.models.delete(key);
  }
 }
}
