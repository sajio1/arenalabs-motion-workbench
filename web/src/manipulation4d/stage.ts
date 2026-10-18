import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import type {SceneAsset,Motion,ObjectFrame,SurfaceFrame,CompleteFrame} from './types';
import {TraceGlyph} from './trace-glyph';

export class Workspace {
 scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(43,1,.01,50);
 renderer=new THREE.WebGLRenderer({antialias:true});controls:OrbitControls;loader=new GLTFLoader();
 grid=new THREE.GridHelper(12,48,0x395745,0x20382b);root=new THREE.Group();robot?:THREE.Group;geoms=new Map<number,THREE.Object3D>();motion?:Motion;asset?:SceneAsset;
 completeRows:{kind:string;fps:number;frames:CompleteFrame[];nodes:Map<number,THREE.Object3D>;boxes:Map<number,THREE.Box3Helper>}[]=[];
 objectRows:{node:THREE.Object3D;box:THREE.Box3Helper;frames:ObjectFrame[];label:number;surface?:{node:THREE.Mesh;buffer:ArrayBuffer;frames:SurfaceFrame[];fps:number;frame:number;all:boolean;boxes:THREE.Box3Helper[]}}[]=[];
 cloud?:THREE.Points;cloudBuffer?:ArrayBuffer;cloudGeometry?:THREE.BufferGeometry;cloudMaterial?:THREE.ShaderMaterial;cloudFrame=-1;
 paths:TraceGlyph[]=[];errors:THREE.LineSegments[]=[];showCloud=true;showObjects=true;showCompletion=false;showBoxes=true;showTraces=true;showRobot=true;segments=false;orbit=false;
 serial=0;robotSerial=0;currentTime=0;
 instanceBoxes=new Map<string,THREE.Box3Helper>();
 constructor(public host:HTMLElement){
  host.append(this.renderer.domElement);this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));this.renderer.outputColorSpace=THREE.SRGBColorSpace;
  this.scene.background=new THREE.Color('#0b1712');this.scene.add(this.root);
  this.scene.add(new THREE.HemisphereLight(0xe8f4e4,0x294035,2.4));const light=new THREE.DirectionalLight(0xfff3de,3);light.position.set(2,5,3);this.scene.add(light);
  const grid=this.grid;(grid.material as THREE.Material).transparent=true;(grid.material as THREE.Material).opacity=.25;this.scene.add(grid);
  this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;this.controls.minDistance=.12;this.controls.maxDistance=15;
  this.controls.addEventListener('start',()=>this.orbit=false);this.camera.position.set(1.5,1.7,2.2);this.controls.target.set(0,1,0);
  new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;if(w<1||h<1)return;this.renderer.setSize(w,h);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();}).observe(host);
 }
 async load(asset:SceneAsset){
  const token=++this.serial;this.asset=asset;const ground=asset.geometry.ground as {height_m?:number}|undefined;this.grid.position.y=ground?.height_m??0;this.motion=undefined;this.geoms.clear();this.objectRows=[];this.completeRows=[];this.paths=[];this.errors=[];
  this.root.traverse(o=>{if(o instanceof THREE.Mesh||o instanceof THREE.Points||o instanceof THREE.Line||o instanceof THREE.LineSegments){o.geometry.dispose();const materials=Array.isArray(o.material)?o.material:[o.material];materials.forEach(m=>m.dispose());}});this.root.clear();this.showRobot=true;this.showCloud=true;this.instanceBoxes.clear();this.robot=undefined;this.cloudFrame=-1;this.cloud=undefined;this.cloudBuffer=undefined;
  if(asset.cloud){
   const response=await fetch(asset.cloud.url);if(!response.ok)throw new Error('Point-cloud stream unavailable');const buffer=await response.arrayBuffer();if(token!==this.serial)return;
   this.cloudBuffer=buffer;const count=Math.max(...asset.cloud.frames.map(f=>f.count));const geometry=new THREE.BufferGeometry();
   geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(count*3),3));geometry.setAttribute('color',new THREE.BufferAttribute(new Float32Array(count*3),3));geometry.setAttribute('label',new THREE.BufferAttribute(new Float32Array(count),1));
   const material=new THREE.ShaderMaterial({uniforms:{pointSize:{value:1.5*this.renderer.getPixelRatio()},segments:{value:false},hidden:{value:new THREE.Vector4(-1,-1,-1,-1)}},vertexShader:`attribute vec3 color;attribute float label;varying vec3 vColor;varying float vLabel;uniform float pointSize;void main(){vColor=color;vLabel=label;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);gl_PointSize=pointSize*(label>.5?1.8:1.);}`,fragmentShader:`varying vec3 vColor;varying float vLabel;uniform bool segments;uniform vec4 hidden;void main(){if(any(lessThan(abs(vec4(vLabel)-hidden),vec4(.1))))discard;if(length(gl_PointCoord-vec2(.5))>.5)discard;vec3 c=vColor;if(segments&&vLabel>.5)c=vec3(.65)+.35*cos(6.2831853*(fract(vLabel*.61803398875)+vec3(0.,.333333,.666667)));gl_FragColor=vec4(c,1.0);}`});
   this.cloudGeometry=geometry;this.cloudMaterial=material;this.cloud=new THREE.Points(geometry,material);this.cloud.frustumCulled=false;this.root.add(this.cloud);
  }
  await Promise.all(asset.objects.map(async object=>{
   const [gltf,frames]=await Promise.all([this.loader.loadAsync(object.mesh_url),fetch(object.track_url).then(r=>r.json())]);if(token!==this.serial)return;
   gltf.scene.traverse(o=>{if(o instanceof THREE.Mesh){const m=o.material as THREE.MeshStandardMaterial;m.metalness=0;m.roughness=.85;m.side=THREE.DoubleSide;}});
   const box=new THREE.Box3Helper(new THREE.Box3(),object.semantic_label===1?0xe7b968:0x6ed7d0);this.root.add(gltf.scene,box);
   let surface;
   if(object.observed_surface){
    const response=await fetch(object.observed_surface.url);if(!response.ok)throw new Error('Observed object surfaces unavailable');const buffer=await response.arrayBuffer();if(token!==this.serial)return;
    const node=new THREE.Mesh(new THREE.BufferGeometry(),new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.DoubleSide}));node.frustumCulled=false;this.root.add(node);
    const boxes=Array.from({length:Math.max(...object.observed_surface.frames.map(row=>row.boxes?.length||0))},()=>new THREE.Box3Helper(new THREE.Box3(),object.semantic_label===1?0xe7b968:0x6ed7d0));boxes.forEach(box=>this.root.add(box));
    surface={node,buffer,frames:object.observed_surface.frames,fps:object.observed_surface.fps,frame:-1,all:!!object.observed_surface.all_instances,boxes};
   }
   this.objectRows.push({node:gltf.scene,box,frames,label:object.semantic_label,surface});
  }));
  await Promise.all((asset.completed_assets||[]).map(async row=>{
   const [gltf,frames]=await Promise.all([this.loader.loadAsync(row.model_url),fetch(row.track_url).then(r=>r.json()) as Promise<CompleteFrame[]>]);if(token!==this.serial)return;
   gltf.scene.traverse(o=>{if(o instanceof THREE.Mesh){const m=o.material as THREE.MeshStandardMaterial;m.roughness=.8;m.metalness=0;m.side=THREE.DoubleSide;}});
   const nodes=new Map<number,THREE.Object3D>(),boxes=new Map<number,THREE.Box3Helper>();const ids=new Set(frames.flatMap(frame=>frame.instances.map(instance=>instance.id)));
   for(const id of ids){const node=gltf.scene.clone(true);node.matrixAutoUpdate=false;node.visible=false;nodes.set(id,node);const box=new THREE.Box3Helper(new THREE.Box3(),0x9dbb83);box.visible=false;boxes.set(id,box);this.root.add(node,box);}
   this.completeRows.push({kind:row.kind,fps:row.fps,frames,nodes,boxes});
  }));
  for(const frame of asset.segmented_instances?.frames||[])for(const item of frame.instances)if(!this.instanceBoxes.has(item.id)){const color=new THREE.Color().fromArray([0,1/3,2/3].map(offset=>.65+.35*Math.cos(2*Math.PI*((item.semantic_label*.61803398875)%1+offset))));const box=new THREE.Box3Helper(new THREE.Box3(),color);box.visible=false;this.instanceBoxes.set(item.id,box);this.root.add(box);}
  if(token===this.serial)this.focus(asset.robots.length?'task':'objects');
 }
 async loadRobot(row:SceneAsset['robots'][number]){
  const token=this.serial,request=++this.robotSerial;const [gltf,motion]=await Promise.all([this.loader.loadAsync(row.model_url),fetch(row.playback_url).then(r=>r.json())]);if(token!==this.serial||request!==this.robotSerial)return;
  if(this.robot)this.root.remove(this.robot);this.geoms.clear();this.robot=gltf.scene;this.motion=motion;
  this.robot.traverse(o=>{if(o.name.startsWith('geom_'))this.geoms.set(Number(o.name.slice(5)),o);if(o instanceof THREE.Mesh){const m=o.material as THREE.MeshStandardMaterial;m.metalness=.08;m.roughness=.65;}});this.root.add(this.robot);
  this.paths.forEach(o=>{this.root.remove(o);o.dispose();});this.errors.forEach(o=>{this.root.remove(o);o.geometry.dispose();});this.paths=[];this.errors=[];
  for(const color of [0xffbb68,0x76d7df]){
   const path=new TraceGlyph(21,color,.0055,.0026,.45);this.root.add(path);this.paths.push(path);
   const error=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0xff6575,depthTest:false}));this.root.add(error);this.errors.push(error);
  }
 }
 focus(mode:'task'|'body'|'objects'|'hands'){
  this.grid.visible=true;
  this.camera.up.set(0,1,0);this.camera.fov=43;this.camera.updateProjectionMatrix();const values=this.motion?.tracking.target;let center=new THREE.Vector3(0,1.2,0);
  if(values?.length){const frame=Math.min(values.length-1,Math.floor(this.currentTime*(this.motion?.fps??30)));center=new THREE.Vector3(...values[frame][0] as [number,number,number]).add(new THREE.Vector3(...values[frame][1] as [number,number,number])).multiplyScalar(.5);}
  if(mode==='hands'&&this.motion){const frame=Math.min(this.motion.frames-1,Math.floor(this.currentTime*this.motion.fps));const hands=this.motion.tracking.actual[frame];center=new THREE.Vector3(...hands[0] as [number,number,number]).add(new THREE.Vector3(...hands[1] as [number,number,number])).multiplyScalar(.5);const span=Math.max(.20,new THREE.Vector3(...hands[0] as [number,number,number]).distanceTo(new THREE.Vector3(...hands[1] as [number,number,number])));this.camera.position.copy(center).add(new THREE.Vector3(-.6,.4,-1).multiplyScalar(span*1.7));}
  else if(mode==='objects'){
   const bounds=new THREE.Box3();const segmented=this.asset?.segmented_instances;if(segmented){const frame=segmented.frames[Math.min(segmented.frames.length-1,Math.floor(this.currentTime*segmented.fps))];for(const item of frame.instances){bounds.expandByPoint(new THREE.Vector3(...item.bounds[0]));bounds.expandByPoint(new THREE.Vector3(...item.bounds[1]));}}if(this.showCompletion)for(const row of this.completeRows){const f=row.frames[Math.min(row.frames.length-1,Math.floor(this.currentTime*row.fps))];for(const item of f.instances){bounds.expandByPoint(new THREE.Vector3(...item.bounds[0]));bounds.expandByPoint(new THREE.Vector3(...item.bounds[1]));}}if(!this.showCompletion||!this.completeRows.length)for(const object of this.objectRows){const frames=object.surface?.frames||object.frames;const row=frames[Math.min(frames.length-1,Math.floor(this.currentTime*15))];if(row.bounds){bounds.expandByPoint(new THREE.Vector3(...row.bounds[0]));bounds.expandByPoint(new THREE.Vector3(...row.bounds[1]));}}
   if(bounds.isEmpty()&&this.asset?.source_camera){
    // With no object mask, start at the recorded camera rather than world origin.
    // This changes the display viewpoint only; source points and poses stay intact.
    const camera=this.asset.source_camera,index=Math.min(camera.poses.length-1,Math.floor(this.currentTime*camera.fps));const pose=new THREE.Matrix4().fromArray(camera.poses[index].flat()).transpose();this.camera.fov=Math.min(150,THREE.MathUtils.radToDeg(2*Math.atan(Math.max(camera.height/(2*camera.intrinsic[1][1]),camera.width/(2*camera.intrinsic[0][0]*this.camera.aspect)))));this.camera.updateProjectionMatrix();this.camera.position.setFromMatrixPosition(pose);this.camera.up.set(0,-1,0).transformDirection(pose);center.copy(this.camera.position).add(new THREE.Vector3(0,0,1).transformDirection(pose));this.controls.target.copy(center);this.controls.update();return;
   }
   if(!bounds.isEmpty())bounds.getCenter(center);const size=bounds.isEmpty() ? .3 : Math.max(.2,bounds.getSize(new THREE.Vector3()).length());this.camera.position.copy(center).add(new THREE.Vector3(-.8,.65,-1.2).multiplyScalar(size*1.6));
  }
  else if(mode==='body'){center.y=.9;this.camera.position.copy(center).add(new THREE.Vector3(-1.8,.7,-2.5));}
  else this.camera.position.copy(center).add(new THREE.Vector3(-.8,.5,-1.1));this.controls.target.copy(center);this.controls.update();
 }
 pose(time:number){
  this.grid.visible=true;this.currentTime=time;const motion=this.motion;const frame=motion?Math.min(motion.frames-1,Math.floor(time*motion.fps)):0;
  if(motion){for(const [i,node] of this.geoms){node.position.fromArray(motion.positions[frame][i]);node.quaternion.fromArray(motion.quaternions[frame][i]);}if(this.robot)this.robot.visible=this.showRobot;
   for(let side=0;side<2;side++){
    const points=motion.tracking.actual.slice(frame,Math.min(motion.frames,frame+21)).map(f=>f[side]);this.paths[side].update(points);this.paths[side].visible=this.showTraces;
    this.errors[side].geometry.dispose();this.errors[side].geometry=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...motion.tracking.actual[frame][side] as [number,number,number]),new THREE.Vector3(...motion.tracking.target[frame][side] as [number,number,number])]);this.errors[side].visible=this.showTraces;
   }
  }
  const cloud=this.asset?.cloud;if(cloud&&this.cloudBuffer&&this.cloudGeometry){const index=Math.min(cloud.frames.length-1,Math.floor(time*cloud.fps));
   if(index!==this.cloudFrame){const f=cloud.frames[index];const xyz=new Float32Array(this.cloudBuffer,f.positions,f.count*3);const rgb=new Uint8Array(this.cloudBuffer,f.colors,f.count*3);const labels=new Uint8Array(this.cloudBuffer,f.labels,f.count);
    const p=this.cloudGeometry.getAttribute('position') as THREE.BufferAttribute,c=this.cloudGeometry.getAttribute('color') as THREE.BufferAttribute,l=this.cloudGeometry.getAttribute('label') as THREE.BufferAttribute;(p.array as Float32Array).set(xyz);
    for(let j=0;j<rgb.length;j++)c.array[j]=rgb[j]/255;for(let j=0;j<labels.length;j++)l.array[j]=labels[j];p.needsUpdate=c.needsUpdate=l.needsUpdate=true;this.cloudGeometry.setDrawRange(0,f.count);this.cloudFrame=index;

   }
  }
  this.instanceBoxes.forEach(box=>box.visible=false);const segmented=this.asset?.segmented_instances;
  if(segmented){const row=segmented.frames[Math.min(segmented.frames.length-1,Math.floor(time*segmented.fps))];for(const item of row.instances){const box=this.instanceBoxes.get(item.id);if(!box)continue;box.visible=this.showBoxes;box.box.min.fromArray(item.bounds[0]);box.box.max.fromArray(item.bounds[1]);}}
  const hidden:number[]=[];
  for(const object of this.objectRows){const f=object.frames[Math.min(object.frames.length-1,Math.floor(time*15))];object.node.visible=!this.completeRows.length&&this.showObjects&&this.showCompletion&&f.status==='observed'&&!!f.matrix;object.box.visible=!(this.completeRows.length&&this.showCompletion)&&this.showBoxes&&f.status==='observed'&&!!f.bounds;
   if(f.matrix){object.node.matrixAutoUpdate=false;object.node.matrix.fromArray(f.matrix.flat()).transpose();object.node.matrixWorldNeedsUpdate=true;}
   if(f.bounds){object.box.box.min.fromArray(f.bounds[0]);object.box.box.max.fromArray(f.bounds[1]);}
   let observedVisible=false;
   if(object.surface){const surface=object.surface,index=Math.min(surface.frames.length-1,Math.floor(time*surface.fps)),row=surface.frames[index];
    observedVisible=this.showObjects&&!this.showCompletion&&row.faces>0;surface.node.visible=observedVisible;
    if(index!==surface.frame){
     const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(surface.buffer,row.positions,row.vertices*3),3));
     geometry.setAttribute('color',new THREE.BufferAttribute(new Uint8Array(surface.buffer,row.colors,row.vertices*3),3,true));geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(surface.buffer,row.indices,row.faces*3),1));surface.node.geometry.dispose();surface.node.geometry=geometry;surface.frame=index;
    }
    if(!this.showCompletion){object.box.visible=!row.boxes&&this.showBoxes&&!!row.bounds&&row.faces>0;if(row.bounds){object.box.box.min.fromArray(row.bounds[0]);object.box.box.max.fromArray(row.bounds[1]);}}
    surface.boxes.forEach((box,j)=>{const entry=row.boxes?.[j];box.visible=this.showBoxes&&!this.showCompletion&&!!entry;if(entry){box.box.min.fromArray(entry.bounds[0]);box.box.max.fromArray(entry.bounds[1]);}});
    if(observedVisible&&surface.all)hidden.push(object.label+2);
   }
   if(object.node.visible||observedVisible)hidden.push(object.label);
  }
  for(const row of this.completeRows){
   row.nodes.forEach(node=>node.visible=false);row.boxes.forEach(box=>box.visible=false);
   const frame=row.frames[Math.min(row.frames.length-1,Math.floor(time*row.fps))];
   if(this.showCompletion&&this.showObjects){for(const item of frame.instances){const node=row.nodes.get(item.id)!;node.visible=true;node.matrix.fromArray(item.matrix.flat()).transpose();node.matrixWorldNeedsUpdate=true;const box=row.boxes.get(item.id)!;box.visible=this.showBoxes;box.box.min.fromArray(item.bounds[0]);box.box.max.fromArray(item.bounds[1]);}const label=this.asset?.objects.find(o=>o.label===row.kind)?.semantic_label;if(label!==undefined)hidden.push(label);}
  }
  if(this.cloud)this.cloud.visible=this.showCloud;if(this.cloudMaterial){this.cloudMaterial.uniforms.segments.value=this.segments;this.cloudMaterial.uniforms.hidden.value.set(hidden[0]??-1,hidden[1]??-1,hidden[2]??-1,hidden[3]??-1);}
  this.controls.autoRotate=this.orbit;this.controls.autoRotateSpeed=.35;this.controls.update();this.renderer.render(this.scene,this.camera);
  return frame;
 }
}
