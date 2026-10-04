import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import type {Robot,Playback,Neutral} from './types';
import {sampleFrame} from './state.js';

interface Instance {group:THREE.Group;model:THREE.Group;geoms:Map<number,THREE.Object3D>;payload?:Playback;skeleton:THREE.LineSegments;targetGroup:THREE.Group;markers:THREE.Mesh[];offset:THREE.Vector3;}
export class Stage {
  renderer:THREE.WebGLRenderer;scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(38,1,.01,100);
  controls:OrbitControls;loader=new GLTFLoader();instances=new Map<number,Instance>();cache=new Map<string,Promise<{model:THREE.Group;neutral:Neutral}>>();
  mesh=true;skeleton=false;targets=false;wireframe=false;autoRotate=true;private last=performance.now();
  readonly anchors=[new THREE.Vector3(-1.6,0,0),new THREE.Vector3(0,0,0),new THREE.Vector3(1.6,0,0)];
  constructor(public host:HTMLElement,public onInteract:()=>void){
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;this.renderer.toneMapping=THREE.ACESFilmicToneMapping;this.renderer.toneMappingExposure=1.15;
    this.renderer.shadowMap.enabled=true;this.renderer.shadowMap.type=THREE.PCFSoftShadowMap;host.append(this.renderer.domElement);
    this.scene.background=new THREE.Color('#0c0c0d');this.scene.fog=new THREE.Fog('#0c0c0d',15,38);
    this.camera.position.set(1.3,2.8,9.8);this.controls=new OrbitControls(this.camera,this.renderer.domElement);
    this.controls.target.set(0,.85,0);this.controls.enableDamping=true;this.controls.dampingFactor=.06;this.controls.minDistance=2;this.controls.maxDistance=25;
    this.controls.maxPolarAngle=Math.PI*.48;this.controls.addEventListener('start',()=>{this.autoRotate=false;this.onInteract();});
    this.scene.add(new THREE.HemisphereLight(0xeee7dd,0x242120,2.6));
    const key=new THREE.DirectionalLight(0xfff5e6,4);key.position.set(-3,7,4);key.castShadow=true;key.shadow.mapSize.set(2048,2048);
    Object.assign(key.shadow.camera,{left:-9,right:9,top:7,bottom:-7,near:.1,far:25});key.shadow.normalBias=.025;this.scene.add(key);
    const fill=new THREE.DirectionalLight(0xd7dce8,2);fill.position.set(4,3,-4);this.scene.add(fill);
    const floor=new THREE.Mesh(new THREE.PlaneGeometry(150,150),new THREE.MeshStandardMaterial({color:0x161617,roughness:.95,metalness:.05}));
    floor.rotation.x=-Math.PI/2;floor.position.y=-.006;floor.receiveShadow=true;this.scene.add(floor);
    const grid=new THREE.GridHelper(40,80,0x48403a,0x282628);grid.material.transparent=true;grid.material.opacity=.46;this.scene.add(grid);
    this.anchors.forEach((p,i)=>{
      const ring=new THREE.Mesh(new THREE.RingGeometry(.7,.714,96),new THREE.MeshBasicMaterial({color:i===0?0xff8906:0x675447,transparent:true,opacity:.75,side:THREE.DoubleSide}));
      ring.rotation.x=-Math.PI/2;ring.position.copy(p).add(new THREE.Vector3(0,.006,0));this.scene.add(ring);
      const dot=new THREE.Mesh(new THREE.CircleGeometry(.022,16),new THREE.MeshBasicMaterial({color:0xff8906}));dot.rotation.x=-Math.PI/2;dot.position.copy(p).add(new THREE.Vector3(0,.007,0));this.scene.add(dot);
    });
    new ResizeObserver(()=>this.resize()).observe(host);this.resize();
  }
  resize(){const w=this.host.clientWidth,h=this.host.clientHeight;this.renderer.setSize(w,h);this.camera.aspect=w/h;this.camera.zoom=w>1300?1:.8;this.camera.setViewOffset(w,h,w>1600?-237:w>760?-218:-95,0,w,h);this.camera.updateProjectionMatrix();}
  async asset(robot:Robot){
    if(!robot.model_url||!robot.neutral_url)throw new Error('这个模型缺少可预览的网格资源');
    if(!this.cache.has(robot.id))this.cache.set(robot.id,Promise.all([this.loader.loadAsync(robot.model_url),fetch(robot.neutral_url).then(r=>r.json())]).then(([g,n])=>({model:g.scene,neutral:n})));
    return this.cache.get(robot.id)!;
  }
  async load(index:number,robot:Robot,payload:Playback,stillCurrent:()=>boolean){
    const asset=await this.asset(robot);if(!stillCurrent())return;
    const model=asset.model.clone(true);const geoms=new Map<number,THREE.Object3D>();
    model.traverse(obj=>{if(obj.name.startsWith('geom_'))geoms.set(Number(obj.name.slice(5)),obj);if(obj instanceof THREE.Mesh){obj.castShadow=true;obj.receiveShadow=true;const material=(obj.material as THREE.MeshStandardMaterial).clone();material.roughness=.55;material.metalness=.18;obj.material=material;}});
    const skeleton=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0xffaf61,depthTest:false,transparent:true,opacity:.8}));skeleton.renderOrder=2;
    const targetGroup=new THREE.Group();const markers:THREE.Mesh[]=[];
    const count=payload.targets[0]?.length||0;
    if(count<=6)for(let i=0;i<count;i++){
      const color=[0xff982e,0xe9b77e,0xa6cfe6,0xd3b1df][i%4];const marker=new THREE.Mesh(new THREE.SphereGeometry(.018,12,8),new THREE.MeshBasicMaterial({color,depthTest:false}));targetGroup.add(marker);markers.push(marker);
      const points=payload.targets.map(f=>new THREE.Vector3(...f[i] as [number,number,number]));
      const trail=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color,transparent:true,opacity:.35}));targetGroup.add(trail);
    }
    const origin=payload.skeleton[0]?.[0]||payload.positions[0]?.[0]||[0,0,0];const offset=new THREE.Vector3(this.anchors[index].x-origin[0],0,-origin[2]);
    const group=new THREE.Group();group.position.copy(offset);group.add(model,skeleton,targetGroup);
    this.remove(index);this.scene.add(group);this.instances.set(index,{group,model,geoms,payload,skeleton,targetGroup,markers,offset});this.pose(index,0);this.visibility();
  }
  remove(index:number){const old=this.instances.get(index);if(!old)return;this.scene.remove(old.group);
    old.group.traverse(o=>{if(o instanceof THREE.Mesh){(o.material as THREE.Material).dispose();}else if(o instanceof THREE.Line){o.geometry.dispose();(o.material as THREE.Material).dispose();}});this.instances.delete(index);
  }
  pose(index:number,time:number){const v=this.instances.get(index);if(!v?.payload)return;const p=v.payload;const f=sampleFrame(time,p.fps,p.frames);
    p.geom_ids.forEach((id,g)=>{const obj=v.geoms.get(id);if(obj){obj.position.fromArray(p.positions[f][g]);obj.quaternion.fromArray(p.quaternions[f][g]);}});
    const joints=p.skeleton[f];const segments:number[]=[];
    if(joints)joints.forEach((point,j)=>{const parent=p.parents[j];if(parent>=0&&parent!==j&&joints[parent])segments.push(...point,...joints[parent]);});
    v.skeleton.geometry.setAttribute('position',new THREE.Float32BufferAttribute(segments,3));
    v.markers.forEach((m,i)=>m.position.fromArray(p.targets[f][i]));
  }
  visibility(){this.instances.forEach(v=>{v.model.visible=this.mesh;v.skeleton.visible=this.skeleton;v.targetGroup.visible=this.targets;
    v.model.traverse(o=>{if(o instanceof THREE.Mesh)(o.material as THREE.MeshStandardMaterial).wireframe=this.wireframe;});});}
  reset(){this.camera.position.set(1.3,2.8,9.8);this.controls.target.set(0,.85,0);this.controls.update();}
  frame(time:number){const now=performance.now();const dt=Math.min((now-this.last)/1000,.1);this.last=now;
    if(this.autoRotate){const relative=this.camera.position.clone().sub(this.controls.target);relative.applyAxisAngle(new THREE.Vector3(0,1,0),dt*.09);this.camera.position.copy(relative.add(this.controls.target));}
    this.controls.update();this.instances.forEach((_,i)=>this.pose(i,time));this.renderer.render(this.scene,this.camera);
  }
  screenAnchor(i:number){const v=this.anchors[i].clone().add(new THREE.Vector3(0,.17,0)).project(this.camera);return{x:(v.x+1)/2*this.host.clientWidth,y:(1-v.y)/2*this.host.clientHeight};}
  pick(x:number,y:number):number|null{const rect=this.renderer.domElement.getBoundingClientRect(),ray=new THREE.Raycaster();ray.setFromCamera(new THREE.Vector2((x-rect.left)/rect.width*2-1,-(y-rect.top)/rect.height*2+1),this.camera);const hit=ray.intersectObjects(Array.from(this.instances.values()).map(v=>v.model),true)[0];if(!hit)return null;let o:THREE.Object3D|null=hit.object;while(o){for(const [i,v] of this.instances)if(o===v.group)return i;o=o.parent;}return null;}
  async thumbnail(robot:Robot):Promise<string>{
    const a=await this.asset(robot);const model=a.model.clone(true);model.traverse(o=>{const m=/^geom_(\d+)$/.exec(o.name);if(m){const i=a.neutral.geom_ids.indexOf(Number(m[1]));if(i>=0){o.position.fromArray(a.neutral.positions[i]);o.quaternion.fromArray(a.neutral.quaternions[i]);}}});
    const scene=new THREE.Scene();scene.add(model,new THREE.HemisphereLight(0xffffff,0x34302d,3));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(3,5,5);scene.add(light);
    const box=new THREE.Box3().setFromObject(model),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
    const camera=new THREE.PerspectiveCamera(33,1,.01,100);camera.position.copy(center).add(new THREE.Vector3(1.1,.3,1.8).multiplyScalar(Math.max(size.y,size.x)*1.05));camera.lookAt(center);
    const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,preserveDrawingBuffer:true});renderer.setSize(144,144);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.render(scene,camera);const url=renderer.domElement.toDataURL();renderer.dispose();return url;
  }
}
