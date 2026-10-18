import * as THREE from 'three';

/** Reusable world-space spheres and translucent cylinders. No line-width driver dependency. */
export class TraceGlyph extends THREE.Group {
  dots:THREE.InstancedMesh;links:THREE.InstancedMesh;instanceTransform=new THREE.Matrix4();linkQuaternion=new THREE.Quaternion();linkScale=new THREE.Vector3();axisY=new THREE.Vector3(0,1,0);
  constructor(public capacity:number,color:number,radius=.005,thickness=.0025,opacity=.45,overlay=false){
    super();const material=new THREE.MeshStandardMaterial({color,transparent:true,opacity,depthWrite:false,depthTest:!overlay,roughness:1});
    this.dots=new THREE.InstancedMesh(new THREE.SphereGeometry(radius,10,7),material,capacity);this.links=new THREE.InstancedMesh(new THREE.CylinderGeometry(thickness,thickness,1,8),material,Math.max(1,capacity-1));this.add(this.dots,this.links);this.dots.frustumCulled=this.links.frustumCulled=false;this.dots.instanceMatrix.setUsage(THREE.DynamicDrawUsage);this.links.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  }
  update(points:number[][],edges?:[number,number][]){
    this.dots.count=Math.min(this.capacity,points.length);for(let i=0;i<this.dots.count;i++){this.instanceTransform.makeTranslation(...points[i] as [number,number,number]);this.dots.setMatrixAt(i,this.instanceTransform);}
    const pairs=edges||points.slice(1).map((_,i)=>[i,i+1] as [number,number]);this.links.count=Math.min(this.capacity-1,pairs.length);const a=new THREE.Vector3(),b=new THREE.Vector3(),delta=new THREE.Vector3(),mid=new THREE.Vector3();
    for(let i=0;i<this.links.count;i++){a.fromArray(points[pairs[i][0]]);b.fromArray(points[pairs[i][1]]);delta.subVectors(b,a);const length=delta.length();mid.addVectors(a,b).multiplyScalar(.5);this.linkQuaternion.setFromUnitVectors(this.axisY,length>1e-8?delta.divideScalar(length):this.axisY);this.instanceTransform.compose(mid,this.linkQuaternion,this.linkScale.set(1,length,1));this.links.setMatrixAt(i,this.instanceTransform);}
    this.dots.instanceMatrix.needsUpdate=this.links.instanceMatrix.needsUpdate=true;
  }
  dispose(){this.dots.geometry.dispose();this.links.geometry.dispose();(this.dots.material as THREE.Material).dispose();}
}
