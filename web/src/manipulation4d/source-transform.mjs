/** Proper column-vector rotation from raw Z-up into the scene's Y-up basis.
 * Legacy manipulation scenes use [x,z,-y]; gripper scenes declare [y,z,x]. */
export function sourceDisplayBasis(value){
 const b=value??[[1,0,0],[0,0,1],[0,-1,0]];
 const invalid=()=>{throw new Error('Invalid source display rotation');};
 if(!Array.isArray(b)||b.length!==3||b.some(row=>!Array.isArray(row)||row.length!==3||!row.every(Number.isFinite)))invalid();
 for(let i=0;i<3;i++)for(let j=0;j<3;j++)if(Math.abs(b[i].reduce((v,x,k)=>v+x*b[j][k],0)-(i===j?1:0))>1e-8)invalid();
 const determinant=b[0][0]*(b[1][1]*b[2][2]-b[1][2]*b[2][1])-b[0][1]*(b[1][0]*b[2][2]-b[1][2]*b[2][0])+b[0][2]*(b[1][0]*b[2][1]-b[1][1]*b[2][0]);
 if(Math.abs(determinant-1)>1e-8||b.some((row,i)=>Math.abs(row[2]-(i===1?1:0))>1e-8))invalid();
 return b.map(row=>[...row]);
}
export function rawSourceToDisplay(point,basis){return basis.map(row=>row.reduce((v,x,i)=>v+x*point[i],0));}
/** Only the source layers move; robot and wrist targets are already transformed. */
export function sourceDisplayTransform(metadata,worldToDisplay){
 const basis=sourceDisplayBasis(worldToDisplay);
 if(!metadata||metadata.applied===true)return {scale:1,offset:[0,0,0]};
 const {scale,anchor_xyz_m:anchor,world_up}=metadata;
 if(world_up!=='z'||!Number.isFinite(scale)||scale<=0||!Array.isArray(anchor)||anchor.length!==3||!anchor.every(Number.isFinite))throw new Error('Invalid source similarity transform');
 const display=rawSourceToDisplay(anchor,basis);
 return {scale,offset:display.map(v=>(1-scale)*v)};
}
export function transformSourcePoint(point,transform){
 return point.map((v,i)=>transform.scale*v+transform.offset[i]);
}
