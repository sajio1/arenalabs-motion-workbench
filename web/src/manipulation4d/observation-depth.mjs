/** Rasterize the current source observation, including foreground occluders.
 * @param {Float32Array} xyz
 * @param {Uint8Array} labels
 * @param {number[]} view Column-major world-to-source-camera matrix (CV +Z).
 * @param {number[][]} k
 * @param {number} width
 * @param {number} height
 * @param {number} rasterWidth
 */
export function observationDepth(xyz, labels, view, k, width, height, rasterWidth=256) {
 const rasterHeight=Math.round(rasterWidth*height/width), data=new Float32Array(rasterWidth*rasterHeight*4);
 for(let i=0;i<labels.length;i++){
  const x=xyz[i*3],y=xyz[i*3+1],z=xyz[i*3+2];
  const cx=view[0]*x+view[4]*y+view[8]*z+view[12],cy=view[1]*x+view[5]*y+view[9]*z+view[13],cz=view[2]*x+view[6]*y+view[10]*z+view[14];
  if(!Number.isFinite(cx)||!Number.isFinite(cy)||!Number.isFinite(cz)||cz<=.03)continue;
  const u=Math.floor((k[0][0]*cx/cz+k[0][2])*rasterWidth/width),v=Math.floor((k[1][1]*cy/cz+k[1][2])*rasterHeight/height);
  if(u<0||u>=rasterWidth||v<0||v>=rasterHeight)continue;
  // Fill sampling holes conservatively; closest foreground wins at boundaries.
  for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
   const a=u+dx,b=v+dy;if(a<0||b<0||a>=rasterWidth||b>=rasterHeight)continue;
   const j=(b*rasterWidth+a)*4;if(!data[j]||cz<data[j]){data[j]=cz;data[j+1]=labels[i]===0?1:0;}
  }
 }
 return {data,width:rasterWidth,height:rasterHeight};
}
