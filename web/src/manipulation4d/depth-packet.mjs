/** Decode a single RGB-D observation without a neural decoder or stored XYZ. */
export function decodeDepthPacket(buffer,decodedRGB) {
 const length=new DataView(buffer).getUint32(0,true);
 if(length%4||length+4>buffer.byteLength)throw new Error('Malformed depth packet header');
 const header=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,length)));
 const [h,w]=header.shape,n=h*w,offset=4+length;
 const rgbBytes=header.rgb_bytes??n*3,jpeg=header.rgb_encoding==='jpeg',labelBytes=header.label_bytes??1,humanBytes=header.human_depth_bytes??0;
 if(![1,2].includes(labelBytes)||!Number.isSafeInteger(n)||n<1||![0,n*2].includes(humanBytes)||buffer.byteLength!==offset+n*(2+labelBytes)+rgbBytes+humanBytes)throw new Error('Depth packet coverage mismatch');
 if(jpeg&&(!decodedRGB||decodedRGB.length!==n*4))throw new Error('Decoded RGB preview unavailable');
 const depth=new Uint16Array(buffer,offset,n),rgb=jpeg?decodedRGB:new Uint8Array(buffer,offset+n*2,n*3),labels=new DataView(buffer,offset+n*2+rgbBytes,n*labelBytes),channels=jpeg?4:3;
 const k=header.intrinsic,t=header.depth_to_display??header.camera_to_world;
 // Invert the full 3x3 calibration; do not assume centered pixels or zero skew.
 const [a,b,c]=k[0],[d,e,f]=k[1],[g,j,i]=k[2];
 const determinant=a*(e*i-f*j)-b*(d*i-f*g)+c*(d*j-e*g);
 if(!Number.isFinite(determinant)||Math.abs(determinant)<1e-15)throw new Error('Invalid depth calibration');
 const inv=[[e*i-f*j,c*j-b*i,b*f-c*e],[f*g-d*i,a*i-c*g,c*d-a*f],[d*j-e*g,b*g-a*j,a*e-b*d]].map(row=>row.map(v=>v/determinant));
 function unproject(values){
  let count=0;for(const value of values)if(value)count++;
  const xyz=new Float32Array(count*3),colors=new Float32Array(count*3),instances=new Float32Array(count);
  let q=0;
  for(let p=0;p<n;p++){
   if(!values[p])continue;const x=p%w,y=Math.floor(p/w),z=values[p]*header.depth_scale;
   const rx=(inv[0][0]*x+inv[0][1]*y+inv[0][2])*z,ry=(inv[1][0]*x+inv[1][1]*y+inv[1][2])*z,rz=(inv[2][0]*x+inv[2][1]*y+inv[2][2])*z;
   for(let axis=0;axis<3;axis++){xyz[q*3+axis]=t[axis][0]*rx+t[axis][1]*ry+t[axis][2]*rz+t[axis][3];colors[q*3+axis]=rgb[p*channels+axis]/255;}
   instances[q]=labelBytes===1?labels.getUint8(p):labels.getUint16(p*2,true);q++;
  }
  return {xyz,colors,labels:instances,count};
 }
 const result=unproject(depth);
 if(humanBytes){
  const view=new DataView(buffer,offset+n*(2+labelBytes)+rgbBytes,humanBytes),values=new Uint16Array(n);
  for(let p=0;p<n;p++)values[p]=view.getUint16(p*2,true);
  result.human=unproject(values);
 }
 return {...result,sourceFrame:header.source_frame};
}

/** JPEG is only a browser colour preview; depth and labels stay lossless. */
export async function decodePreviewPacket(buffer){
 const length=new DataView(buffer).getUint32(0,true);
 const header=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,length)));
 if(header.rgb_encoding!=='jpeg')return decodeDepthPacket(buffer);
 const [h,w]=header.shape,start=4+length+h*w*2;
 const image=await createImageBitmap(new Blob([new Uint8Array(buffer,start,header.rgb_bytes)],{type:'image/jpeg'}));
 if(image.width!==w||image.height!==h){image.close();throw new Error('RGB/depth dimensions differ');}
 const canvas=new OffscreenCanvas(w,h),ctx=canvas.getContext('2d');
 if(!ctx){image.close();throw new Error('RGB preview decoder unavailable');}
 ctx.drawImage(image,0,0);image.close();
 return decodeDepthPacket(buffer,ctx.getImageData(0,0,w,h).data);
}
