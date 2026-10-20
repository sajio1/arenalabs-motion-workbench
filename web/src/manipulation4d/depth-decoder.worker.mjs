import {decodePreviewPacket} from './depth-packet.mjs';

self.onmessage=async({data:{id,buffer}})=>{
 try{
  const value=await decodePreviewPacket(buffer);
  const arrays=[value.xyz,value.colors,value.labels,value.human?.xyz,value.human?.colors,value.human?.labels].filter(Boolean);
  self.postMessage({id,value},arrays.map(a=>a.buffer));
 }catch(error){self.postMessage({id,error:error instanceof Error?error.message:String(error)});}
};
