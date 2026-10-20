/** Keep image decoding and dense unprojection away from animation and input. */
export class DepthDecoder {
 constructor(){
  this.worker=new Worker(new URL('./depth-decoder.worker.mjs',import.meta.url),{type:'module'});
  this.pending=new Map();this.serial=0;
  this.worker.onmessage=({data})=>{
   const task=this.pending.get(data.id);if(!task)return;
   this.pending.delete(data.id);task.cleanup();
   if(data.error)task.reject(new Error(data.error));else task.resolve(data.value);
  };
  this.worker.onerror=()=>this.dispose(new Error('Point-cloud decoder failed'));
 }
 decode(buffer,signal){
  return new Promise((resolve,reject)=>{
   if(signal.aborted||!this.worker){reject(new DOMException('Cancelled','AbortError'));return;}
   const id=++this.serial;
   const abort=()=>{this.pending.delete(id);signal.removeEventListener('abort',abort);reject(new DOMException('Cancelled','AbortError'));};
   this.pending.set(id,{resolve,reject,cleanup:()=>signal.removeEventListener('abort',abort)});
   signal.addEventListener('abort',abort,{once:true});this.worker.postMessage({id,buffer},[buffer]);
  });
 }
 dispose(error=new DOMException('Cancelled','AbortError')){
  this.worker?.terminate();this.worker=undefined;
  for(const task of this.pending.values()){task.cleanup();task.reject(error);}this.pending.clear();
 }
}
