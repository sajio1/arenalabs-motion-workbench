/** Bounded RGB-D prefetch with seek priority and cancellation of obsolete work. */
export class DepthStream {
 constructor(count,load,{concurrency=6,capacity=72,lookahead=45,maxBytes=256*1024*1024,resumeFrames=18}={}){
  this.count=count;this.load=load;this.concurrency=concurrency;this.capacity=capacity;
  this.lookahead=Math.min(lookahead,capacity-1);this.frames=new Map();this.requests=new Map();
  this.errors=new Map();this.target=0;this.disposed=false;this.maxBytes=maxBytes;this.bytes=0;this.sizes=new Map();
  this.resumeFrames=Math.min(resumeFrames,this.lookahead+1);this.primed=false;this.frameBytes=1;
 }
 horizon(){return Math.min(this.lookahead,Math.max(0,Math.floor(this.maxBytes/this.frameBytes)-1));}
 ahead(frame){return (frame-this.target+this.count)%this.count;}
 want(index){
  if(this.disposed)return;
  const target=Math.max(0,Math.min(this.count-1,index));
  if(Math.abs(target-this.target)>2)this.primed=false;
  this.target=target;
  for(const [frame,request] of this.requests)if(this.ahead(frame)>this.horizon())request.controller.abort();
  for(const frame of this.errors.keys())if(this.ahead(frame)>this.horizon())this.errors.delete(frame);
  this.pump();return this.frames.get(this.target);
 }
 ready(index,playing=true,speed=1){
  const value=this.want(index);
  if(!playing)return !!value;
  if(!value)this.primed=false;
  if(!this.primed){
   const required=Math.min(this.count-this.target,this.horizon()+1,Math.max(1,Math.ceil(this.resumeFrames*speed)));
   if(this.buffered()<required)return false;
   this.primed=true;
  }
  return !!value;
 }
 buffered(){let n=0;while(this.frames.has(this.target+n))n++;return n;}
 forget(index){this.bytes-=this.sizes.get(index)||0;this.sizes.delete(index);this.frames.delete(index);}
 pump(){
  if(this.disposed)return;
  for(let offset=0;offset<this.count&&offset<=this.horizon();offset++){
   const frame=(this.target+offset)%this.count;
   if(this.requests.size>=this.concurrency)break;
   if(this.frames.has(frame)||this.requests.has(frame)||this.errors.has(frame))continue;
   const controller=new AbortController(),request={controller,promise:undefined};
   request.promise=Promise.resolve().then(()=>this.load(frame,controller.signal)).then(value=>{
    if(this.disposed||controller.signal.aborted)return;
    const byteSize=v=>ArrayBuffer.isView(v)?v.byteLength:v instanceof ArrayBuffer?v.byteLength:v&&typeof v==='object'?Object.values(v).reduce((n,x)=>n+byteSize(x),0):0;
    this.frames.set(frame,value);const size=byteSize(value);this.sizes.set(frame,size);this.bytes+=size;this.frameBytes=Math.max(this.frameBytes,size);
    while(this.frames.size>this.capacity||this.bytes>this.maxBytes&&this.frames.size>1){
     const wanted=i=>this.ahead(i)<=this.horizon();
     const farthest=[...this.frames.keys()].filter(i=>i!==this.target).sort((a,b)=>Number(wanted(a))-Number(wanted(b))||Math.abs(b-this.target)-Math.abs(a-this.target))[0];
     this.forget(farthest);
    }
   }).catch(error=>{if(!this.disposed&&!controller.signal.aborted)this.errors.set(frame,error);})
   .finally(()=>{this.requests.delete(frame);this.pump();});
   this.requests.set(frame,request);
  }
 }
 async initial(index){this.want(index);await this.requests.get(index)?.promise;return this.frames.get(index);}
 retry(){for(const i of this.errors.keys())if(this.ahead(i)<=this.horizon())this.errors.delete(i);this.pump();}
 dispose(){this.disposed=true;for(const request of this.requests.values())request.controller.abort();this.frames.clear();this.sizes.clear();this.bytes=0;}
}
