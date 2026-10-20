/** Bounded RGB-D prefetch with seek priority and cancellation of obsolete work. */
export class DepthStream {
 constructor(count,load,{concurrency=6,capacity=12,lookahead=6}={}){
  this.count=count;this.load=load;this.concurrency=concurrency;this.capacity=capacity;
  this.lookahead=Math.min(lookahead,capacity-1);this.frames=new Map();this.requests=new Map();
  this.errors=new Map();this.target=0;this.disposed=false;
 }
 want(index){
  if(this.disposed)return;
  this.target=Math.max(0,Math.min(this.count-1,index));
  const end=Math.min(this.count-1,this.target+this.lookahead);
  for(const [frame,request] of this.requests)if(frame<this.target||frame>end)request.controller.abort();
  for(const frame of this.errors.keys())if(frame<this.target||frame>end)this.errors.delete(frame);
  this.pump();return this.frames.get(this.target);
 }
 pump(){
  if(this.disposed)return;
  for(let frame=this.target;frame<this.count&&frame<=this.target+this.lookahead;frame++){
   if(this.requests.size>=this.concurrency)break;
   if(this.frames.has(frame)||this.requests.has(frame)||this.errors.has(frame))continue;
   const controller=new AbortController(),request={controller,promise:undefined};
   request.promise=Promise.resolve().then(()=>this.load(frame,controller.signal)).then(value=>{
    if(this.disposed||controller.signal.aborted)return;
    this.frames.set(frame,value);
    while(this.frames.size>this.capacity){
     const wanted=i=>i>=this.target&&i<=this.target+this.lookahead;
     const farthest=[...this.frames.keys()].filter(i=>i!==this.target).sort((a,b)=>Number(wanted(a))-Number(wanted(b))||Math.abs(b-this.target)-Math.abs(a-this.target))[0];
     this.frames.delete(farthest);
    }
   }).catch(error=>{if(!this.disposed&&!controller.signal.aborted)this.errors.set(frame,error);})
   .finally(()=>{this.requests.delete(frame);this.pump();});
   this.requests.set(frame,request);
  }
 }
 async initial(index){this.want(index);await this.requests.get(index)?.promise;return this.frames.get(index);}
 retry(){for(let i=this.target;i<=this.target+this.lookahead;i++)this.errors.delete(i);this.pump();}
 dispose(){this.disposed=true;for(const request of this.requests.values())request.controller.abort();this.frames.clear();}
}
