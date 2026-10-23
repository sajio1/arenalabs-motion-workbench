import type {RunPlan,OutputModality} from './platform-api';

const labels:Record<OutputModality,string>={trajectory:'Robot trajectory',point_cloud:'Point cloud',contact:'Contact estimates',inpaint_fov:'Inpainted first-person video'};

/** Availability comes from the registered source API, never a preview video. */
export function chooseOutputModalities(plan:RunPlan,label:string):Promise<OutputModality[]>{
 return new Promise((resolve,reject)=>{
  if(!plan.outputs?.available?.length){reject(new Error('Output availability is not available. Refresh the page and try again.'));return;}
  const dialog=document.createElement('dialog');dialog.setAttribute('aria-labelledby','output-selector-title');
  dialog.style.cssText='max-width:560px;max-height:85vh;overflow:auto;padding:24px;border:1px solid #56634d;border-radius:14px;background:#17201b;color:#edf2e8;box-shadow:0 20px 80px #0009;font:15px/1.5 system-ui';
  const title=document.createElement('h2');title.id='output-selector-title';title.textContent='Choose outputs';
  const intro=document.createElement('p');intro.textContent=`Select the results you need for ${label}. Unavailable outputs are disabled for this recording.`;
  const all=document.createElement('button');all.type='button';all.textContent='Select all available';
  const list=document.createElement('div');list.style.cssText='display:grid;gap:12px;margin:16px 0';
  const checks:{input:HTMLInputElement,id:OutputModality}[]=[];
  for(const choice of plan.outputs.available){
   const row=document.createElement('label');row.style.cssText='display:grid;grid-template-columns:20px 1fr;gap:10px;align-items:start';
   const input=document.createElement('input');input.type='checkbox';input.name=choice.id;input.disabled=!choice.available;input.checked=choice.available;
   const text=document.createElement('span');const name=document.createElement('strong');name.textContent=labels[choice.id];text.append(name);
   if(!choice.available){row.style.opacity='.55';const reason=document.createElement('small');reason.style.display='block';reason.textContent=choice.reason||'Not supported by this recording';text.append(reason);}
   row.append(input,text);list.append(row);checks.push({input,id:choice.id});
  }
  const note=document.createElement('p');note.textContent='First-person video also needs robot motion and scene reconstruction internally. Contact estimates are inferred from source video; they are not measured force.';
  const status=document.createElement('p');status.setAttribute('role','status');
  const actions=document.createElement('div');actions.style.cssText='display:flex;gap:12px;margin-top:20px';
  const confirm=document.createElement('button');confirm.type='button';confirm.textContent='Continue';
  const cancel=document.createElement('button');cancel.type='button';cancel.textContent='Cancel';
  for(const button of [all,confirm,cancel])button.style.cssText='padding:10px 14px;border:1px solid #8a9c7b;border-radius:8px;background:#29392d;color:inherit;cursor:pointer';
  const update=()=>{const count=checks.filter(c=>c.input.checked&&!c.input.disabled).length;confirm.disabled=count===0;status.textContent=count?`${count} output${count===1?'':'s'} selected`:'Select at least one output.';};
  checks.forEach(c=>c.input.onchange=update);all.onclick=()=>{checks.forEach(c=>c.input.checked=!c.input.disabled);update();};
  let accepted=false;
  confirm.onclick=()=>{const selected=checks.filter(c=>c.input.checked&&!c.input.disabled).map(c=>c.id);if(!selected.length)return;accepted=true;resolve(selected);dialog.close();};
  cancel.onclick=()=>dialog.close();
  dialog.addEventListener('close',()=>{dialog.remove();if(!accepted)reject(new Error('Generation cancelled before submission.'));},{once:true});
  actions.append(confirm,cancel);dialog.append(title,intro,all,list,note,status,actions);document.body.append(dialog);update();dialog.showModal();
 });
}
