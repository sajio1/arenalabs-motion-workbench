/** Size decisions precede submission; both viewers use the same API preflight. */
import {api,type RunSpec,type RunPlan,type TrajectoryScale} from './platform-api';
import {chooseOutputModalities} from './output-selector';

let dialogQueue:Promise<unknown>=Promise.resolve();
function prompt(plan:RunPlan,label:string):Promise<TrajectoryScale>{
 return new Promise((resolve,reject)=>{
  const dialog=document.createElement('dialog');dialog.setAttribute('aria-labelledby','trajectory-scale-title');
  dialog.style.cssText='max-width:520px;padding:24px;border:1px solid #56634d;border-radius:14px;background:#17201b;color:#edf2e8;box-shadow:0 20px 80px #0009;font:15px/1.5 system-ui';
  const title=document.createElement('h2');title.id='trajectory-scale-title';title.textContent='Choose interaction size';
  const explanation=document.createElement('p');explanation.textContent=`The source body and ${label} differ substantially in size. Choose how to transfer this interaction.`;
  const morphology=plan.policy.morphology,dimensions=document.createElement('p');
  dimensions.textContent=[morphology?.source_height_m!=null?`Source: ${morphology.source_height_m.toFixed(2)} m`:null,
   morphology?.target_height_m!=null?`Robot: ${morphology.target_height_m.toFixed(2)} m`:null,
   morphology?.scale_factor!=null?`Estimated size ratio: ${morphology.scale_factor.toFixed(3)}×`:null].filter(Boolean).join(' · ');
  const note=document.createElement('p');note.textContent='Preserve keeps original distances and object sizes. Match robot scales the body, trajectories and scene together to preserve relative contact; object sizes then differ from the original task.';
  const actions=document.createElement('div');actions.style.cssText='display:flex;gap:10px;flex-wrap:wrap;margin-top:20px';
  let selected=false;
  for(const [value,text] of [['preserve','Preserve original size'],['match_robot','Match robot size']] as const){
   const button=document.createElement('button');button.textContent=text;button.type='button';button.style.cssText='padding:10px 14px;border:1px solid #8a9c7b;border-radius:8px;background:#29392d;color:inherit;cursor:pointer';
   button.onclick=()=>{selected=true;resolve(value);dialog.close();};actions.append(button);
  }
  const cancel=document.createElement('button');cancel.textContent='Cancel';cancel.onclick=()=>dialog.close();actions.append(cancel);
  dialog.append(title,explanation,dimensions,note,actions);document.body.append(dialog);
  dialog.addEventListener('close',()=>{dialog.remove();if(!selected)reject(new Error('Generation cancelled before submission.'));},{once:true});
  dialog.showModal();
 });
}

export async function prepareRun(spec:RunSpec,label:string):Promise<RunSpec>{
 let plan=await api.plan(spec);
 if(spec.workflow!=='export'&&spec.output_modalities==null){
  const choice=dialogQueue.then(()=>chooseOutputModalities(plan,label));dialogQueue=choice.catch(()=>{});
  spec={...spec,output_modalities:await choice};plan=await api.plan(spec);
 }
 if(plan.outputs?.error)throw new Error(plan.outputs.error);
 if(plan.required_choice==='trajectory_scale'||plan.policy.scale_choice_required){
  const choice=dialogQueue.then(()=>prompt(plan,label));dialogQueue=choice.catch(()=>{});
  const selected={...spec,trajectory_scale:await choice};
  const checked=await api.plan(selected);
  if(!checked.executable)throw new Error(checked.outputs?.error||checked.policy.morphology?.reason||'This size choice is not available for the current source.');
  return selected;
 }
 if(!plan.executable)throw new Error(plan.policy.morphology?.reason||'This recording is not ready for generation.');
 return {...spec,trajectory_scale:plan.policy.trajectory_scale??spec.trajectory_scale??'preserve'};
}
