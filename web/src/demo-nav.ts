export const demoPages=[
  {id:'locomotion',href:'/locomotion',label:'Locomotion',description:'SOMA / SMPL · Full-body motion'},
  {id:'manipulation',href:'/manipulation',label:'Manipulation',description:'Ego motion · 4D workspace'},
  {id:'operator',href:'/operator/',label:'Operator',description:'OpenArm · Simulation playback'},
] as const;
export function demoNavigation(active:string){return `<nav class="arenalab-demo-nav" aria-label="Demos">${demoPages.map(page=>`<a href="${page.href}" ${page.id===active?'aria-current="page"':''}>${page.label}</a>`).join('')}</nav>`;}

export function demoBrand(){return `<a class="arena-brand" href="/locomotion"><img src="/web/arenalab-logo-transparent.png" alt=""/>ArenaLabs</a>`;}
