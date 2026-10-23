import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import ts from 'typescript';

// Minimal event/DOM surface: exercise the actual selector without a browser/GPU.
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.style={};this.handlers={};this.checked=false;this.disabled=false;this.textContent='';}
 setAttribute(name,value){this[name]=value;}
 append(...items){this.children.push(...items);items.forEach(item=>item.parent=this);}
 addEventListener(name,fn){this.handlers[name]=fn;}
 remove(){this.parent.children=this.parent.children.filter(child=>child!==this);}
 showModal(){this.open=true;}
 close(){this.open=false;this.handlers.close?.();}
 click(){if(!this.disabled)this.onclick?.();}
 all(tag){return this.children.flatMap(child=>[...(child.tag===tag?[child]:[]),...child.all(tag)]);}
}
async function setup(t,available){
 const previous=globalThis.document;globalThis.document={body:new Element('body'),createElement:tag=>new Element(tag)};
 t.after(()=>{if(previous===undefined)delete globalThis.document;else globalThis.document=previous;});
 const source=await readFile(new URL('../src/output-selector.ts',import.meta.url),'utf8');
 const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
 const {chooseOutputModalities}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
 const result=chooseOutputModalities({outputs:{available}},'Target robot');
 return {result,dialog:document.body.children[0]};
}
const choices=['trajectory','point_cloud','contact','inpaint_fov'];

test('motion-only source disables CV outputs and Select all never selects unavailable choices',async t=>{
 const {result,dialog}=await setup(t,choices.map(id=>({id,available:id==='trajectory',reason:'No registered RGB'})));
 const inputs=dialog.all('input');assert.deepEqual(inputs.map(i=>[i.name,i.disabled]),[['trajectory',false],['point_cloud',true],['contact',true],['inpaint_fov',true]]);
 assert.equal(dialog.all('small').length,3);
 inputs[0].checked=false;inputs[0].onchange();const confirm=dialog.all('button').find(b=>b.textContent==='Continue');assert.equal(confirm.disabled,true);
 dialog.all('button').find(b=>b.textContent==='Select all available').click();assert.equal(confirm.disabled,false);confirm.click();
 assert.deepEqual(await result,['trajectory']);assert.equal(document.body.children.length,0);
});

test('individual output selection stays exact; no dependency outputs are added by UI',async t=>{
 const {result,dialog}=await setup(t,choices.map(id=>({id,available:true})));
 for(const input of dialog.all('input')){input.checked=input.name==='inpaint_fov';input.onchange();}
 dialog.all('button').find(b=>b.textContent==='Continue').click();assert.deepEqual(await result,['inpaint_fov']);
});

test('cancel closes selection before a request can be submitted',async t=>{
 const {result,dialog}=await setup(t,[{id:'trajectory',available:true}]);
 const rejected=assert.rejects(result,/cancelled before submission/);dialog.all('button').find(b=>b.textContent==='Cancel').click();await rejected;
 assert.equal(document.body.children.length,0);
});
