/** Operator uses the same versioned platform API as the motion workbenches. */
export const operatorApi={
 async catalog(){
  const collection=new URLSearchParams(location.search).get('review')||'default';
  const items=[];let offset=0;
  do{const response=await fetch('/api/v1/simulation-episodes?'+new URLSearchParams({collection,limit:'200',offset:String(offset)}),{cache:'no-store'});
   const value=await response.json();if(!response.ok)throw Error(value.detail||'Motion catalog unavailable');
   items.push(...value.items);offset+=value.items.length;if(offset>=value.total||!value.items.length)break;
  }while(true);return items;
 }
};
