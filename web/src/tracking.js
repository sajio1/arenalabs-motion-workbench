// Keep extrema when compacting a long trace; a brief spike must stay visible.
export function errorChart(series, threshold, width=260, height=24) {
  const ceiling=Math.max(threshold*1.25,...series.flat(),1);
  const y=value=>height-2-(height-4)*value/ceiling;
  const paths=[0,1].map(hand=>{
    const indices=new Set([0,series.length-1]),bucket=Math.max(1,Math.ceil(series.length/160));
    for(let start=0;start<series.length;start+=bucket){
      let low=start,high=start;
      for(let i=start;i<Math.min(start+bucket,series.length);i++){
        if(series[i][hand]<series[low][hand])low=i;
        if(series[i][hand]>series[high][hand])high=i;
      }
      indices.add(low);indices.add(high);
    }
    return [...indices].sort((a,b)=>a-b).map((i,n)=>`${n?'L':'M'}${(i*width/Math.max(1,series.length-1)).toFixed(2)},${y(series[i][hand]).toFixed(2)}`).join(' ');
  });
  return {paths,thresholdY:y(threshold),ceiling};
}
