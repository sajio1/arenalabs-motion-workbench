import test from 'node:test';
import assert from 'node:assert/strict';
import {errorChart} from '../src/tracking.js';

test('small chart retains a single-frame error spike and both hands',()=>{
 const series=Array.from({length:1000},()=>[1,2]);series[501]=[80,2];
 const chart=errorChart(series,10);
 assert.equal(chart.ceiling,80);
 assert.match(chart.paths[0],/130\.39,2\.00/);
 assert.ok(chart.thresholdY>2&&chart.thresholdY<22);
 assert.equal(chart.paths.length,2);
});
