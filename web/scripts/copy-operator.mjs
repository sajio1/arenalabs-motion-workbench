import {cp,mkdir} from 'node:fs/promises';
await mkdir('dist/operator',{recursive:true});
await cp('operator','dist/operator',{recursive:true});
