"""Immutable, losslessly compressed display models, scoped to their asset owner."""
from pathlib import Path
from functools import lru_cache
import gzip
import hashlib
import json
import os
import shutil
import uuid


def owner_of(path,root):
    for parent in Path(path).resolve().parents:
        marker=parent/'.workspace.json'
        if marker.is_file():return json.loads(marker.read_text())['owner']
        if parent==Path(root).resolve():break
    return None


def publish_model(data,root,owner=None):
    root=Path(root);digest=hashlib.sha256(data).hexdigest()
    folder=root/'manipulation4d/robot-models'/(owner or 'public')/digest
    folder.mkdir(parents=True,exist_ok=True)
    from .gpu_scheduler import resource_lock,write
    with resource_lock('display-model:'+str(folder)):
        if owner:write(folder/'.workspace.json',{'owner':owner})
        path=folder/'model.glb'
        if not path.exists():
            temporary=folder/(uuid.uuid4().hex+'.tmp');temporary.write_bytes(data);temporary.replace(path)
        compressed=folder/'model.glb.gz'
        if not compressed.exists():
            temporary=folder/(uuid.uuid4().hex+'.tmp');temporary.write_bytes(gzip.compress(data,compresslevel=6,mtime=0));temporary.replace(compressed)
    return '/assets/'+path.relative_to(root).as_posix()


def export_model(scene,path,root):
    """Preserve the per-result model for portable export; viewers share its URL."""
    path=Path(path);data=scene.export(file_type='glb')
    url=publish_model(data,root,owner_of(path,root))
    cached=Path(root)/url.removeprefix('/assets/')
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:os.link(cached,temporary)
    except OSError:shutil.copyfile(cached,temporary)
    temporary.replace(path)
    return url


@lru_cache(maxsize=2)
def _index(path,modified,size):return json.loads(Path(path).read_text())


def rewrite_scene_models(scene,root):
    path=Path(root)/'manipulation4d/.model-index.json'
    if not path.exists():return scene
    stat=path.stat();index=_index(str(path),stat.st_mtime_ns,stat.st_size)
    for row in [*scene.get('robots',[]),scene.get('source_robot') or {}]:
        url=row.get('model_url','');entry=index.get(url)
        if entry:
            original=Path(root)/url.removeprefix('/assets/')
            try:s=original.stat()
            except OSError:continue
            if [s.st_size,s.st_mtime_ns]==entry['stat']:row['model_url']=entry['url']
    return scene


def prepare_existing(root):
    root=Path(root);index={};raw=compressed=0
    # Never rewrite historical scenes or trajectories. Only public display assets
    # are prebuilt; private models keep their existing permission boundary.
    for path in (root/'manipulation4d').rglob('model.glb'):
        if 'robot-models' in path.parts or owner_of(path,root):continue
        data=path.read_bytes();url=publish_model(data,root)
        stat=path.stat();index['/assets/'+path.relative_to(root).as_posix()]={'stat':[stat.st_size,stat.st_mtime_ns],'url':url}
        raw+=len(data);compressed+=(root/url.removeprefix('/assets/')).with_suffix('.glb.gz').stat().st_size
    from .gpu_scheduler import write
    write(root/'manipulation4d/.model-index.json',index)
    return {'models':len(index),'unique_models':len({v['url'] for v in index.values()}),'raw_bytes':raw,'gzip_bytes':compressed}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--store',type=Path,required=True)
    print(json.dumps(prepare_existing(parser.parse_args().store)))
