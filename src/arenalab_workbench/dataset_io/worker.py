"""CPU export entry; consumes explicit artifact paths supplied by the API."""
from pathlib import Path
import sys
from .storage import read,write
from .export import export_dataset,collect_raw_sources,collect_video_sources

if __name__=='__main__':
    request=Path(sys.argv[1]);job=read(request)
    raw=job.get('raw_sources')
    if raw is None:raw=collect_raw_sources(job['scene_id'],read(job['scene_path']),job['robot_id'])
    result=export_dataset(job['scene_path'],job['robot_id'],request.parent/'dataset',raw,job.get('video_sources'))
    write(request.parent/'result.json',result)
