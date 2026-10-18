"""Read-only stage/queue audit. Never launches inference or changes caches."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def summarize(root, job_ids=None):
    groups=defaultdict(list)
    for path in (root/"scheduler/events").glob("*.json"):
        try:
            event=json.loads(path.read_text())
            if job_ids is not None and event.get("job") not in job_ids:continue
            if event.get("state")!="completed" or not event.get("finished_at"):continue
            label=event["label"]
            if label.startswith("SAM:"):label="SAM category worker"
            groups[label].append((event["finished_at"]-event["started_at"],event["started_at"]-event["queued_at"]))
        except (OSError,ValueError,KeyError):continue
    return {name:dict(samples=len(rows),execution_median_s=median(r[0] for r in rows),
        execution_max_s=max(r[0] for r in rows),queue_median_s=median(r[1] for r in rows))
        for name,rows in sorted(groups.items())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--jobs",type=Path,help="JSON array of job id strings; omit for all retained history")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    ids=set(json.loads(args.jobs.read_text())) if args.jobs else None
    result=dict(scope="selected job ids" if ids is not None else "all retained scheduler history",
        notes=["Execution includes subprocess startup, model loading and CPU work inside a GPU lease.",
               "Queue time is reported separately; cache hits without leases are absent.",
               "Stages can overlap; do not sum medians into end-to-end latency."],
        stages=summarize(args.root,ids))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
