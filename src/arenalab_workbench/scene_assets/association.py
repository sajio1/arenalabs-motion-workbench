"""Geometry/appearance association of SAM observations into physical tracks.

No category-specific rules, selected frames, robot names or wrist-pose shortcuts.
Ambiguous associations remain separate and are exposed in the audit.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def observations(frame):
    result = []
    for label in np.unique(frame["labels"]):
        if label in (0,65535):
            continue
        mask = (frame["labels"] == label) & frame["valid"]
        points = frame["points"][mask]
        if len(points) < 6:
            continue
        low,high = np.quantile(points,[.05,.95],axis=0)
        rgb = frame["images"][mask].astype(float)/255
        hist = np.concatenate([np.histogram(rgb[:,c],bins=8,range=(0,1))[0] for c in range(3)]).astype(float)
        hist /= max(np.linalg.norm(hist),1e-8)
        result.append(dict(label=int(label),center=np.median(points,axis=0),
                           size=max(float(np.linalg.norm(high-low)),.01),hist=hist,area=int(mask.sum())))
    return result


def associate(store):
    # A whole-clip co-visibility check prevents fragments from being merged
    # when they are actually two neighbouring objects visible together later.
    co_visible=set()
    for index in range(len(store)):
        frame=store.frame(index)
        labels=np.unique(frame["labels"][(frame["labels"]!=65535)&frame["valid"]])
        co_visible.update((int(a),int(b)) for a in labels for b in labels if a and b and a!=b)
    tracks = []
    for index in range(len(store)):
        frame = store.frame(index)
        current = observations(frame)
        timestamp = int(frame["ids"])/30
        # SAM already supplies a temporal identity. Retain it before matching
        # disconnected fragments; otherwise ambiguous neighbouring cups can
        # recursively spawn one new mesh per frame.
        assigned = set()
        updated = set()
        for b,obs in enumerate(current):
            same = [t for t in tracks if obs["label"] in t["source_labels"]]
            if same:
                track = same[-1]
                track["observations"].append(dict(index=index,label=obs["label"],area=obs["area"]))
                track["last"],track["last_time"] = obs,timestamp
                assigned.add(b);updated.add(track["id"])
        active = [t for t in tracks if timestamp-t["last_time"] <= 2.]
        costs = np.full((len(active),len(current)),1e6)
        for a,track in enumerate(active):
            for b,obs in enumerate(current):
                if b in assigned or track["id"] in updated:
                    continue
                if any((label,obs["label"]) in co_visible for label in track["source_labels"]):
                    continue
                dt = max(timestamp-track["last_time"],1/30)
                distance = np.linalg.norm(obs["center"]-track["last"]["center"])
                scale = max(obs["size"],track["last"]["size"])
                appearance = 1-float(obs["hist"]@track["last"]["hist"])
                ratio = abs(np.log(obs["size"]/track["last"]["size"]))
                if distance <= scale*.75 + dt*.5 and appearance < .35 and ratio < 1.1:
                    costs[a,b] = distance/(scale+dt*.5) + appearance + .1*ratio - .1*(obs["label"] in track["source_labels"])
        if costs.size:
            aa,bb = linear_sum_assignment(costs)
            for a,b in zip(aa,bb):
                if costs[a,b] >= 1e5:
                    continue
                # Do not silently swap indistinguishable neighbours.
                alternatives = np.r_[np.delete(costs[a],b),np.delete(costs[:,b],a)]
                if len(alternatives) and np.min(alternatives)-costs[a,b] < .08:
                    active[a]["ambiguous_frames"].append(int(frame["ids"]))
                    continue
                track = active[a]
                track["observations"].append(dict(index=index,label=current[b]["label"],area=current[b]["area"]))
                track["source_labels"].add(current[b]["label"])
                track["last"],track["last_time"] = current[b],timestamp
                assigned.add(b)
        for b,obs in enumerate(current):
            if b not in assigned:
                tracks.append(dict(id=len(tracks)+1,observations=[dict(index=index,label=obs["label"],area=obs["area"])],
                    source_labels={obs["label"]},last=obs,last_time=timestamp,ambiguous_frames=[]))
    for track in tracks:
        # Maximum visible reliable area selects the seed, not an action-specific frame.
        track["keyframe"] = max(track["observations"],key=lambda x:x["area"])["index"]
        track["source_labels"] = sorted(track["source_labels"])
        track.pop("last");track.pop("last_time")
    return tracks
