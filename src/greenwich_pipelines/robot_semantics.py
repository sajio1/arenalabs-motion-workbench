"""Shared four-endpoint resolution from a robot's physical tree, never its ID."""
import re
import numpy as np
from alphamotion.engine.spatial import rest_positions

ROLES=('left_wrist','right_wrist','left_ankle','right_ankle')

def shoulder_landmarks(spec,wrists):
    chains=[]
    for wrist in wrists:
        chain=[];j=int(wrist)
        while spec.parents[j]>=0:
            chain.append(j);j=int(spec.parents[j])
        chain.append(j);chains.append(chain)
    common=set(chains[0]) & set(chains[1])
    if not common:raise ValueError('Wrist chains have no common trunk')
    result=[]
    for chain in chains:
        named=[j for j in chain if 'shoulder' in spec.joint_names[j].lower()]
        if named:result.append(named[-1]);continue
        branch=next((j for j in chain if int(spec.parents[j]) in common),None)
        if branch is None:raise ValueError('Cannot resolve independent arm branches')
        result.append(branch)
    return result

def resolve_endpoints(spec,explicit=None):
    if explicit:
        ids=[spec.joint_names.index(explicit[r]) for r in ROLES]
        if len(set(ids))!=4:raise ValueError('Four distinct endpoint links required')
        return ids
    def depth(i):
        count=0
        while spec.parents[i]>=0:i=int(spec.parents[i]);count+=1
        return count
    result=[]
    for role in ROLES:
        side,part=role.split('_');terms=('tcp','end_effector','wrist','hand') if part=='wrist' else ('ankle','foot')
        candidates=[]
        for i,n in enumerate(spec.joint_names):
            name=n.lower()
            sided=side in name or re.search(r'(?:^|_)'+side[0]+r'(?:_|$)',name)
            if not sided or any(t in name for t in ('finger','thumb','knuckle')):continue
            match=next((k for k,t in enumerate(terms) if t in name),None)
            if match is not None:candidates.append((-match,depth(i),i))
        if not candidates:raise ValueError(f'Cannot identify {role}. Supply endpoint link names; finger links are not wrist endpoints.')
        result.append(max(candidates)[2])
    if len(set(result))!=4:raise ValueError('Four distinct endpoint links required')
    p=rest_positions(spec)
    if min(p[result[0],1],p[result[1],1]) <= max(p[result[2],1],p[result[3],1]):
        raise ValueError('Endpoint topology is not a supported upright humanoid')
    return result
