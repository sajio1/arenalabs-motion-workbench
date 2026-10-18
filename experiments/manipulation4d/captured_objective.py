"""Replay unchanged FP64 refinement objectives with CUDA Graphs."""
import time
import numpy as np
import torch

class CapturedObjective:
    def __init__(self, evaluate, parameters):
        self.parameters=parameters
        self.sizes=[p.numel() for p in parameters]
        self.evaluate=evaluate
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                for p in parameters:p.grad=None
                loss,*_=evaluate();loss.backward()
        torch.cuda.current_stream().wait_stream(stream)
        for p in parameters:p.grad=None
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss,self.terms,*self.results=evaluate()
            self.loss.backward()
            self.packed=torch.cat([self.loss.detach().reshape(1),*[p.grad.flatten() for p in parameters]])
        self.term_names=list(self.terms)
        self.started=time.monotonic()
    def __call__(self,x):
        values=torch.as_tensor(x,device=self.parameters[0].device,dtype=self.parameters[0].dtype)
        offset=0
        with torch.no_grad():
            for p,size in zip(self.parameters,self.sizes):
                p.copy_(values[offset:offset+size].reshape(p.shape));offset+=size
        self.graph.replay()
        result=self.packed.cpu().numpy()
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite objective or gradient')
        return float(result[0]),result[1:].copy()
    def term_values(self):
        values=torch.stack([self.terms[k].detach() for k in self.term_names]).cpu().tolist()
        return dict(zip(self.term_names,values))
