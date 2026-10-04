"""Inference configuration only; no old app/server machine paths."""
import os
from types import SimpleNamespace
import torch
CONFIG=SimpleNamespace(device=os.environ.get('ALPHAMOTION_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu'), greenwich_model=os.environ.get('ALPHAMOTION_MODEL','prime'), hf_repo=os.environ.get('ALPHAMOTION_HF_REPO','lloydlei/Greenwich'))
