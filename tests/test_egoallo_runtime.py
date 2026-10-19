"""A scheduler-selected device must survive CUDA/JAX environment preparation."""
import importlib.util
from pathlib import Path


def test_explicit_gpu_lease_is_not_replaced_by_parent_device(tmp_path, monkeypatch):
    source=Path(__file__).parents[1]/"src/arenalab_workbench/egoallo_runtime.py"
    spec=importlib.util.spec_from_file_location("tested_egoallo_runtime",source)
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    (tmp_path/"deps/jax_plugins").mkdir(parents=True)
    monkeypatch.setattr(runtime,"RUNTIME",tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","3")
    base={"CUDA_VISIBLE_DEVICES":"1","ARENA_GPU_LEASE":"selected-card-1"}
    result=runtime.gpu_environment(base)
    assert result["CUDA_VISIBLE_DEVICES"]=="1"
    assert result["ARENA_GPU_LEASE"]=="selected-card-1"
    assert result["JAX_PLATFORMS"]=="cuda"
    assert base=={"CUDA_VISIBLE_DEVICES":"1","ARENA_GPU_LEASE":"selected-card-1"}
