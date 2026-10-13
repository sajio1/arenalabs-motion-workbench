import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('organize_library', Path(__file__).parents[1] / 'tools/organize_motion_library.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def package(folder, uri, mesh=b'same mesh', offset='1'):
    folder.mkdir()
    (folder / 'mesh.stl').write_bytes(mesh)
    (folder / 'model.urdf').write_text(f'<robot name="same"><mujoco><compiler meshdir="ignored"/></mujoco><link name="base"><visual><origin xyz="0 0 {offset}"/><geometry><mesh filename="{uri}"/></geometry></visual></link></robot>')
    return folder


def test_identity_ignores_package_paths_but_preserves_physical_differences(tmp_path):
    first = package(tmp_path / 'first', '../meshes/mesh.stl')
    second = package(tmp_path / 'second', 'package://robot/mesh.stl')
    assert module.package_identity(first) == module.package_identity(second)
    changed_mesh = package(tmp_path / 'changed_mesh', 'mesh.stl', mesh=b'different')
    changed_geometry = package(tmp_path / 'changed_geometry', 'mesh.stl', offset='2')
    assert module.package_identity(first) != module.package_identity(changed_mesh)
    assert module.package_identity(first) != module.package_identity(changed_geometry)


def test_missing_mesh_is_not_evidence_for_duplicate(tmp_path):
    folder = package(tmp_path / 'missing', 'not_here.stl')
    assert module.package_identity(folder) is None
