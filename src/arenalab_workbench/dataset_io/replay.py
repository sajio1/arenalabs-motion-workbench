"""Portable USD kinematic replay of the exact exported visual geometry.

Playback mesh poses are global Y-up metres/xyzw, not local joint transforms.
No physics articulation or controller is manufactured from the joint table.
"""
from pathlib import Path
import json
import re
import shutil
import numpy as np

Y_UP_TO_Z_UP = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def replay_arrays(motion):
    positions = np.asarray(motion['positions'], dtype=np.float64)
    quaternions = np.asarray(motion['quaternions'], dtype=np.float64).copy()
    n = int(motion['frames'])
    if float(motion['fps']) != 30 or positions.ndim != 3 or positions.shape[0] != n or positions.shape[2] != 3:
        raise ValueError('Replay requires complete 30 Hz mesh poses')
    if quaternions.shape != positions.shape[:2] + (4,) or not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError('Invalid replay mesh poses')
    norms = np.linalg.norm(quaternions, axis=-1)
    if np.max(np.abs(norms - 1)) > 1e-4:
        raise ValueError('Replay quaternion is not unit length')
    quaternions /= norms[..., None]
    # Equivalent q/-q must interpolate on the same hemisphere.
    for frame in range(1, n):
        flip = np.sum(quaternions[frame - 1] * quaternions[frame], axis=-1) < 0
        quaternions[frame, flip] *= -1
    return positions, quaternions


def _tuple(values):
    return '(' + ', '.join(format(float(v), '.12g') for v in values) + ')'


def build_replay(model_path, motion, output):
    """Write a self-contained, asset-path-free USD animation and its contract."""
    import trimesh
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    positions, quaternions = replay_arrays(motion)
    scene = trimesh.load(model_path, force='scene', process=False)
    nodes = {}
    for name in scene.graph.nodes_geometry:
        match = re.fullmatch(r'geom_(\d+)', name)
        if not match:
            raise ValueError('Robot mesh is missing an indexed playback identity: ' + name)
        index = int(match[1])
        transform, geometry = scene.graph[name]
        if not np.allclose(transform, np.eye(4), atol=1e-8):
            raise ValueError('Expected identity robot asset transforms; refusing ambiguous double transforms')
        nodes[index] = scene.geometry[geometry]
    if set(nodes) != set(range(positions.shape[1])):
        raise ValueError('Robot model and playback mesh identities differ')
    target = output / 'robot.usda'
    with target.open('w', encoding='utf-8', newline='\n') as stream:
        def line(value):
            stream.write(value + '\n')
        line('#usda 1.0')
        line('(defaultPrim = "World"\n metersPerUnit = 1\n upAxis = "Z"\n timeCodesPerSecond = 30\n framesPerSecond = 30\n startTimeCode = 0\n endTimeCode = %d\n)' % (len(positions) - 1))
        line('def Xform "World" {\n def Xform "Robot" {')
        # Right-handed basis: (x,y,z)_display -> (x,-z,y)_Isaac.
        line('  float xformOp:rotateX = 90\n  uniform token[] xformOpOrder = ["xformOp:rotateX"]')
        line('  custom string replayMode = "kinematic_visual_world_poses"')
        for index, mesh in sorted(nodes.items()):
            line('  def Xform "geom_%d" {' % index)
            line('   double3 xformOp:translate.timeSamples = {')
            for frame, pose in enumerate(positions[:, index]):
                line('    %d: %s,' % (frame, _tuple(pose)))
            line('   }\n   quatd xformOp:orient.timeSamples = {')
            for frame, quat in enumerate(quaternions[:, index]):
                line('    %d: %s,' % (frame, _tuple(quat[[3, 0, 1, 2]])))
            line('   }\n   uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]')
            line('   def Mesh "visual" {')
            line('    point3f[] points = [' + ', '.join(_tuple(v) for v in mesh.vertices) + ']')
            line('    int[] faceVertexCounts = [' + ', '.join(['3'] * len(mesh.faces)) + ']')
            line('    int[] faceVertexIndices = [' + ', '.join(str(int(i)) for i in mesh.faces.ravel()) + ']')
            line('    normal3f[] normals = [' + ', '.join(_tuple(v) for v in mesh.vertex_normals) + '] (interpolation = "vertex")')
            line('    uniform token subdivisionScheme = "none"\n    uniform bool doubleSided = true')
            material = getattr(mesh.visual, 'material', None)
            color = getattr(material, 'baseColorFactor', None)
            color = np.asarray(color if color is not None else [215, 220, 210, 255], dtype=float)[:3]
            if color.max() > 1: color /= 255
            line('    color3f[] primvars:displayColor = [' + _tuple(color) + ']')
            line('   }\n  }')
        line(' }\n}')
    contract = dict(schema='arenalabs.isaac-replay.v1', stage='robot.usda', fps=30, frames=len(positions), meshes=len(nodes),
                    mode='kinematic_visual_world_poses', physics_articulation=False, units='m', world_up='Z', handedness='right',
                    source_world_up='Y', source_quaternion_order='xyzw', usd_quaternion_order='wxyz',
                    source_to_isaac_rotation=Y_UP_TO_Z_UP.tolist(),
                    pose_semantics='Each geom has a global source-world pose. Apply the basis change once at Robot, never as joint-local transforms.',
                    joint_table_semantics='Named position references; NOT a simulator qpos or actuator vector. Passive/coupled joints and end-effector joints may be present.',
                    start_time_s=0, last_frame_time_s=(len(positions)-1)/30)
    (output/'replay.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    shutil.copyfile(Path(__file__).with_name('isaac_replay.py'), output/'isaac_replay.py')
    (output/'README.md').write_text('''# Isaac Sim replay\n\nOpen `robot.usda` in Isaac Sim and press Play. All meshes, mounted hands/grippers, floating-base motion and the 30 Hz timeline are included. No private paths, weights, URDF imports or network assets are required.\n\nStandalone (Isaac Sim 5.1):\n\n```sh\n/path/to/isaac-sim/python.sh isaac_replay.py --stage robot.usda\n# Headless validation and rendered frames:\n/path/to/isaac-sim/python.sh isaac_replay.py --stage robot.usda --headless --validate --output ./validation\n```\n\nThis is exact kinematic visual playback, without a PhysX articulation or controller. It validates replay, not dynamic stability. Do not feed LeRobot `observation.state`/`action` directly to simulator qpos: these are named joint-position references, not a native DOF vector. `action[t]` is the next-frame reference; replay `observation.state[t]` at timestamp[t].\n\nThe original GLB/playback and raw model results are retained one directory above. `replay.json` declares metres, the one-time Y-up to Z-up conversion, quaternion ordering and world-pose semantics. USD contains global geometry poses under a single fixed basis transform; never apply these again as local joint rotations.\n''', encoding='utf-8')
    return contract
