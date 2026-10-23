"""Replay a named reference on a customer-local Isaac Sim articulation.

Run with Isaac Sim 5.1's Python. This sets articulated states and checks FK;
it does not run a tracking controller or establish dynamic feasibility.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

if __package__:
    from .native_reference_mujoco import load
else:
    from native_reference_mujoco import load


def rotation_xyzw(quaternion):
    q = np.asarray(quaternion)
    x, y, z, w = q.reshape(-1, 4).T
    return np.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ], axis=-1).reshape(*q.shape[:-1], 3, 3)


def fixed_transforms(urdf):
    """Preserve original link frames when Isaac merges fixed URDF links."""
    if urdf is None:
        return {}
    fixed = {}
    for joint in ET.parse(urdf).getroot().findall('joint'):
        if joint.get('type') != 'fixed':
            continue
        origin = joint.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
        rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
        if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite([xyz, rpy]).all():
            raise ValueError('Invalid fixed-joint transform: ' + str(joint.get('name')))
        a, b, c = rpy
        ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
        transform = np.eye(4)
        transform[:3, :3] = [[cc*cb, cc*sb*sa-sc*ca, cc*sb*ca+sc*sa],
                             [sc*cb, sc*sb*sa+cc*ca, sc*sb*ca-cc*sa],
                             [-sb, cb*sa, cb*ca]]
        transform[:3, 3] = xyz
        fixed[joint.find('child').get('link')] = (joint.find('parent').get('link'), transform)
    return fixed


def mimic_relations(urdf):
    """Only use relationships explicitly declared by the customer's model."""
    if urdf is None:
        return {}
    result = {}
    for joint in ET.parse(urdf).getroot().findall('joint'):
        mimic = joint.find('mimic')
        if mimic is None:
            continue
        multiplier, offset = float(mimic.get('multiplier', 1)), float(mimic.get('offset', 0))
        if not np.isfinite([multiplier, offset]).all() or not mimic.get('joint'):
            raise ValueError('Invalid local URDF mimic relationship')
        result[joint.get('name')] = (mimic.get('joint'), multiplier, offset)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument('--urdf', type=Path, help='Matching local URDF with accessible local mesh assets')
    model.add_argument('--usd', type=Path, help='Matching local Z-up, metre-scale articulated USD')
    parser.add_argument('--joint-map', type=Path, help='Optional JSON object: source joint name to Isaac DOF name')
    component_mode = parser.add_mutually_exclusive_group()
    component_mode.add_argument('--with-hands', action='store_true', help='Also apply available hand/gripper reference channels; requires their matching local assembly')
    component_mode.add_argument('--body-only', action='store_false', dest='with_hands', help='Body reference only (default); hand/gripper models are optional')
    parser.set_defaults(with_hands=False)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--validate', action='store_true', help='Check all frames and render timeline seeks, then exit')
    parser.add_argument('--output', type=Path, default=Path('isaac-native-validation'))
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    args.body_only = not args.with_hands
    meta, frames = load(args.bundle)
    if any(joint['type'] not in ('free', 'hinge', 'slide') for joint in meta['native_joints']):
        raise ValueError('This Isaac adapter supports one free root and named scalar hinge/slide joints; ball-joint conversion must be explicit')
    body_names = list(meta['joint_names'])
    extra_names = list(meta.get('additional_joints', {}).get('names', []))
    source_names = body_names + ([] if args.body_only else extra_names)
    reference_units = {joint['name']: joint['unit'] for joint in meta['joints']}
    reference_units.update(zip(extra_names, meta.get('additional_joints', {}).get('units', [])))
    source_position, source_velocity = frames['joint_position'], frames['joint_velocity']
    if extra_names and not args.body_only:
        extra_position = frames['additional_joint_position']
        extra_velocity = np.zeros_like(extra_position)
        if len(extra_position) > 1:
            extra_velocity[:-1] = np.diff(extra_position, axis=0)*30
            extra_velocity[-1] = extra_velocity[-2]
        source_position = np.concatenate([source_position, extra_position], axis=1)
        source_velocity = np.concatenate([source_velocity, extra_velocity], axis=1)
    link_names = list(meta['link_names'])
    selected_model = (args.urdf or args.usd).resolve()
    if not selected_model.is_file():
        raise FileNotFoundError(selected_model)
    mapping = json.loads(args.joint_map.read_text()) if args.joint_map else {}
    if not isinstance(mapping, dict) or set(mapping) - set(body_names + extra_names) or any(not isinstance(v, str) for v in mapping.values()):
        raise ValueError('Joint map must map known source joint names to Isaac DOF names')
    target_names = [mapping.get(name, name) for name in source_names]
    if len(set(target_names)) != len(target_names):
        raise ValueError('Joint map must be one-to-one')
    args.output.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=args.headless, active_gpu=args.gpu, physics_gpu=args.gpu,
                             multi_gpu=False, renderer='RaytracedLighting', width=960, height=720))
    try:
        import omni.usd
        import omni.kit.commands
        import omni.timeline
        import omni.physx
        from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema, Gf
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.simulation_manager import SimulationManager

        world = World(physics_dt=1/30, rendering_dt=1/30, stage_units_in_meters=1, backend='numpy')
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        if args.urdf:
            enable_extension('isaacsim.asset.importer.urdf')
            ok, config = omni.kit.commands.execute('URDFCreateImportConfig')
            if not ok:
                raise RuntimeError('Isaac URDF importer is unavailable')
            config.merge_fixed_joints = True
            config.fix_base = False
            config.import_inertia_tensor = True
            config.distance_scale = 1.
            config.make_default_prim = True
            config.collision_from_visuals = False
            config.set_parse_mimic(False)
            model_usd = args.output.resolve() / 'articulated.usd'
            ok, _ = omni.kit.commands.execute('URDFParseAndImportFile', urdf_path=str(selected_model),
                                             import_config=config, dest_path=str(model_usd), get_articulation_root=True)
            if not ok:
                raise RuntimeError('Isaac could not import the local URDF')
        else:
            model_usd = selected_model
            source_stage = Usd.Stage.Open(str(model_usd))
            if not source_stage or UsdGeom.GetStageUpAxis(source_stage) != UsdGeom.Tokens.z or abs(UsdGeom.GetStageMetersPerUnit(source_stage)-1) > 1e-12:
                raise ValueError('Local USD must explicitly use Z-up and metres; no implicit basis or scale conversion')
            if not source_stage.GetDefaultPrim():
                raise ValueError('Local USD requires a default prim containing one robot articulation')
        assembly = UsdGeom.Xform.Define(stage, '/World/Robot')
        assembly.GetPrim().GetReferences().AddReference(str(model_usd))
        app.update()
        roots = [p for p in Usd.PrimRange(assembly.GetPrim()) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(roots) != 1:
            raise RuntimeError('Expected one articulation, found ' + str(len(roots)))
        root = str(roots[0].GetPath())
        # State replay does not ask drives, gravity, or contact to reproduce the reference.
        # These changes are limited to this isolated in-memory validation stage.
        model_joint_units = {}
        for prim in Usd.PrimRange(assembly.GetPrim()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(False)
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                name = prim.GetName()
                if name in model_joint_units:
                    raise ValueError('Ambiguous local model joint name: ' + name)
                model_joint_units[name] = 'rad' if prim.IsA(UsdPhysics.RevoluteJoint) else 'm'
                drive = UsdPhysics.DriveAPI.Apply(prim, 'angular' if prim.IsA(UsdPhysics.RevoluteJoint) else 'linear')
                drive.CreateStiffnessAttr(0)
                drive.CreateDampingAttr(0)
                drive.CreateMaxForceAttr(0)
        world.reset()
        robot = Articulation(prim_paths_expr=root, name='native_reference_robot')
        robot.initialize()
        engine_names, engine_bodies = list(robot.dof_names), list(robot.body_names)
        if len(set(engine_names)) != len(engine_names):
            raise ValueError('Local articulation has duplicate DOF names')
        missing = set(target_names)-set(engine_names)
        if missing:
            raise RuntimeError('Local assembly is missing reference DOFs: ' + str(sorted(missing)) + '. Supply the matching complete model, or choose --body-only for an explicit partial audit.')
        for source, target in zip(source_names, target_names):
            if model_joint_units.get(target) != reference_units[source]:
                raise ValueError('Local joint type/unit differs from reference: ' + source)
        if not engine_bodies or engine_bodies[0] != meta['root_link']:
            raise RuntimeError('Local articulation base differs from reference root_link: ' + str((engine_bodies[:1], meta['root_link'])))
        resolved = {name: (source_position[:, index], source_velocity[:, index]) for index, name in enumerate(target_names)}
        relations = mimic_relations(args.urdf)
        derived_mimic_names = []

        def resolve(name, visited=()):
            if name in resolved:
                return resolved[name]
            if name not in relations or name in visited:
                raise RuntimeError('Unspecified model DOF or cyclic/unsupported mimic relationship: ' + name)
            parent, multiplier, offset = relations[name]
            position, velocity = resolve(parent, visited+(name,))
            resolved[name] = (position*multiplier+offset, velocity*multiplier)
            derived_mimic_names.append(name)
            return resolved[name]

        if not args.body_only:
            for name in engine_names:
                resolve(name)
            # Independently supplied mimic coordinates must obey the same
            # explicit mechanical relationship, not silently override it.
            for name in set(target_names) & set(relations):
                parent, multiplier, offset = relations[name]
                expected, _ = resolve(parent)
                if not np.allclose(resolved[name][0], expected*multiplier+offset, atol=1e-5, rtol=0):
                    raise ValueError('Reference violates the local URDF mimic relationship for ' + name)
        applied_names = [name for name in engine_names if name in resolved]
        indices = np.array([engine_names.index(name) for name in applied_names])
        reference_position = np.stack([resolved[name][0] for name in applied_names], axis=1)
        reference_velocity = np.stack([resolved[name][1] for name in applied_names], axis=1)
        sim = SimulationManager.get_physics_sim_view()
        view = sim.create_articulation_view(root)
        fixed = fixed_transforms(args.urdf)
        bindings = []
        for name in link_names:
            current, transform, visited = name, np.eye(4), set()
            while current not in engine_bodies:
                if current not in fixed or current in visited:
                    raise RuntimeError('Cannot resolve canonical link ' + name + '. Use its matching URDF to resolve merged fixed-link frames.')
                visited.add(current)
                parent, local = fixed[current]
                transform = local @ transform
                current = parent
            bindings.append((engine_bodies.index(current), transform))
        body_indices = np.array([binding[0] for binding in bindings])
        local_frames = np.stack([binding[1] for binding in bindings])
        root_com_positions, _ = robot.get_body_coms(body_indices=np.array([0]))
        root_com_local = np.asarray(root_com_positions)[0, 0]
        if root_com_local.shape != (3,) or not np.isfinite(root_com_local).all():
            raise RuntimeError('Local articulation has an invalid root centre of mass')
        omni.timeline.get_timeline_interface().pause()

        def apply(frame):
            robot.set_world_poses(frames['root_position_m'][frame:frame+1].astype('f4'),
                                  frames['root_quaternion_wxyz'][frame:frame+1].astype('f4'))
            robot.set_joint_positions(reference_position[frame:frame+1].astype('f4'), joint_indices=indices)
            robot.set_joint_velocities(reference_velocity[frame:frame+1].astype('f4'), joint_indices=indices)
            velocity = np.concatenate([frames['root_linear_velocity_world_m_s'][frame:frame+1],
                                       frames['root_angular_velocity_world_rad_s'][frame:frame+1]], axis=-1)
            # PhysX set_root_velocities uses the root body's centre of mass;
            # the exported linear velocity belongs to the root link origin.
            root_rotation = rotation_xyzw(frames['root_quaternion_wxyz'][frame, [1, 2, 3, 0]])
            velocity[:, :3] += np.cross(velocity[:, 3:], root_rotation @ root_com_local)
            robot.set_velocities(velocity.astype('f4'))
            sim.update_articulations_kinematic()
            actual = np.asarray(view.get_link_transforms())[0, body_indices]
            if not np.isfinite(actual).all():
                raise RuntimeError('Local articulation produced non-finite link transforms')
            rotations = rotation_xyzw(actual[:, 3:])
            positions = actual[:, :3] + np.einsum('lij,lj->li', rotations, local_frames[:, :3, 3])
            return positions, rotations @ local_frames[:, :3, :3], velocity[0]

        max_position = max_rotation = max_joint = 0.
        max_joint_velocity = max_root_velocity = 0.
        started = time.monotonic()
        # A wrong joint axis/offset can coincide at frame zero. Check the full
        # reference before permitting interactive replay as well as validation.
        for frame in range(meta['frames']):
            actual_position, actual_rotation, root_velocity = apply(frame)
            expected = frames['link_pose_world'][frame]
            max_position = max(max_position, float(np.max(np.abs(actual_position-expected[:, :3]))))
            max_rotation = max(max_rotation, float(np.max(np.abs(actual_rotation-rotation_xyzw(expected[:, [4, 5, 6, 3]])))))
            actual_joint = np.asarray(robot.get_joint_positions())[0, indices]
            if not np.isfinite(actual_joint).all():
                raise RuntimeError('Local articulation produced non-finite joint positions')
            max_joint = max(max_joint, float(np.max(np.abs(actual_joint-reference_position[frame]))))
            actual_velocity = np.asarray(robot.get_velocities())[0]
            actual_joint_velocity = np.asarray(robot.get_joint_velocities())[0, indices]
            if not np.allclose(actual_velocity, root_velocity, atol=2e-5, rtol=1e-6) or not np.allclose(actual_joint_velocity, reference_velocity[frame], atol=2e-5, rtol=1e-6):
                raise RuntimeError('Local articulation velocity state differs from the mapped reference')
            max_root_velocity = max(max_root_velocity, float(np.max(np.abs(actual_velocity-root_velocity))))
            max_joint_velocity = max(max_joint_velocity, float(np.max(np.abs(actual_joint_velocity-reference_velocity[frame]))))
        if max_position > 2e-4 or max_rotation > 2e-4 or max_joint > 1e-5:
            raise RuntimeError('Local articulation FK differs from reference: ' + str((max_position, max_rotation, max_joint)))
        apply(0)
        omni.physx.get_physx_interface().update_transformations(False, True, True, False)
        light = UsdLux.DomeLight.Define(stage, '/ReplayLight')
        light.CreateIntensityAttr(500)
        light.CreateColorAttr(Gf.Vec3f(.4, .45, .5))
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default']).ComputeWorldBound(assembly.GetPrim()).ComputeAlignedRange()
        low, high = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        if not np.isfinite([low, high]).all():
            raise RuntimeError('Local robot has no finite visual bounds')
        # Fit the full pose range relative to the moving base: an initial
        # crouched pose otherwise crops the head/feet when the robot stands.
        relative_links = frames['link_pose_world'][..., :3] - frames['root_position_m'][:, None, :]
        low = np.minimum(low-frames['root_position_m'][0], relative_links.min(axis=(0, 1)))
        high = np.maximum(high-frames['root_position_m'][0], relative_links.max(axis=(0, 1)))
        center, size = (low+high)/2, max(float(np.max(high-low))*1.4, .5)
        camera = UsdGeom.Camera.Define(stage, '/ReplayCamera')
        camera_op = camera.AddTransformOp()
        camera.CreateFocalLengthAttr(40)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 1000))

        def point_camera(frame):
            target = center + frames['root_position_m'][frame]
            eye = target + np.array([1.5, 2.1, .7])*size
            camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())

        point_camera(0)
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport:
            viewport.set_active_camera(str(camera.GetPath()))
        if args.validate:
            import omni.replicator.core as rep
            from PIL import Image
            product = rep.create.render_product(str(camera.GetPath()), (960, 720))
            rgb = rep.AnnotatorRegistry.get_annotator('rgb')
            rgb.attach([product])
            captures = []
            for sample, frame in enumerate([0, meta['frames']//2, meta['frames']-1, meta['frames']//4, 0, meta['frames']-1]):
                apply(frame)
                point_camera(frame)
                omni.physx.get_physx_interface().update_transformations(False, True, True, False)
                for _ in range(3):
                    app.update()
                rep.orchestrator.step(rt_subframes=4, delta_time=0., pause_timeline=True)
                pixels = np.asarray(rgb.get_data())
                if pixels.shape != (720, 960, 4) or float(pixels[..., :3].std(axis=(0, 1)).max()) < 1:
                    raise RuntimeError('Isaac rendered an empty frame')
                filename = f'frame-{sample:02d}-{frame:04d}.png'
                Image.fromarray(pixels).save(args.output/filename)
                captures.append(dict(frame=frame, path=filename))
            from importlib.metadata import PackageNotFoundError, version
            try:
                runtime_version = version('isaacsim')
            except PackageNotFoundError:
                runtime_version = 'unavailable (adapter tested with Isaac Sim 5.1.0)'
            report = dict(engine='Isaac Sim', engine_version=runtime_version, mode='articulated_named_state_replay', frames=meta['frames'],
                          joints=len(indices), reference_joints=len(source_names), links=len(bindings), checked_link_poses=meta['frames']*len(bindings),
                          max_link_position_error_m=max_position, max_link_rotation_matrix_error=max_rotation,
                          max_joint_position_error=max_joint, elapsed_s=time.monotonic()-started, renders=captures,
                          max_joint_velocity_error=max_joint_velocity, max_root_com_velocity_error=max_root_velocity,
                          root_velocity_mapping='Exported root-link origin world velocity converted to PhysX root COM velocity: v_com = v_link + omega cross R_link*p_com_local',
                          local_model_sha256=hashlib.sha256(selected_model.read_bytes()).hexdigest(),
                          scope='body_only' if args.body_only else 'all_reference_channels',
                          additional_reference_joints_applied=0 if args.body_only else len(extra_names),
                          omitted_reference_joints=extra_names if args.body_only else [],
                          derived_mimic_joints=derived_mimic_names,
                          unapplied_model_joints=[name for name in engine_names if name not in applied_names],
                          fk_scope='All exported canonical link frames; any additional hand/gripper joint states are checked separately',
                          additional_velocity_semantics='30 Hz forward position difference; final sample repeats previous, one-frame clip zero; not measured',
                          controller_used=False, physics_control_validated=False, passed=True)
            (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
            print('ISAAC_NATIVE_REFERENCE_PASS ' + json.dumps(report), flush=True)
        else:
            frame = 0
            while app.is_running():
                started = time.monotonic()
                apply(frame)
                point_camera(frame)
                omni.physx.get_physx_interface().update_transformations(False, True, True, False)
                app.update()
                frame = (frame+1) % meta['frames']
                time.sleep(max(0, 1/30-(time.monotonic()-started)))
    except BaseException:
        import traceback
        import omni.kit.app
        (args.output/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        traceback.print_exc()
        # Kit's default fast shutdown can exit the process from app.close().
        # Set its return code first so a failed validation cannot exit zero.
        omni.kit.app.get_app().post_quit(1)
        raise
    finally:
        app.close()


if __name__ == '__main__':
    main()
