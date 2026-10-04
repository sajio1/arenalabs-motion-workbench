"""Use the same vendor kinematic tree for optimization and mesh playback."""
from alphamotion.embodiment.registry import load, load_from_xml


def load_robot(robot):
    xml = robot.get('model') or robot.get('xml')
    return load_from_xml(xml, robot['body']) if xml else load(robot['body'])
