"""Assemble a portable URDF using physical fixed mount metadata, without solver changes."""
from pathlib import Path
import json,sys,copy,hashlib,shutil,xml.etree.ElementTree as E
root=Path(__import__("os").environ.get("ARENA_EGO_WORKDIR", str(Path(__file__).resolve().parent)));label=sys.argv[1]
row=json.loads((root/(label+'-robot.json')).read_text())[0]
source=Path(row['xml']).with_name('H2.urdf') if label=='h2' else Path(row['xml']).with_name('resolved.urdf')
out=root/(label+'-wuji-description');out.mkdir(exist_ok=True);(out/'meshes').mkdir(exist_ok=True)
robot=E.parse(source).getroot();mounts=json.loads((root/(label+'-wuji-mounts.json')).read_text())
links={x.get('name') for x in robot.findall('link')}
removed={x for x in links if x.lower() in ['left_hand_link','right_hand_link']}
while True:
 new=removed|{j.find('child').get('link') for j in robot.findall('joint') if j.find('parent').get('link') in removed}
 if new==removed:break
 removed=new
for el in list(robot):
 if (el.tag=='link' and el.get('name') in removed) or (el.tag=='joint' and el.find('child').get('link') in removed) or el.tag=='mujoco':robot.remove(el)
def localize(element,base):
 for m in element.iter('mesh'):
  name=m.get('filename');p=Path(name) if Path(name).is_absolute() else (base/name).resolve()
  if not p.exists() and '/robot/' in name.replace('\\','/'):
   p=base/'robot'/name.replace('\\','/').split('/robot/',1)[1]
  if not p.exists():raise FileNotFoundError(p)
  dst=out/'meshes'/(hashlib.sha256(p.read_bytes()).hexdigest()[:16]+p.suffix.lower())
  if not dst.exists():shutil.copy2(p,dst)
  m.set('filename','meshes/'+dst.name)
localize(robot,source.parent)
from scipy.spatial.transform import Rotation
for mount in mounts:
 side=mount['side'];src=root/'wuji/urdf'/(side+'.urdf');hand=E.parse(src).getroot();localize(hand,src.parent)
 for el in hand:
  if el.tag in ['link','joint']:robot.append(copy.deepcopy(el))
 j=E.SubElement(robot,'joint',name=side+'_wuji_mount_joint',type='fixed')
 E.SubElement(j,'parent',link=mount['parent']);E.SubElement(j,'child',link=side+'_palm_link')
 E.SubElement(j,'origin',xyz=' '.join(map(str,mount['origin_m'])),rpy=' '.join(map(str,Rotation.from_matrix(mount['rotation']).as_euler('xyz'))))
robot.set('name',label+'_wuji')
# MuJoCo strips URDF mesh path components by default; supply the localized directory.
ext=E.SubElement(robot,'mujoco');E.SubElement(ext,'compiler',meshdir='meshes',discardvisual='false',balanceinertia='true')
E.indent(robot);E.ElementTree(robot).write(out/'model.urdf',encoding='utf-8',xml_declaration=True)
names=[l.get('name') for l in robot.findall('link')];assert len(names)==len(set(names))
for j in robot.findall('joint'):assert j.find('parent').get('link') in names and j.find('child').get('link') in names
assert sum('finger' in j.get('name') and j.get('type')=='revolute' for j in robot.findall('joint'))==40
shutil.copy2(root/(label+'-wuji-mounts.json'),out/'mounts.json');print('URDF_COMPLETE',label,len(names))
