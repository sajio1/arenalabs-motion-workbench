"""URDF assets are local only; loading does not certify pipeline readiness."""
from pathlib import Path,PurePosixPath
import json,stat,zipfile
import xml.etree.ElementTree as ET

MAX_BYTES=150*1024*1024

def unpack_robot(data,filename,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    if len(data)>MAX_BYTES:raise ValueError('Upload exceeds 150 MB')
    if filename.lower().endswith('.zip'):
        import io
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if len(archive.infolist())>3000 or sum(i.file_size for i in archive.infolist())>MAX_BYTES:raise ValueError('Robot package is too large')
            for item in archive.infolist():
                name=PurePosixPath(item.filename.replace('\\','/'))
                if name.is_absolute() or '..' in name.parts or ':' in str(name) or stat.S_ISLNK(item.external_attr>>16):raise ValueError('Unsafe archive path')
                dest=(folder/str(name)).resolve()
                if not dest.is_relative_to(folder.resolve()):raise ValueError('Unsafe archive path')
                if item.is_dir():dest.mkdir(parents=True,exist_ok=True)
                else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(archive.read(item))
    elif filename.lower().endswith('.urdf'):(folder/'robot.urdf').write_bytes(data)
    else:raise ValueError('Upload a .urdf or .zip robot package')
    urdfs=list(folder.rglob('*.urdf'))
    if len(urdfs)!=1:raise ValueError('Package must contain exactly one URDF')
    urdf=urdfs[0];root=ET.parse(urdf).getroot()
    if root.tag!='robot':raise ValueError('Expected URDF robot root')
    links=[e.attrib['name'] for e in root.findall('link')];joints=root.findall('joint')
    if len(links)!=len(set(links)):raise ValueError('Duplicate link names')
    missing=[]
    for mesh in root.findall('.//mesh'):
        raw=mesh.get('filename','');relative=raw.removeprefix('package://').removeprefix('file://')
        relative=relative.replace('\\','/')
        if Path(relative).is_absolute() or ':' in relative:raise ValueError('Mesh references must stay inside the robot package')
        choices=[urdf.parent/relative,folder/relative]
        # package://name/... often points to the root of this uploaded package.
        if raw.startswith('package://') and '/' in relative:
            package,tail=relative.split('/',1)
            choices.extend([folder/tail,*[p/tail for p in folder.rglob(package) if p.is_dir()]])
        choices=[p.resolve() for p in choices]
        if not any(p.is_relative_to(folder.resolve()) for p in choices):raise ValueError('Mesh references must stay inside the robot package')
        found=next((p.resolve() for p in choices if p.is_file() and p.resolve().is_relative_to(folder.resolve())),None)
        if found:mesh.set('filename',str(found))
        else:missing.append(raw)
    mujoco=root.find('mujoco')
    if mujoco is None:mujoco=ET.SubElement(root,'mujoco')
    compiler=mujoco.find('compiler')
    if compiler is None:compiler=ET.SubElement(mujoco,'compiler')
    compiler.attrib.pop('meshdir',None)
    compiler.set('strippath','false');compiler.set('discardvisual','false');compiler.set('fusestatic','false')
    normalized=folder/'resolved.urdf';ET.ElementTree(root).write(normalized,encoding='utf-8',xml_declaration=True)
    report=dict(links=len(links),joints=len(joints),missing_meshes=missing,mimic_joints=[j.get('name') for j in joints if j.find('mimic') is not None],
        conversion_ready=False,status='needs_metadata',message='Preview only. Conversion requires semantic endpoints, palm/TCP frames and a mechanical descriptor; closed chains require explicit constraints.')
    if missing:report.update(status='missing_assets',message='Meshes missing. Upload a ZIP containing the URDF and meshes.')
    return normalized,root.get('name','Uploaded robot'),report
