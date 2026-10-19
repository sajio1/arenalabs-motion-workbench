"""Exercise uploads privately; never add test packages to the public registry."""
from pathlib import Path
import tempfile,io,zipfile
from fastapi import FastAPI
from fastapi.testclient import TestClient
from arenalab_workbench.hand_uploads import attach_hand_routes

def test_hand_uploads():
    URDF=b'<robot name="Test gripper"><link name="palm"><visual><geometry><box size=".04 .04 .02"/></geometry></visual></link><link name="finger"><inertial><mass value=".1"/><inertia ixx=".01" ixy="0" ixz="0" iyy=".01" iyz="0" izz=".01"/></inertial><visual><geometry><box size=".01 .01 .04"/></geometry></visual></link><joint name="close" type="prismatic"><parent link="palm"/><child link="finger"/><axis xyz="1 0 0"/><limit lower="0" upper=".02" effort="1" velocity="1"/></joint></robot>'
    with tempfile.TemporaryDirectory() as root:
     app=FastAPI();attach_hand_routes(app,root);client=TestClient(app)
     result=client.post('/hands',files={'file':('test.urdf',URDF,'application/xml')});assert result.status_code==200,result.text
     row=result.json();assert row['status']=='needs_mapping' and row['supported'] is False and row.get('model_url'),row
     assert client.post('/hands',files={'file':('test.urdf',URDF,'application/xml')}).json()['id']==row['id']
     assert {r['id'] for r in client.get('/hands').json()}=={'none','wuji','robotiq-2f-140',row['id']}
     archive=io.BytesIO()
     with zipfile.ZipFile(archive,'w') as z:z.writestr('../escape.urdf',URDF)
     assert client.post('/hands',files={'file':('bad.zip',archive.getvalue(),'application/zip')}).status_code==400
     assert client.post('/hands',files={'file':('bad.txt',b'x','text/plain')}).status_code==400
     print('Hand upload: valid model export, explicit missing mapping, duplicate reuse, registry, unsafe ZIP and bad extension checks passed')
