import hashlib
import httpx
import pytest
from arena_api.client import ArenaClient

@pytest.mark.parametrize('valid',[True,False])
def test_download_integrity_and_no_partial_publication(tmp_path,valid):
    data=b'example-complete-archive'
    def handler(request):
        if request.url.path.endswith('/download'):return httpx.Response(200,content=data)
        return httpx.Response(200,json={'workflow':'export','status':'succeeded','result':{'bytes':len(data),'sha256':hashlib.sha256(data if valid else b'bad').hexdigest()}})
    with ArenaClient('http://example.test',transport=httpx.MockTransport(handler)) as client:
        target=tmp_path/'export.zip'
        if valid:
            assert client.download('job',target).read_bytes()==data
            with pytest.raises(FileExistsError):client.download('job',target)
        else:
            with pytest.raises(ValueError,match='SHA256'):client.download('job',target)
            assert not target.exists()
        assert not list(tmp_path.glob('*.partial'))


def test_plan_run_and_batch_preserve_explicit_scale_and_keys():
    requests=[]
    def handler(request):
        import json
        requests.append((request.url.path,json.loads(request.content),request.headers.get('Idempotency-Key')))
        return httpx.Response(200,json={'executable':True})
    body={'workflow':'locomotion','recording_id':'body','robot_id':'robot','trajectory_scale':'match_robot'}
    with ArenaClient('http://example.test',workspace_token='owner',transport=httpx.MockTransport(handler)) as client:
        client.plan(**body);client.run(idempotency_key='retained',**body)
        client.batch([body],idempotency_key='batch')
    assert requests==[('/api/v1/plans',body,None),('/api/v1/runs',body,'retained'),('/api/v1/batches',{'items':[body]},'batch')]


def test_cli_explicit_scale_reaches_the_same_sdk(monkeypatch,capsys):
    import sys
    import arena_api.__main__ as cli
    seen=[]
    class Client:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def run(self,**spec):seen.append(spec);return {'id':'one'}
    monkeypatch.setattr(cli,'ArenaClient',Client)
    monkeypatch.setenv('ARENA_WORKSPACE_TOKEN','test-workspace')
    monkeypatch.setattr(sys,'argv',['arena-api','run','--workflow','manipulation','--recording','clip','--robot','r','--hand','none','--trajectory-scale','match_robot','--key','same-key'])
    cli.main()
    assert seen==[dict(workflow='manipulation',recording_id='clip',robot_id='r',hand_id='none',trajectory_scale='match_robot',idempotency_key='same-key')]
    assert 'one' in capsys.readouterr().out
