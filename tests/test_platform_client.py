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
