import gzip
import arenalab_workbench.scene_geometry as geometry


def test_preview_cache_bounds_memory_and_keeps_geometry_revisions_separate(monkeypatch):
    calls=[]
    def packet(folder,index,rgb):
        calls.append((folder,index,rgb))
        return bytes([index])*8
    monkeypatch.setattr(geometry,'frame_packet',packet)
    cache=geometry.PreviewPackets(maximum_bytes=16)
    assert cache.get('revision-a',0,'jpeg',False)==bytes(8)
    cache.get('revision-a',1,'jpeg',False)
    cache.get('revision-a',0,'jpeg',False)
    cache.get('revision-a',2,'jpeg',False)
    assert len(calls)==3 and cache.bytes==16
    cache.get('revision-a',1,'jpeg',False)
    assert len(calls)==4
    cache.get('revision-b',1,'jpeg',False)
    assert len(calls)==5 and cache.bytes<=16


def test_preview_cache_preserves_exact_wire_payload_and_bypasses_oversized_packets(monkeypatch):
    data=b'exact-depth-mask-and-human-layer'*10
    monkeypatch.setattr(geometry,'frame_packet',lambda *args:data)
    cache=geometry.PreviewPackets(maximum_bytes=64)
    assert cache.get('a',0,'jpeg',False)==data
    assert cache.bytes==0
    encoded=cache.get('a',0,'jpeg',True)
    assert gzip.decompress(encoded)==data
    assert cache.bytes==len(encoded)
