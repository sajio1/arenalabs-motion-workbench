"""Registered download identities retain bytes across mutable worker publications."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

from arena_api.store import Store


def test_identical_reindex_deduplicates_snapshots_under_concurrency(tmp_path):
    store = Store(tmp_path / 'api.sqlite')
    source = tmp_path / 'contact.json'
    source.write_bytes(b'{"likelihood":[null]}')
    with ThreadPoolExecutor(max_workers=6) as workers:
        rows = list(workers.map(lambda _: store.artifact('owner', source, 'contact'), range(12)))
    assert len({row['id'] for row in rows}) == 1
    assert len({row['path'] for row in rows}) == 1
    assert Path(rows[0]['path']).read_bytes() == source.read_bytes()
    assert not list((tmp_path / '.artifact-revisions').glob('*.tmp'))


def test_legacy_id_is_migrated_without_changing_identity(tmp_path):
    store = Store(tmp_path / 'api.sqlite')
    source = tmp_path / 'contact.json'
    original = b'{"likelihood":[null]}'
    source.write_bytes(original)
    sha = hashlib.sha256(original).hexdigest()
    ident = hashlib.sha256(('owner\0' + str(source) + '\0' + sha).encode()).hexdigest()
    stat = source.stat()
    with store.connect() as db:
        db.execute('INSERT INTO artifacts VALUES(?,?,?,?,?,?,?)',
                   (ident, 'owner', str(source), sha, stat.st_size, stat.st_mtime_ns, 'contact'))
    migrated = store.artifact('owner', source, 'contact')
    assert migrated['id'] == ident
    assert migrated['path'] != str(source)
    source.write_bytes(b'{"likelihood":[1]}')
    assert Path(store.get_artifact(ident, 'owner')['path']).read_bytes() == original
    assert store.get_artifact(ident, 'other') is None


def test_same_basename_in_distinct_directories_cannot_alias(tmp_path):
    store = Store(tmp_path / 'api.sqlite')
    records = []
    for name in ('first', 'second'):
        source = tmp_path / name / 'contact.json'
        source.parent.mkdir()
        source.write_bytes(b'identical')
        records.append(store.artifact('owner', source, 'contact'))
    assert records[0]['id'] != records[1]['id']
    assert records[0]['path'] != records[1]['path']
    Path(records[0]['path']).unlink()
    assert Path(records[1]['path']).read_bytes() == b'identical'


def test_reindex_repairs_damaged_snapshot_from_verified_source(tmp_path):
    store = Store(tmp_path / 'api.sqlite')
    source = tmp_path / 'contact.json'
    source.write_bytes(b'original')
    first = store.artifact('owner', source, 'contact')
    Path(first['path']).write_bytes(b'damaged')
    again = store.artifact('owner', source, 'contact')
    assert again['id'] == first['id']
    assert Path(again['path']).read_bytes() == b'original'
    assert again['size'] == len(b'original')
