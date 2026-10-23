"""Durable facade identities, idempotency and event history.

The legacy schedulers still own computation. A dispatch whose outcome was lost
is marked unknown instead of being blindly resubmitted after a process crash.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import time
import uuid

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

class Conflict(Exception):
    pass

class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, key TEXT NOT NULL,
                    request TEXT NOT NULL, status TEXT NOT NULL, backend_id TEXT,
                    result TEXT NOT NULL DEFAULT '{}', error TEXT,
                    created REAL NOT NULL, updated REAL NOT NULL,
                    UNIQUE(owner,key));
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT NOT NULL,
                    created REAL NOT NULL, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, path TEXT NOT NULL,
                    sha256 TEXT NOT NULL, size INTEGER NOT NULL, mtime INTEGER NOT NULL,
                    kind TEXT NOT NULL, UNIQUE(owner,path,sha256));
                CREATE TABLE IF NOT EXISTS batches (
                    owner TEXT NOT NULL, key TEXT NOT NULL, request TEXT NOT NULL,
                    PRIMARY KEY(owner,key));
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def create(self, owner, key, request):
        payload = canonical(request)
        now = time.time()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT * FROM jobs WHERE owner=? AND key=?', (owner,key)).fetchone()
            if old:
                if old['request'] != payload:
                    raise Conflict('Idempotency-Key already used with different inputs')
                return dict(old), False
            jid = uuid.uuid4().hex
            db.execute('INSERT INTO jobs(id,owner,key,request,status,created,updated) VALUES(?,?,?,?,?,?,?)',
                       (jid,owner,key,payload,'dispatching',now,now))
            db.execute('INSERT INTO events(job,created,value) VALUES(?,?,?)',
                       (jid,now,canonical({'status':'dispatching'})))
            return dict(db.execute('SELECT * FROM jobs WHERE id=?',(jid,)).fetchone()), True

    def get(self, ident, owner):
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=? AND owner=?',(ident,owner)).fetchone()
            return dict(row) if row else None

    def claim_batch(self, owner, key, request):
        payload=canonical(request)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT request FROM batches WHERE owner=? AND key=?',(owner,key)).fetchone()
            if old and old['request']!=payload:raise Conflict('Batch key already used with different inputs or item order')
            db.execute('INSERT OR IGNORE INTO batches VALUES(?,?,?)',(owner,key,payload))

    def list(self, owner, limit=50, offset=0):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM jobs WHERE owner=? ORDER BY created DESC,id DESC LIMIT ? OFFSET ?', (owner,limit,offset))]

    def update(self, ident, *, status, result=None, backend_id=None, error=None):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT * FROM jobs WHERE id=?',(ident,)).fetchone()
            if not old: raise KeyError(ident)
            # Do not overwrite a terminal result with an older concurrent poll.
            terminal = {'succeeded','failed','cancelled','interrupted','outcome_unknown'}
            if old['status'] in terminal: return dict(old)
            value = canonical(result if result is not None else json.loads(old['result']))
            error_value = canonical(error) if error else None
            backend_id = backend_id or old['backend_id']
            if (old['status'],old['result'],old['error'],old['backend_id']) == (status,value,error_value,backend_id):
                return dict(old)
            now = time.time()
            db.execute('UPDATE jobs SET status=?,result=?,backend_id=?,error=?,updated=? WHERE id=?',
                       (status,value,backend_id,error_value,now,ident))
            db.execute('INSERT INTO events(job,created,value) VALUES(?,?,?)',
                       (ident,now,canonical({'status':status,'result':json.loads(value),'error':error})))
            return dict(db.execute('SELECT * FROM jobs WHERE id=?',(ident,)).fetchone())

    def events(self, ident, after=0):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM events WHERE job=? AND seq>? ORDER BY seq LIMIT 100', (ident,after))]

    def artifact(self, owner, path, kind):
        path = Path(path)
        before = path.stat()
        digest = hashlib.sha256()
        # Keep revisions beside their source: existing workspace-marker access
        # checks still apply, but later worker writes cannot change these bytes.
        revisions = path.parent / '.artifact-revisions'
        revisions.mkdir(exist_ok=True)
        temporary = revisions / (uuid.uuid4().hex + '.tmp')
        try:
            with path.open('rb') as stream, temporary.open('xb') as target:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    digest.update(block)
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
            after = path.stat()
            if (before.st_size,before.st_mtime_ns,before.st_ino) != (after.st_size,after.st_mtime_ns,after.st_ino):
                raise Conflict('Artifact changed while indexing; retry when generation completes')
            sha = digest.hexdigest()
            snapshot = revisions / sha / path.name
            snapshot.parent.mkdir(exist_ok=True)
            existing_sha = hashlib.sha256()
            if snapshot.is_file():
                with snapshot.open('rb') as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):
                        existing_sha.update(block)
            if not snapshot.is_file() or existing_sha.hexdigest() != sha:
                os.utime(temporary, ns=(after.st_atime_ns, after.st_mtime_ns))
                temporary.replace(snapshot)
                fd = os.open(snapshot.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            snapshot_stat = snapshot.stat()
        finally:
            temporary.unlink(missing_ok=True)
        ident = hashlib.sha256((owner+'\0'+str(path)+'\0'+sha).encode()).hexdigest()
        with self.connect() as db:
            # Migrate an existing legacy ID while its original bytes are still
            # available. IDs whose old content was already lost remain explicit.
            db.execute('INSERT INTO artifacts VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET path=excluded.path,mtime=excluded.mtime',
                       (ident,owner,str(snapshot),sha,snapshot_stat.st_size,snapshot_stat.st_mtime_ns,kind))
        return self.get_artifact(ident,owner)

    def get_artifact(self, ident, owner):
        with self.connect() as db:
            row=db.execute('SELECT * FROM artifacts WHERE id=? AND owner=?',(ident,owner)).fetchone()
            return dict(row) if row else None
