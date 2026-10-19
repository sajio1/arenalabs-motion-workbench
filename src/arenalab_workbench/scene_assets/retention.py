"""Explicit expiry sweep for owned delivery folders; canonical assets are ineligible."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil


def expire(root, now=None, delete=False):
    root=Path(root).resolve(strict=True)
    now=now or datetime.now(timezone.utc)
    candidates=[]
    # Only immediate child directories with our delivery marker may be removed.
    # No recursive discovery of arbitrary paths or permanent package manifests.
    for folder in root.iterdir():
        if not folder.is_dir() or folder.is_symlink():
            continue
        resolved=folder.resolve()
        if resolved.parent!=root or (folder/"manifest.json").exists():
            continue
        if any(p.is_symlink() for p in folder.rglob("*")):
            continue
        marker=folder/"delivery.json"
        if not marker.is_file():
            continue
        data=json.loads(marker.read_text())
        if data.get("owned_temporary_delivery") is not True:
            continue
        deadline=datetime.fromisoformat(data["expires_at"])
        if deadline.tzinfo is None:
            raise ValueError("Expiry timestamp must include timezone")
        if deadline<=now:
            candidates.append(str(resolved))
            if delete:
                shutil.rmtree(resolved)
    return candidates
