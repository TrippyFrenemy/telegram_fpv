import hashlib, io
from datetime import datetime
from pathlib import Path
from typing import Optional
from src.config import settings
from src.store.s3 import client as s3
from src.store.fs import root as fs_root

PREFIX = "telegram"

class StoredObject:
    def __init__(self, sha256: str, relpath: str):
        self.sha256 = sha256
        self.relpath = relpath


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_path(channel: str | int, when: datetime, sha: str, ext: str) -> str:
    y = when.strftime("%Y")
    m = when.strftime("%m")
    ch = str(channel)
    return f"{PREFIX}/{ch}/{y}/{m}/{sha[:8]}.{ext}"


def put(data: bytes, channel: str | int, when: datetime, ext: str) -> StoredObject:
    sha = sha256_bytes(data)
    rel = build_path(channel, when, sha, ext)
    if settings.storage_backend == "s3":
        s3.put_object(settings.s3_bucket, rel, io.BytesIO(data), length=len(data))
    else:
        dst = fs_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    return StoredObject(sha, rel)
