import hashlib, io
from datetime import datetime
from pathlib import Path
from typing import Optional
from src.config import settings
from src.store.s3 import client as s3
from src.store.fs import root as fs_root

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
    return f"telegram/{ch}/{y}/{m}/{sha}.{ext}"


def build_dataset_path(accept: int, sha: str, ext: str)->str:
    assert accept in (0, 1)
    return f"dataset/{accept}/{sha}.{ext}"


def put(data: bytes, channel: str | int, when: datetime, ext: str) -> StoredObject:
    sha = sha256_bytes(data)
    rel = build_path(channel, when, sha, ext)
    if settings.storage_backend == "s3":
        try:
            s3.stat_object(settings.s3_bucket, rel)
            exists = True
        except Exception:
            exists = False
        if not exists:
            s3.put_object(settings.s3_bucket, rel, io.BytesIO(data), length=len(data))
    else:
        dst = fs_root / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
    return StoredObject(sha, rel)


def put_dataset(data: bytes, accept: int, ext: str) -> StoredObject:
    sha = sha256_bytes(data)
    rel = build_dataset_path(accept, sha, ext)
    if settings.storage_backend == "s3":
        try:
            s3.stat_object(settings.s3_bucket, rel)
            exists = True
        except Exception:
            exists = False
        if not exists:
            s3.put_object(settings.s3_bucket, rel, io.BytesIO(data), length=len(data))
    else:
        dst = fs_root / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
    return StoredObject(sha, rel)

def get(relpath: str) -> Optional[bytes]:
    if settings.storage_backend == "s3":
        try:
            obj = s3.get_object(settings.s3_bucket, relpath)
            return obj.read()
        except Exception:
            return None
    else:
        dst = fs_root / relpath
        if dst.exists():
            return dst.read_bytes()
        else:
            return None

def delete(relpath: str) -> None:
    if settings.storage_backend == "s3":
        try:
            s3.remove_object(settings.s3_bucket, relpath)
        except Exception:
            pass
    else:
        dst = fs_root / relpath
        if dst.exists():
            dst.unlink()
