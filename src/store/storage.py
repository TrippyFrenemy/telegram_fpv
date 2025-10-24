from datetime import datetime
from hashlib import sha256
from typing import Optional

from src.store.storage_backend import get_storage_backend, StoredObject


def sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def build_path(channel: str, when: datetime, sha: str, ext: str) -> str:
    """Build path for telegram downloads"""
    y = when.strftime("%Y")
    m = when.strftime("%m")
    ch = str(channel)
    return f"telegram/{ch}/{y}/{m}/{sha}.{ext}"


def build_dataset_path(accept: int, sha: str, ext: str) -> str:
    """Build path for dataset"""
    assert accept in (0, 1)
    return f"dataset/{accept}/{sha}.{ext}"


def _put_object(data: bytes, relpath: str) -> StoredObject:
    """
    Common logic for putting objects to storage (DRY).
    Returns StoredObject with sha256 and relative path.
    """
    sha = sha256_bytes(data)
    backend = get_storage_backend()
    backend.put(relpath, data)
    return StoredObject(sha, relpath)


def put(data: bytes, channel: str, when: datetime, ext: str) -> StoredObject:
    """Store telegram media"""
    sha = sha256_bytes(data)
    relpath = build_path(channel, when, sha, ext)
    return _put_object(data, relpath)


def put_dataset(data: bytes, accept: int, ext: str) -> StoredObject:
    """Store dataset media"""
    sha = sha256_bytes(data)
    relpath = build_dataset_path(accept, sha, ext)
    return _put_object(data, relpath)


def get(relpath: str) -> Optional[bytes]:
    """Get object from storage"""
    backend = get_storage_backend()
    return backend.get(relpath)


def delete(relpath: str) -> None:
    """Delete object from storage"""
    backend = get_storage_backend()
    backend.delete(relpath)
