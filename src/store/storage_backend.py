from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass
from datetime import datetime
import io
import shutil


@dataclass
class StorageObject:
    """Represents a storage object with metadata"""
    object_name: str
    size: int
    last_modified: Optional[datetime] = None


@dataclass
class StoredObject:
    """Represents a stored file with SHA256 hash"""
    sha256: str
    relpath: str


class StorageBackend(ABC):
    """Abstract storage interface"""
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if object exists at path"""
        pass
    
    @abstractmethod
    def put(self, path: str, data: bytes) -> None:
        """Put bytes data to storage"""
        pass
    
    @abstractmethod
    def get(self, path: str) -> Optional[bytes]:
        """Get bytes data from storage"""
        pass
    
    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete object from storage"""
        pass
    
    @abstractmethod
    def list_objects(self, prefix: str = "", recursive: bool = False) -> Iterator[StorageObject]:
        """List objects with given prefix"""
        pass
    
    @abstractmethod
    def fget_object(self, remote_path: str, local_path: str) -> None:
        """Download object to local file"""
        pass
    
    @abstractmethod
    def fput_object(self, remote_path: str, local_path: str) -> None:
        """Upload local file to storage"""
        pass
    
    @abstractmethod
    def stat_object(self, path: str) -> Optional[StorageObject]:
        """Get object metadata"""
        pass
    
    @abstractmethod
    def download_to_file(self, remote_path: str, local_path: str) -> None:
        """Download object to local file (alias for fget_object)"""
        pass
    
    @abstractmethod
    def copy(self, src_path: str, dst_path: str) -> None:
        """Copy object within storage"""
        pass


class S3Backend(StorageBackend):
    """S3 storage implementation (MinIO/AWS S3)"""
    
    def __init__(self, bucket: str):
        from src.store.s3 import client as s3_client
        self.bucket = bucket
        self.client = s3_client
    
    def exists(self, path: str) -> bool:
        try:
            self.client.stat_object(self.bucket, path)
            return True
        except Exception:
            return False
    
    def put(self, path: str, data: bytes) -> None:
        if not self.exists(path):
            self.client.put_object(
                self.bucket, 
                path, 
                io.BytesIO(data), 
                length=len(data)
            )
    
    def get(self, path: str) -> Optional[bytes]:
        try:
            obj = self.client.get_object(self.bucket, path)
            return obj.read()
        except Exception:
            return None
    
    def delete(self, path: str) -> None:
        try:
            self.client.remove_object(self.bucket, path)
        except Exception:
            pass
    
    def list_objects(self, prefix: str = "", recursive: bool = False) -> Iterator[StorageObject]:
        """List objects from S3"""
        try:
            objects = self.client.list_objects(
                self.bucket, 
                prefix=prefix, 
                recursive=recursive
            )
            for obj in objects:
                yield StorageObject(
                    object_name=obj.object_name,
                    size=obj.size,
                    last_modified=obj.last_modified
                )
        except Exception:
            return
    
    def fget_object(self, remote_path: str, local_path: str) -> None:
        """Download from S3 to local file"""
        self.client.fget_object(self.bucket, remote_path, local_path)
    
    def fput_object(self, remote_path: str, local_path: str) -> None:
        """Upload local file to S3"""
        self.client.fput_object(self.bucket, remote_path, local_path)
    
    def stat_object(self, path: str) -> Optional[StorageObject]:
        """Get S3 object metadata"""
        try:
            stat = self.client.stat_object(self.bucket, path)
            return StorageObject(
                object_name=path,
                size=stat.size,
                last_modified=stat.last_modified
            )
        except Exception:
            return None
    
    def download_to_file(self, remote_path: str, local_path: str) -> None:
        """Download from S3 to local file (alias for fget_object)"""
        self.fget_object(remote_path, local_path)
    
    def copy(self, src_path: str, dst_path: str) -> None:
        """Copy object within S3 bucket"""
        from minio.commonconfig import CopySource
        self.client.copy_object(
            self.bucket,
            dst_path,
            CopySource(self.bucket, src_path)
        )


class FileSystemBackend(StorageBackend):
    """Local filesystem storage implementation"""
    
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve storage path to filesystem path"""
        return self.root / path
    
    def exists(self, path: str) -> bool:
        return self._resolve_path(path).exists()
    
    def put(self, path: str, data: bytes) -> None:
        dst = self._resolve_path(path)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
    
    def get(self, path: str) -> Optional[bytes]:
        dst = self._resolve_path(path)
        return dst.read_bytes() if dst.exists() else None
    
    def delete(self, path: str) -> None:
        dst = self._resolve_path(path)
        if dst.exists():
            dst.unlink()
    
    def list_objects(self, prefix: str = "", recursive: bool = False) -> Iterator[StorageObject]:
        """List objects from filesystem"""
        prefix_path = self.root / prefix if prefix else self.root
        
        if not prefix_path.exists():
            return
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in prefix_path.glob(pattern):
            if file_path.is_file():
                relative = file_path.relative_to(self.root)
                stat = file_path.stat()
                yield StorageObject(
                    object_name=str(relative).replace("\\", "/"),  # Normalize paths
                    size=stat.st_size,
                    last_modified=datetime.fromtimestamp(stat.st_mtime)
                )
    
    def fget_object(self, remote_path: str, local_path: str) -> None:
        """Copy from storage to local file"""
        src = self._resolve_path(remote_path)
        if not src.exists():
            raise FileNotFoundError(f"Object not found: {remote_path}")
        shutil.copy2(src, local_path)
    
    def fput_object(self, remote_path: str, local_path: str) -> None:
        """Copy local file to storage"""
        dst = self._resolve_path(remote_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst)
    
    def stat_object(self, path: str) -> Optional[StorageObject]:
        """Get filesystem object metadata"""
        file_path = self._resolve_path(path)
        if not file_path.exists():
            return None
        
        stat = file_path.stat()
        return StorageObject(
            object_name=path,
            size=stat.st_size,
            last_modified=datetime.fromtimestamp(stat.st_mtime)
        )
    
    def download_to_file(self, remote_path: str, local_path: str) -> None:
        """Copy from storage to local file (alias for fget_object)"""
        self.fget_object(remote_path, local_path)
    
    def copy(self, src_path: str, dst_path: str) -> None:
        """Copy file within storage"""
        src = self._resolve_path(src_path)
        dst = self._resolve_path(dst_path)
        
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src_path}")
        
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def get_storage_backend() -> StorageBackend:
    """
    Factory function for storage backend.
    Returns appropriate backend based on settings.
    """
    from src.config import settings
    
    if settings.storage_backend == "s3":
        return S3Backend(settings.s3_bucket)
    else:
        return FileSystemBackend(Path(settings.fs_root))
    