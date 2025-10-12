from minio import Minio
from src.config import settings


_client = Minio(
    endpoint=settings.s3_endpoint.replace("http://", "").replace("https://", ""),
    access_key=settings.s3_access_key,
    secret_key=settings.s3_secret_key,
    secure=settings.s3_secure,
)


def ensure_bucket(name: str):
    if not _client.bucket_exists(name):
        _client.make_bucket(name)
    return _client

client = ensure_bucket(settings.s3_bucket)
