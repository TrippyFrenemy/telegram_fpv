import redis
from src.config import settings

_client = redis.Redis.from_url(settings.redis_dsn, decode_responses=True)
