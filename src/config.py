from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """
    Centralized configuration for the video classification system.
    All settings are loaded from environment variables with sensible defaults.
    """
    
    # Telegram API credentials
    tg_api_id: int = Field(alias="TG_API_ID")
    tg_api_hash: str = Field(alias="TG_API_HASH")
    tg_phone_number: str | None = Field(alias="TG_PHONE_NUMBER", default=None)
    tg_session: str = Field(alias="TG_SESSION", default="fpv_session")
    tg_workdir: Path = Field(alias="TG_WORKDIR", default=Path(".tg_sessions"))
    tg_lang: str = Field(alias="TG_LANG", default="uk")
    
    # Telegram rate limiting
    tg_rps: int = Field(alias="TG_RPS", default=20)
    tg_rps_burst: int = Field(alias="TG_RPS_BURST", default=30)
    tg_sleep_jitter_ms: int = Field(alias="TG_SLEEP_JITTER_MS", default=250)
    tg_bot_token: str = Field(alias="TG_BOT_TOKEN", default="")

    # Database configuration
    db_dsn: str = Field(alias="DB_DSN", default="postgresql://user:pass@localhost:5432/fpv")

    # Storage configuration
    storage_backend: str = Field(alias="STORAGE_BACKEND", default="s3")
    s3_endpoint: str | None = Field(alias="S3_ENDPOINT", default="localhost:9000")
    s3_access_key: str | None = Field(alias="S3_ACCESS_KEY", default="minioadmin")
    s3_secret_key: str | None = Field(alias="S3_SECRET_KEY", default="minioadmin")
    s3_bucket: str | None = Field(alias="S3_BUCKET", default="fpv")
    s3_region: str | None = Field(alias="S3_REGION", default="us-east-1")
    s3_secure: bool = Field(alias="S3_SECURE", default=False)
    
    # Redis configuration for worker coordination
    redis_dsn: str = Field(alias="REDIS_DSN", default="redis://localhost:6379/0")

    # Local filesystem fallback (for development)
    fs_root: Path = Field(alias="FS_ROOT", default=Path("./data"))

    # Video filtering parameters
    fpv_min_duration_s: int = Field(alias="FPV_MIN_DURATION_S", default=5)
    fpv_max_duration_s: int = Field(alias="FPV_MAX_DURATION_S", default=1200)
    fpv_keywords: str = Field(alias="FPV_KEYWORDS", default="#fpv,#drone")
    fpv_min_confidence: float = Field(alias="FPV_MIN_CONFIDENCE", default=0.3)

    # Crawler configuration
    crawl_backfill_since: str = Field(alias="CRAWL_BACKFILL_SINCE", default="2024-01-01")
    crawl_queue_maxsize: int = Field(alias="CRAWL_QUEUE_MAXSIZE", default=2000)
    crawl_concurrency: int = Field(alias="CRAWL_CONCURRENCY", default=4)
    
    # Channel rotation settings (NEW)
    channel_rotation_interval_hours: int = Field(alias="CHANNEL_ROTATION_INTERVAL_HOURS", default=2)
    
    # ML Worker configuration (NEW)
    ml_frames_per_second: int = Field(alias="ML_FRAMES_PER_SECOND", default=2)
    ml_positive_threshold: int = Field(alias="ML_POSITIVE_THRESHOLD", default=3, description="Number of consecutive positive frames (at 2 FPS = 1.5 seconds)")
    ml_model_path: str = Field(alias="ML_MODEL_PATH", default="models/classifier.pth")
    ml_batch_size: int = Field(alias="ML_BATCH_SIZE", default=32)
    ml_num_workers: int = Field(alias="ML_NUM_WORKERS", default=4)
    
    # Worker coordination (NEW)
    worker_lock_ttl_seconds: int = Field(alias="WORKER_LOCK_TTL_SECONDS", default=300, description="TTL for distributed worker locks in Redis")
    worker_metrics_update_interval: int = Field(alias="WORKER_METRICS_UPDATE_INTERVAL", default=60, description="How often workers update their metrics in Redis (seconds)")
    
    # API Service configuration (NEW)
    api_host: str = Field(alias="API_HOST", default="0.0.0.0")
    api_port: int = Field(alias="API_PORT", default=8000)
    api_reload: bool = Field(alias="API_RELOAD", default=False)
    api_cors_origins: str = Field(alias="API_CORS_ORIGINS", default="http://localhost:3000,http://localhost:8000")
    
    # Processing timeouts
    encode_timeout_s: int = Field(alias="ENCODE_TIMEOUT_S", default=120)
    
    # Reporting
    reports_daily_hour: int = Field(alias="REPORTS_DAILY_HOUR", default=9)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }

settings = Settings()