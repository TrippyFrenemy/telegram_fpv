from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    tg_api_id: int = Field(alias="TG_API_ID")
    tg_api_hash: str = Field(alias="TG_API_HASH")
    tg_phone_number: str | None = Field(alias="TG_PHONE_NUMBER")
    tg_session: str = Field(alias="TG_SESSION", default="fpv_session")
    tg_workdir: Path = Field(alias="TG_WORKDIR", default=Path(".tg_sessions"))
    tg_lang: str = Field(alias="TG_LANG", default="uk")
    tg_rps: int = Field(alias="TG_RPS", default=20)
    tg_rps_burst: int = Field(alias="TG_RPS_BURST", default=30)
    tg_sleep_jitter_ms: int = Field(alias="TG_SLEEP_JITTER_MS", default=250)
    tg_bot_token: str = Field(alias="TG_BOT_TOKEN")

    db_dsn: str = Field(alias="DB_DSN", default="sqlite:///./fpv.db")

    storage_backend: str = Field(alias="STORAGE_BACKEND", default="fs")
    s3_endpoint: str | None = Field(alias="S3_ENDPOINT", default=None)
    s3_access_key: str | None = Field(alias="S3_ACCESS_KEY", default=None)
    s3_secret_key: str | None = Field(alias="S3_SECRET_KEY", default=None)
    s3_bucket: str | None = Field(alias="S3_BUCKET", default=None)
    s3_region: str | None = Field(alias="S3_REGION", default="us-east-1")
    s3_secure: bool = Field(alias="S3_SECURE", default=False)
    redis_dsn: str = Field(alias="REDIS_DSN", default="redis://localhost:6379/0")

    fs_root: Path = Field(alias="FS_ROOT", default=Path("./data"))

    fpv_min_duration_s: int = Field(alias="FPV_MIN_DURATION_S", default=5)
    fpv_max_duration_s: int = Field(alias="FPV_MAX_DURATION_S", default=1200)
    fpv_keywords: str = Field(alias="FPV_KEYWORDS", default="#fpv,#drone")
    fpv_min_confidence: float = Field(alias="FPV_MIN_CONFIDENCE", default=0.3)

    encode_timeout_s: int = Field(alias="ENCODE_TIMEOUT_S", default=120)

    crawl_backfill_since: str = Field(alias="CRAWL_BACKFILL_SINCE", default="2024-01-01")
    crawl_queue_maxsize: int = Field(alias="CRAWL_QUEUE_MAXSIZE", default=2000)
    crawl_concurrency: int = Field(alias="CRAWL_CONCURRENCY", default=4)

    reports_daily_hour: int = Field(alias="REPORTS_DAILY_HOUR", default=9)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }

settings = Settings()