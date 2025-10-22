from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from src.db.models import Base
from src.config import settings


_engine = create_engine(settings.db_dsn, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


def init_db():
    Base.metadata.create_all(bind=_engine)
    with _engine.begin() as conn:
        conn.execute(text("ALTER TABLE media ADD COLUMN IF NOT EXISTS tg_file_unique_id VARCHAR(64)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS uq_messages_channel_mid ON messages(channel_id, message_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS uq_edges_unique ON edges(src_channel_id, dst_channel_id, kind)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS uq_media_tg_uid ON media(tg_file_unique_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_messages_channel_mid ON messages(channel_id, message_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_media_message_pk ON media(message_pk)"))
        conn.execute(text("CREATE TABLE IF NOT EXISTS labels (id SERIAL PRIMARY KEY, segment_path TEXT, decision INTEGER, created_at TIMESTAMP DEFAULT NOW())"))
        