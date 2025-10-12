from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.db.models import Base
from src.config import settings


_engine = create_engine(settings.db_dsn, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


def init_db():
    Base.metadata.create_all(bind=_engine)
    with _engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_messages_channel_mid ON messages(channel_id, message_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_media_message_pk ON media(message_pk)"))
    