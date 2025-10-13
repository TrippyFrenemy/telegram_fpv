from datetime import datetime
from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Channel(Base):
    __tablename__ = "channels"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True) # tg id
    username: Mapped[str | None] = mapped_column(String(255))
    title: Mapped[str | None] = mapped_column(String(512))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    last_scanned_id: Mapped[int | None] = mapped_column(BigInteger)
    last_scanned_at: Mapped[datetime | None] = mapped_column(DateTime)


class Edge(Base):
    __tablename__ = "edges"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    src_channel_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("channels.id"))
    dst_channel_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("channels.id"))
    kind: Mapped[str] = mapped_column(String(32), default="FORWARDED_FROM")
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    channel_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("channels.id"))
    message_id: Mapped[int] = mapped_column(BigInteger)
    date: Mapped[datetime] = mapped_column(DateTime)
    has_media: Mapped[bool] = mapped_column(Boolean, default=False)
    is_fwd: Mapped[bool] = mapped_column(Boolean, default=False)
    fwd_src_channel_id: Mapped[int | None] = mapped_column(BigInteger)
    text_hash: Mapped[str | None] = mapped_column(String(64))
    lang: Mapped[str | None] = mapped_column(String(8))
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class Media(Base):
    __tablename__ = "media"
    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    message_pk: Mapped[int] = mapped_column(Integer, ForeignKey("messages.id"))
    tg_file_unique_id: Mapped[str | None] = mapped_column(String(64))  # NEW
    mime: Mapped[str | None] = mapped_column(String(128))
    size: Mapped[int | None] = mapped_column(BigInteger)
    duration_s: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    s3_path: Mapped[str | None] = mapped_column(String(512))
    fpv_confidence: Mapped[float | None] = mapped_column(Float)


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    mode: Mapped[str] = mapped_column(String(32))
    stats_json: Mapped[dict | None] = mapped_column(JSON)


class DeadLetter(Base):
    __tablename__ = "dead_letter"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    channel_id: Mapped[int | None] = mapped_column(BigInteger)
    message_id: Mapped[int | None] = mapped_column(BigInteger)
    error: Mapped[str] = mapped_column(String(1024))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    retries: Mapped[int] = mapped_column(Integer, default=0)
