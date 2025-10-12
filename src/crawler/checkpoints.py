from sqlalchemy import select
from sqlalchemy.orm import Session
from src.db.models import Channel

def get_last_message_id(db: Session, channel_id: int) -> int | None:
    ch = db.execute(select(Channel).where(Channel.id == channel_id)).scalar_one_or_none()
    return ch.last_scanned_id if ch else None


def set_last_message_id(db: Session, channel_id: int, last_id: int):
    ch = db.get(Channel, channel_id)
    if not ch:
        ch = Channel(id=channel_id)
    ch.last_scanned_id = last_id
    db.add(ch)
    db.commit()
