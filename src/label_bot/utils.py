import random
from src.db.session import SessionLocal
from src.db.models import Media, Label
from src.store.s3 import client as s3
from src.config import settings


def pick_unlabeled_segment() -> str | None:
    """
    Возвращает случайный сегмент из S3/segments/,
    который ещё не размечен в таблице labels.
    """
    db = SessionLocal()
    try:
        labeled = {r[0] for r in db.query(Label.segment_path).all()}
    finally:
        db.close()

    all_segments = [
        obj.object_name for obj in s3.list_objects(settings.s3_bucket, prefix="segments/", recursive=True)
    ]

    unlabeled = [p for p in all_segments if p not in labeled]
    return random.choice(unlabeled) if unlabeled else None
