import random
from src.db.session import SessionLocal
from src.db.models import Label
from src.store.storage_backend import get_storage_backend


def pick_unlabeled_segment() -> str | None:
    """
    Returns a segment from storage that hasn't been labeled yet.
    
    Returns:
        Path to unlabeled segment or None if all labeled
    """
    storage = get_storage_backend()
    labeled_paths = _get_labeled_paths()
    all_segments = _get_all_segments(storage)
    
    unlabeled = [p for p in all_segments if p not in labeled_paths]
    return random.choice(unlabeled) if unlabeled else None


def _get_labeled_paths() -> set[str]:
    """Get set of already labeled segment paths (Single Responsibility)"""
    db = SessionLocal()
    try:
        return {r[0] for r in db.query(Label.segment_path).all()}
    finally:
        db.close()


def _get_all_segments(storage) -> list[str]:
    """Get list of all segments from storage (Single Responsibility)"""
    return [
        obj.name 
        for obj in storage.list_objects(prefix="segments/", recursive=True)
    ]
