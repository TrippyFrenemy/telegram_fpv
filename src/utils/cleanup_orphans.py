import re
from src.db.session import SessionLocal
from src.db.models import Label
from src.store.storage_backend import get_storage_backend


def cleanup_orphans():
    """Remove orphaned files from storage that are not referenced in database"""
    storage = get_storage_backend()
    db = SessionLocal()
    
    try:
        # Get all labeled segment paths from database
        labeled = {r[0] for r in db.query(Label.segment_path).all()}
    finally:
        db.close()

    # --- 1. Удалить ВСЁ из telegram/ ---
    print("Deleting all telegram/ ...")
    for obj in storage.list_objects(prefix="telegram/", recursive=True):
        try:
            storage.delete(obj.object_name)
            print(f"[del] {obj.object_name}")
        except Exception as e:
            print(f"[skip] {obj.object_name}: {e}")

    # --- 2. Сегменты ---
    print("Scanning segments/ ...")
    all_segments = [
        obj.object_name 
        for obj in storage.list_objects(prefix="segments/", recursive=True)
    ]

    # Extract base group names (without _0001.mp4 etc)
    labeled_bases = {_extract_base_name(p) for p in labeled if p.startswith("segments/")}
    to_keep = {p for p in all_segments if any(p.startswith(b) for b in labeled_bases)}

    deleted = 0
    for path in all_segments:
        if path not in to_keep:
            try:
                storage.delete(path)
                deleted += 1
                print(f"[del] {path}")
            except Exception as e:
                print(f"[skip] {path}: {e}")

    print(f"Total segments deleted: {deleted}")


def _extract_base_name(path: str) -> str:
    """Extract base name from segment path (removes _0001.mp4 suffix)"""
    match = re.search(r"(.+?)_\d{4}\.mp4$", path)
    return match.group(1) if match else path


if __name__ == "__main__":
    cleanup_orphans()
