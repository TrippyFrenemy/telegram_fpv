import re
from src.db.session import SessionLocal
from src.db.models import Label
from src.store.s3 import client as s3
from src.config import settings


def cleanup_orphans():
    db = SessionLocal()
    try:
        labeled = {r[0] for r in db.query(Label.segment_path).all()}
    finally:
        db.close()

    # --- 1. Удалить ВСЁ из telegram/ ---
    print("Deleting all telegram/ ...")
    for obj in s3.list_objects(settings.s3_bucket, prefix="telegram/", recursive=True):
        try:
            s3.remove_object(settings.s3_bucket, obj.object_name)
            print(f"[del] {obj.object_name}")
        except Exception as e:
            print(f"[skip] {obj.object_name}: {e}")

    # --- 2. Сегменты ---
    print("Scanning segments/ ...")
    all_segments = [
        o.object_name for o in s3.list_objects(settings.s3_bucket, prefix="segments/", recursive=True)
    ]

    # Извлекаем базовые имена групп (без _0001.mp4 и т.п.)
    def base_name(path: str) -> str:
        m = re.search(r"(.+?)_\d{4}\.mp4$", path)
        return m.group(1) if m else path

    labeled_bases = {base_name(p) for p in labeled if p.startswith("segments/")}
    to_keep = {p for p in all_segments if any(p.startswith(b) for b in labeled_bases)}

    deleted = 0
    for p in all_segments:
        if p not in to_keep:
            try:
                s3.remove_object(settings.s3_bucket, p)
                deleted += 1
                print(f"[del] {p}")
            except Exception as e:
                print(f"[skip] {p}: {e}")

    print(f"Total segments deleted: {deleted}")


if __name__ == "__main__":
    cleanup_orphans()
