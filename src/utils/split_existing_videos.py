from datetime import datetime, timedelta
import os, tempfile, uuid
from pathlib import Path
from minio.error import S3Error
import pytz
from src.config import settings
from src.store.s3 import client as s3
from src.utils.video_splitter import split_video

STALE_DELTA = timedelta(minutes=2)

def split_and_upload(src_key: str):
    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp4")
    s3.fget_object(settings.s3_bucket, src_key, tmp_path)
    try:
        segments = split_video(tmp_path, tempfile.gettempdir(), segment_s=5)
        seg_prefix = f"segments/{os.path.dirname(src_key)}"
        for seg in segments:
            seg_name = os.path.basename(seg)
            dst_key = f"{seg_prefix}/{seg_name}"
            s3.fput_object(settings.s3_bucket, dst_key, seg)
            os.remove(seg)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def process_s3():
    now = datetime.now(pytz.timezone("Europe/Kyiv"))
    objs = s3.list_objects(settings.s3_bucket, prefix="telegram/", recursive=True)
    for obj in objs:
        # пропускаємо нові файли
        if not hasattr(obj, "last_modified"):
            continue
        if (now - obj.last_modified) < STALE_DELTA:
            print(f"[skip: too fresh] {obj.object_name}")
            continue

        tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp4")
        seg_prefix = f"segments/{os.path.dirname(obj.object_name)}"
        try:
            s3.fget_object(settings.s3_bucket, obj.object_name, tmp_path)
            segments = split_video(tmp_path, tempfile.gettempdir(), segment_s=5)
            for seg in segments:
                seg_name = os.path.basename(seg)
                dst_key = f"{seg_prefix}/{seg_name}"
                s3.fput_object(settings.s3_bucket, dst_key, seg)
                os.remove(seg)
                print(f"[uploaded] {dst_key}")
        except S3Error as e:
            print("skip:", e)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    process_s3()
