from __future__ import annotations
import os, io, tempfile, hashlib, subprocess, concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd

from sqlalchemy import select
from src.db.session import SessionLocal
from src.db.models import Label
from src.config import settings
from src.store.s3 import client as s3

BUCKET = settings.s3_bucket

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _s3_exists(key: str) -> bool:
    try:
        s3.stat_object(BUCKET, key)
        return True
    except Exception:
        return False

def _upload_bytes(key: str, data: bytes):
    if not _s3_exists(key):
        s3.put_object(BUCKET, key, io.BytesIO(data), length=len(data))

def _extract_frames_ffmpeg(src_path: str, fps: float) -> list[bytes]:
    """Повертає список байтів JPEG-кодованих кадрів, витягнутих з відео за допомогою ffmpeg."""
    # Выгрузим в temp-папку пачку JPEG и прочитаем обратно
    tmpdir = Path(tempfile.mkdtemp())
    pat = tmpdir / "frame_%06d.jpg"
    try:
        # ffmpeg -r fps після -i дає уніфіковану довжину семплування
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", src_path, "-r", str(fps), "-q:v", "2", str(pat)
        ]
        subprocess.run(cmd, check=True)
        frames = []
        for jpg in sorted(tmpdir.glob("frame_*.jpg")):
            frames.append(jpg.read_bytes())
        return frames
    finally:
        for f in tmpdir.glob("*"):
            try: f.unlink()
            except: pass
        try: tmpdir.rmdir()
        except: pass

def _frames_dst_keys(label: int, seg_key: str, ts: datetime, count: int) -> list[str]:
    y = ts.strftime("%Y"); m = ts.strftime("%m")
    base = Path(seg_key).stem  # ім'я файлу без розширення
    return [f"frames/{label}/{y}/{m}/{base}_f{idx:08d}.jpg" for idx in range(count)]

def process_one(seg_key: str, decision: int, fps: float) -> list[dict]:
    # 1) скачати сегмент у тимчасовий файл
    tmp = Path(tempfile.gettempdir()) / f"{hashlib.md5(seg_key.encode()).hexdigest()}.mp4"
    s3.fget_object(BUCKET, seg_key, str(tmp))

    # 2) витягти кадри з відео
    frames = _extract_frames_ffmpeg(str(tmp), fps=fps)
    tmp.unlink(missing_ok=True)

    # 3) завантажити кадри на S3 і зібрати метадані
    now = datetime.now()
    dst_keys = _frames_dst_keys(decision, seg_key, now, len(frames))

    out_rows = []
    for data, key in zip(frames, dst_keys):
        sha = _sha256(data)
        _upload_bytes(key, data)
        out_rows.append({
            "s3_path": key,
            "label": int(decision),
            "source_segment": seg_key,
            "sha256": sha,
        })
    return out_rows

def iter_labeled_segments(limit: int | None = None) -> Iterable[tuple[str, int]]:
    db = SessionLocal()
    try:
        q = select(Label.segment_path, Label.decision)  # тільки сегменти з мітками
        if limit:
            rows = db.execute(q).fetchmany(limit)
        else:
            rows = db.execute(q).all()
        for seg, dec in rows:
            if not seg:
                continue
            if seg.startswith("segments/"):
                yield seg, int(dec)
    finally:
        db.close()
    
def sync_frames_to_local(
    out_dir: str = "data/frames",
    prefix: str = "frames/",
    workers: int = 8,
    overwrite: bool = False,
    verify_size: bool = True,
) -> None:
    """
    Синхронізує всі кадри з s3://{BUCKET}/frames/... у локальний out_dir,
    зберігаючи структуру клас/рік/місяць: data/frames/<0|1>/<YYYY>/<MM>/file.jpg
    """
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    # збираємо список об'єктів (list_objects повертає генератор)
    objs = list(s3.list_objects(BUCKET, prefix=prefix, recursive=True))

    def _download(obj) -> int:
        key = obj.object_name
        if not key.lower().endswith(".jpg"):
            return 0  # пропускаємо не-JPG
        rel = key[len(prefix):] if key.startswith(prefix) else key
        dst = base / rel  # data/frames/1/2025/10/xxx.jpg
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not overwrite:
            if verify_size and getattr(obj, "size", None) == dst.stat().st_size:
                return 0  # вже є актуальний файл
        # тягнемо з S3
        s3.fget_object(BUCKET, key, str(dst))
        return 1

    downloaded = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for n in ex.map(_download, objs):
            downloaded += n
    print(f"[sync] downloaded {downloaded} files -> {base}")

def run(
    fps: float = 2.0,
    workers: int = 4,
    limit: int | None = None,
    manifest_out: str = "data/frames_manifest.csv",
    sync_local_dir: Optional[str] = "data/frames_local",
    sync_workers: int = 8,
):
    tasks = list(iter_labeled_segments(limit=limit))
    results: list[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_one, seg, dec, fps) for seg, dec in tasks]
        for fut in concurrent.futures.as_completed(futs):
            results.extend(fut.result())

    df = pd.DataFrame(results)
    if manifest_out.endswith(".parquet"):
        df.to_parquet(manifest_out, index=False)
    else:
        df.to_csv(manifest_out, index=False)
    print(f"wrote {len(df)} rows -> {manifest_out}")

    # NEW: локальна синхронізація після формування кадрів у S3
    if sync_local_dir:
        sync_frames_to_local(out_dir=sync_local_dir, workers=sync_workers)

if __name__ == "__main__":
    # run()
    sync_frames_to_local(out_dir="data/frames", workers=8)
