import os
import shutil
import subprocess
import tempfile
import asyncio
from datetime import datetime
from typing import Iterable
from pyrogram.enums import MessageMediaType
from pyrogram.errors import FloodWait
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.crawler.checkpoints import get_last_message_id, set_last_message_id
from src.tg_client.client import client
from src.db.session import SessionLocal
from src.db.models import Channel, Message, Media, Edge, DeadLetter
from src.extractors.media import ffprobe_info
from src.store.storage import put, sha256_bytes
from src.filters.fpv import score
from src.logger import log
from src.config import settings
from src.tg_client.limits import RateLimiter
from src.utils.split_existing_videos import split_and_upload
from src.store.s3 import client as s3

HAS_GPU = shutil.which("nvidia-smi") is not None
print("GPU acceleration is", "enabled" if HAS_GPU else "disabled")
VIDEO_MEDIA = {MessageMediaType.VIDEO, MessageMediaType.DOCUMENT}
VIDEO_EXT = {"mp4", "mov", "mkv"}


async def crawl_channel(username_or_id: str | int, mode: str = "backfill", since_id: int | None = None, backfill_since: datetime | None = None):
    since_id = None
    db = SessionLocal()
    try:
        if isinstance(username_or_id, str):
            ch = db.execute(select(Channel).where(Channel.username == username_or_id)).scalar_one_or_none()
        else:
            ch = db.get(Channel, int(username_or_id))
        if ch and mode in ("latest", "resume"):
            since_id = ch.last_scanned_id
    finally:
        db.close()

    async with client:
        async for m in client.get_chat_history(username_or_id, offset_id=0):
            if since_id and m.id <= since_id:
                break
            if backfill_since and m.date < backfill_since:
                break
            await handle_message(username_or_id, m)


async def handle_message(channel, m):
    db = SessionLocal()
    try:
        # ensure channel rows exist
        def ensure_channel(chat):
            ch = db.get(Channel, chat.id)
            if not ch:
                ch = Channel(
                    id=chat.id,
                    username=getattr(chat, "username", None),
                    title=getattr(chat, "title", None),
                )
            else:
                ch.username = getattr(chat, "username", ch.username)
                ch.title = getattr(chat, "title", ch.title)
            ch.last_scanned_id = m.id
            ch.last_scanned_at = datetime.now()
            db.add(ch)

        ensure_channel(m.chat)

        is_fwd = bool(m.forward_from_chat)
        fwd_src_id = m.forward_from_chat.id if m.forward_from_chat else None
        if is_fwd and fwd_src_id:
            ensure_channel(m.forward_from_chat)
            db.flush()
            db.add(Edge(src_channel_id=fwd_src_id, dst_channel_id=m.chat.id))

        text = m.caption or m.text or ""
        has_media = m.media in VIDEO_MEDIA

        # спроба знайти існуюче повідомлення
        existing_msg = db.execute(
            select(Message).where(
                Message.channel_id == m.chat.id,
                Message.message_id == m.id
            )
        ).scalar_one_or_none()

        # якщо повідомлення є, перевіряємо чи є в ньому медіа
        if existing_msg:
            already = db.execute(
                select(Media).where(Media.message_pk == existing_msg.id)
            ).first()
            if already:
                db.close()
                return

        # створюємо нове повідомлення, якщо не знайшли
        if existing_msg:
            msg = existing_msg
        else:
            msg = Message(
                channel_id=m.chat.id,
                message_id=m.id,
                date=m.date,
                has_media=has_media,
                is_fwd=bool(m.forward_from_chat),
                fwd_src_channel_id=(m.forward_from_chat.id if m.forward_from_chat else None),
                text_hash=str(hash(m.caption or m.text or "")) if (m.caption or m.text) else None,
                lang=None,
            )
            db.add(msg)
            db.flush()

        if not has_media:
            db.commit()
            log.info("message", channel=channel, mid=m.id)
            return

        # ----- Stage 0: фільтр без завантаження -----
        file_name = None
        duration = None
        width = height = None

        if m.video:
            file_name = m.video.file_name
            duration = getattr(m.video, "duration", None)
            width = getattr(m.video, "width", None)
            height = getattr(m.video, "height", None)
            tg_uid = getattr(m.video, "file_unique_id", None)
        elif m.document:
            file_name = m.document.file_name  # інколи відео приходить як document
            tg_uid = getattr(m.document, "file_unique_id", None)

        # якщо UID є, пропускаємо дублікат по ньому
        if tg_uid:
            hit = db.execute(
                select(Media).where(Media.tg_file_unique_id == tg_uid)
            ).scalar_one_or_none()
            if hit:
                db.commit()
                log.info("dup_skip", channel=channel, mid=m.id)
                return
            
        ext = (file_name.rsplit(".", 1)[-1].lower() if file_name and "." in file_name else None)
        if ext not in VIDEO_EXT:
            db.commit()
            return

        prelim = score(text=text, duration_s=duration, width=width, height=height)
        if prelim.confidence < settings.fpv_min_confidence:
            db.commit()
            return

        # ----- Stage 1: завантаження + точні метадані -----
        file = await m.download(in_memory=True)
        data = file.getbuffer()

        fd_in, tmp_in = tempfile.mkstemp(suffix=f".{ext}")
        os.close(fd_in)
        with open(tmp_in, "wb") as f:
            f.write(data)

        tmp_out = tmp_in + "_compressed.mp4"

        try:
            # швидке кодування без втрати видимої якості
            if HAS_GPU:
                codec = "h264_nvenc"
                preset = "fast"
            else:
                codec = "libx264"
                preset = "ultrafast"

            subprocess.run([
                "ffmpeg", "-y", "-i", tmp_in,
                "-c:v", codec, "-preset", preset,
                "-crf", "23", "-movflags", "+faststart",
                "-c:a", "aac", "-b:a", "128k",
                tmp_out
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            with open(tmp_out, "rb") as f:
                data = f.read()
            ext = "mp4"

        except subprocess.CalledProcessError:
            pass
        finally:
            for p in (tmp_in, tmp_out):
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

        # --- Завантаження в S3 або FS ---
        stored = put(data, channel=m.chat.username or m.chat.id, when=m.date, ext=ext)

        # --- Витяг метаданих після компресії ---
        info = ffprobe_info(str(file.name))
        dec = score(
            text=text,
            duration_s=info.duration_s or duration,
            width=info.width or width,
            height=info.height or height,
        )

        existing = db.get(Media, stored.sha256)
        if not existing:
            try:
                db.add(Media(
                    sha256=stored.sha256,
                    message_pk=msg.id,
                    tg_file_unique_id=tg_uid,
                    mime=info.mime,
                    size=info.size,
                    duration_s=info.duration_s,
                    width=info.width,
                    height=info.height,
                    s3_path=stored.relpath,
                    fpv_confidence=dec.confidence,
                ))
                db.commit()
            except IntegrityError:
                db.rollback()  # інший воркер міг вже вставити
        else:
            # дубликат контента — ничего не вставляем в media
            pass
        db.commit()
        log.info("message", channel=channel, mid=m.id)

        # # ----- Stage 2: сегментація -----
        # try:
        #     threading.Thread(target=split_and_upload, args=(stored.relpath,), daemon=True).start()
        # except Exception as e:
        #     log.error("segmentation_failed", err=str(e))

    except FloodWait as fw:
        log.warning("flood_wait", seconds=fw.value)
        await asyncio.sleep(fw.value)
    except Exception as e:
        log.error("dead_letter", err=str(e))
        db.rollback()
        db.add(DeadLetter(channel_id=m.chat.id, message_id=m.id, error=str(e)))
        db.commit()
    finally:
        db.close()
