import asyncio
from datetime import datetime, timedelta
import hashlib
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, BufferedInputFile
from aiogram.filters import CommandStart
from minio import S3Error
from src.config import settings
from src.db.session import init_db, SessionLocal
from src.db.models import Label
from src.label_bot.utils import pick_unlabeled_segment
from src.store.s3 import client as s3
from src.store.redis import _client as redis

BOT = Bot(token=settings.tg_bot_token)
DP = Dispatcher()
BUCKET = settings.s3_bucket
CALLBACK_PREFIX = "label:"
ASSIGNED = "fpv:assigned"
TEMP_MAP = "fpv:cbmap"

def short_id(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()[:10]

async def release_segment(path: str, sid: str):
    redis.srem(ASSIGNED, path)
    redis.hdel(TEMP_MAP, sid)

@DP.message(CommandStart())
async def start(msg: Message):
    await msg.answer("Привіт! Натисни /next щоб почати розмітку.")

async def get_next_segment(uid: int):
    for _ in range(5):
        path = pick_unlabeled_segment()
        if not path:
            return None
        if redis.sadd(ASSIGNED, path):
            try:
                obj = s3.get_object(BUCKET, path)
                data = obj.read()
                obj.close()
                obj.release_conn()
                return path, data
            except S3Error:
                redis.srem(ASSIGNED, path)
                continue
    return None

@DP.message(F.text == "/next")
async def next_video(msg: Message):
    seg = await get_next_segment(msg.from_user.id)
    if not seg:
        await msg.answer("Немає відео для розмітки.")
        return
    path, data = seg
    sid = short_id(path)
    redis.hset(TEMP_MAP, sid, path)  # сохранить соответствие

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Підходить", callback_data=f"{CALLBACK_PREFIX}{sid}:1"),
            InlineKeyboardButton(text="❌ Не підходить", callback_data=f"{CALLBACK_PREFIX}{sid}:0"),
        ]
    ])
    video = BufferedInputFile(data, filename=path.split("/")[-1])
    await msg.answer_video(video=video, caption=path, reply_markup=kb)

@DP.callback_query(F.data.startswith(CALLBACK_PREFIX))
async def label(cb):
    _, rest = cb.data.split(CALLBACK_PREFIX, 1)
    sid, dec = rest.split(":")
    path = redis.hget(TEMP_MAP, sid)
    if not path:
        await cb.answer("Застаріла кнопка", show_alert=True)
        return

    dec = int(dec)
    db = SessionLocal()
    db.add(Label(user_id=cb.from_user.id, segment_path=path, decision=dec))
    db.commit()
    db.close()
    await release_segment(path, sid)
    await cb.message.delete()
    await cb.answer("Збережено ✅")
    await next_video(cb.message)

async def main():
    init_db()
    await DP.start_polling(BOT)

if __name__ == "__main__":
    asyncio.run(main())
