import asyncio
from datetime import datetime, timedelta
import hashlib
import signal
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
    await msg.answer(
        "Привіт! Бот допомагає відбирати потрібні відео.\n"
        "Твоє завдання — переглядати відео та визначати, чи підходить воно за заданими критеріями.\n\n"
        "Для підказки по вибору тисніть /info\n"
        "Щоб почати — /next"
    )

async def notify_all_users(text: str):
    db = SessionLocal()
    try:
        user_ids = {r[0] for r in db.query(Label.user_id).distinct()}
    finally:
        db.close()

    for uid in user_ids:
        try:
            await BOT.send_message(uid, text)
        except Exception:
            continue

async def get_next_segment(uid: int):
    for _ in range(5):
        path = pick_unlabeled_segment()
        if not path:
            return None
        if redis.sismember(ASSIGNED, path):
            continue
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

@DP.message(F.text == "/info")
async def info(msg: Message):
    help_text = (
        "Натискай «Підходить», якщо відео:\n"
        "- зняте з дрона (вид зверху, аерозйомка);\n"
        "- видно місцевість, пейзаж, будівлі чи техніку згори;\n"
        "- камера рухається плавно, ніби летить — є помітний політ або паралакс об’єктів;\n"
        "- є проліт над територією, обліт цілі, dive, набір чи втрата висоти;\n"
        "- кадр під кутом до горизонту (понад 30°);\n"
        "- у кадрі видно тінь дрона, гвинти, або ефект «желе» від вібрацій;\n"
        "- є ознаки FPV-зйомки (широкий кут, швидкі крени, OSD/телеметрія на екрані);\n"
        "- це перехоплення дрона або політ у нічному/інфрачервоному режимі;\n"
        "- навіть якщо це репост, кроп чи запис екрану — підходить, якщо видно політ згори;\n"
        "- відео має мінімальну якість (не надто розмите, ≥360p).\n"
        "- все вище перечислене якщо у кадрі більше 2.5 секунд.\n"
        "Натискай «Не підходить», якщо відео:\n"
        "- зняте від першої особи (видно руки, шолом, камеру або рух від обличчя людини);\n"
        "- це прев’ю, заставка чи коротка нарізка без польоту;\n"
        "- зняте з землі або з телефону, без польоту згори;\n"
        "- у кадрі немає польоту чи виду зверху.\n\n"
        "Поради:\n"
        "- Не поспішай — переконайся, що відео справді з дрона або вид зверху.\n"
        "- Якщо сумніваєшся, краще натиснути «Не підходить»."
    )
    await msg.answer(help_text)
    

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
    await notify_all_users("Бот розмітки запущений і готовий до роботи. Він у розробці! Можуть бути баги.")

    stop_event = asyncio.Event()

    def handle_stop():
        stop_event.set()

    # перехоплення Ctrl+C і SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, f: stop_event.set())

    poller = asyncio.create_task(DP.start_polling(BOT))

    await stop_event.wait()

    poller.cancel()
    await notify_all_users("Бот розмітки вимкнений. ДЯКУЄМО за участь!")
    await BOT.session.close()

if __name__ == "__main__":
    asyncio.run(main())
