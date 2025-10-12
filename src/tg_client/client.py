import sys, asyncio
from pyrogram import Client
from pathlib import Path
from src.config import settings

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

workdir = Path(settings.tg_workdir)
workdir.mkdir(parents=True, exist_ok=True)


client = Client(
    name=settings.tg_session,
    api_id=settings.tg_api_id,
    api_hash=settings.tg_api_hash,
    phone_number=settings.tg_phone_number,
    workdir=str(workdir),
    lang_code=settings.tg_lang,
    in_memory=False,
    no_updates=True,
)
