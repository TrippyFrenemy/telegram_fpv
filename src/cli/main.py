from sqlalchemy import exists, select
import typer, asyncio, sys
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from pathlib import Path
from datetime import datetime
from src.config import settings
from src.db.session import init_db, SessionLocal
from src.db.models import Channel
from src.tg_client.client import client
from src.crawler.worker import crawl_channel
from src.utils.split_existing_videos import process_s3
from src.label_bot.bot import main as run_bot


app = typer.Typer(add_help_option=True)

@app.command()
def init():
    """Ініціалізація БД/сховища, створення сесії Pyrogram."""
    init_db()
    print("DB ok")
    # прогрів сесії
    async def warm():
        async with client: # перший запуск попросить код/2FA в консолі
            me = await client.get_me()
            print("TG ok:", me.id)
    asyncio.run(warm())


@app.command("add-seeds")
def add_seeds(file: Path = typer.Option(..., exists=True)):
    db = SessionLocal()
    for line in set(file.read_text(encoding="utf-8").splitlines()):
        u = line.strip()
        if not u:
            continue
        if db.execute(select(Channel).where(Channel.username == u)).scalar_one_or_none(): 
            continue
        ch = Channel(username=u) # id оновимо при першому обході
        db.add(ch)
    db.commit()
    db.close()
    print("seeds added")


@app.command("crawl")
def crawl(mode: str = typer.Option("backfill", help="latest|backfill"),
          since: str | None = None):
    async def run():
        seeds = [s.strip() for s in set(Path("seeds.txt").read_text(encoding="utf-8").splitlines()) if s.strip()]
        backfill_since = None
        if mode == "backfill":
            backfill_since = datetime.fromisoformat(since) if since else datetime.fromisoformat(settings.crawl_backfill_since)

        for seed in seeds:
            await crawl_channel(seed, mode=mode, backfill_since=backfill_since)

    asyncio.run(run())

@app.command("partition")
def partition():
    process_s3()


@app.command("bot")
def bot():
    asyncio.run(run_bot())


@app.command("resume")
def resume():
    crawl()


if __name__ == "__main__":
    app()
