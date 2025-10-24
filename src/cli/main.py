from sqlalchemy import exists, select
import typer
import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from pathlib import Path
from datetime import datetime
from src.config import settings
from src.db.session import init_db, SessionLocal
from src.db.models import Channel
from src.tg_client.client import client
from src.store.redis import _client as redis

from src.crawler.worker import crawl_channel  # Був src.crawler.worker
from src.ml.ml_worker import main as run_ml_worker  # Новий ML воркер
from src.utils.split_existing_videos import process_s3
from src.label_bot.bot import main as run_bot


app = typer.Typer(add_help_option=True)

# Redis ключі
ML_QUEUE = "fpv:ml_queue"
ML_PROCESSING = "fpv:processing"
ML_STATS = "fpv:ml_stats"


@app.command()
def init():
    """Ініціалізація БД/сховища, створення сесії Pyrogram."""
    init_db()
    print("✓ DB initialized")
    
    # Прогрів сесії Telegram
    async def warm():
        async with client:
            me = await client.get_me()
            print(f"✓ Telegram session OK: {me.first_name} (ID: {me.id})")
    
    asyncio.run(warm())


@app.command("crawl")
def crawl(
    mode: str = typer.Option("backfill", help="latest|backfill"),
    since: str | None = None,
    continuous: bool = typer.Option(False, help="Безперервний обхід з ротацією")
):
    """
    Краулінг Telegram каналів з підтримкою ротації.
    
    Args:
        mode: Режим обходу (latest тільки нові, backfill з історією)
        since: Дата початку для backfill (YYYY-MM-DD)
        continuous: Безперервний режим з автоматичною ротацією каналів
    """
    async def run():
        # Читаємо seed канали
        seeds_file = Path("seeds.txt")
        if not seeds_file.exists():
            print("ERROR: seeds.txt не знайдено")
            return
            
        seeds = [
            s.strip() 
            for s in set(seeds_file.read_text(encoding="utf-8").splitlines()) 
            if s.strip() and not s.strip().startswith("#")
        ]
        
        if not seeds:
            print("ERROR: Жодного каналу в seeds.txt")
            return
        
        print(f"📡 Starting crawler for {len(seeds)} channels")
        print(f"   Mode: {mode}")
        print(f"   Rotation interval: {settings.channel_rotation_interval_hours}h")
        print(f"   ML queue: {ML_QUEUE}")
        
        # Визначаємо дату початку для backfill
        backfill_since = None
        if mode == "backfill":
            backfill_since = (
                datetime.fromisoformat(since) 
                if since 
                else datetime.fromisoformat(settings.crawl_backfill_since)
            )
            print(f"   Backfill since: {backfill_since}")
        
        # Режим безперервного обходу з ротацією
        if continuous:
            print("\n🔄 Continuous mode with rotation enabled")
            while True:
                for idx, seed in enumerate(seeds, 1):
                    print(f"\n[{idx}/{len(seeds)}] Processing: {seed}")
                    await crawl_channel(seed, mode=mode, backfill_since=backfill_since)
                    
                print("\n✓ Completed full rotation, restarting...")
        else:
            # Одноразовий прохід по всіх каналах
            for idx, seed in enumerate(seeds, 1):
                print(f"\n[{idx}/{len(seeds)}] Processing: {seed}")
                await crawl_channel(seed, mode=mode, backfill_since=backfill_since)
            
            print("\n✓ Crawling completed")
    
    asyncio.run(run())


@app.command("ml-worker")
def ml_worker():
    """
    Запуск ML воркера для обробки відео з Redis черги.
    
    Воркер:
    - Слухає чергу fpv:ml_queue
    - Витягує кадри з відео (FPS з config)
    - Класифікує кожен кадр
    - Переміщує підходящі відео в dataset/
    """
    print("🤖 Starting ML Worker")
    print(f"   Model: {settings.ml_model_path}")
    print(f"   FPS: {settings.ml_frames_per_second}")
    print(f"   Positive threshold: {settings.ml_positive_threshold} consecutive frames")
    print(f"   Batch size: {settings.ml_batch_size}")
    print(f"   Queue: {ML_QUEUE}")
    print()
    
    # Перевірка моделі
    model_path = Path(settings.ml_model_path)
    if not model_path.exists():
        print(f"❌ ERROR: Model not found at {model_path}")
        print("\nPlease train the model first:")
        print("  python -m src.ml.frame_classifier")
        return
    
    run_ml_worker()


@app.command("queue-status")
def queue_status():
    """
    Показує статус Redis черги ML обробки.
    """
    try:
        queue_len = redis.llen(ML_QUEUE)
        processing_count = redis.scard(ML_PROCESSING)
        
        # Статистика
        stats = redis.hgetall(ML_STATS)
        
        print("📊 ML Worker Queue Status")
        print("=" * 50)
        print(f"Queue length:        {queue_len}")
        print(f"Currently processing: {processing_count}")
        print()
        
        if stats:
            print("Statistics:")
            total = int(stats.get(b"total_processed", 0))
            positive = int(stats.get(b"positive_videos", 0))
            negative = int(stats.get(b"negative_videos", 0))
            avg_time = float(stats.get(b"avg_processing_time", 0))
            
            print(f"  Total processed:   {total}")
            print(f"  Positive videos:   {positive} ({positive/max(total,1)*100:.1f}%)")
            print(f"  Negative videos:   {negative} ({negative/max(total,1)*100:.1f}%)")
            print(f"  Avg processing:    {avg_time:.2f}s")
        else:
            print("No statistics available yet")
        
        print()
        
        # Показуємо кілька задач з черги
        if queue_len > 0:
            print("Next tasks in queue:")
            tasks = redis.lrange(ML_QUEUE, 0, 4)
            for i, task in enumerate(tasks, 1):
                task_str = task.decode() if isinstance(task, bytes) else task
                s3_path = task_str.split("|")[0]
                print(f"  {i}. {s3_path}")
            
            if queue_len > 5:
                print(f"  ... and {queue_len - 5} more")
    
    except Exception as e:
        print(f"❌ Error: {e}")


@app.command("clear-queue")
def clear_queue(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
):
    """
    Очищує ML чергу (використовувати обережно).
    """
    queue_len = redis.llen(ML_QUEUE)
    
    if queue_len == 0:
        print("Queue is already empty")
        return
    
    if not confirm:
        response = typer.confirm(
            f"Are you sure you want to clear {queue_len} tasks from the queue?"
        )
        if not response:
            print("Cancelled")
            return
    
    redis.delete(ML_QUEUE)
    redis.delete(ML_PROCESSING)
    print(f"✓ Cleared {queue_len} tasks from queue")


@app.command("partition")
def partition():
    """
    Сегментація існуючих відео у S3 на 5-секундні фрагменти.
    
    УВАГА: Ця команда більше не потрібна для нового воркфлоу,
    оскільки ML воркер працює з повними відео.
    """
    print("⚠️  WARNING: This command segments videos into 5s chunks")
    print("   The new ML workflow processes full videos instead")
    print()
    
    response = typer.confirm("Continue anyway?")
    if not response:
        print("Cancelled")
        return
    
    process_s3()


@app.command("bot")
def bot():
    """
    Запуск Telegram бота для розмітки сегментів.
    """
    print("🤖 Starting Label Bot")
    asyncio.run(run_bot())


@app.command("info")
def info():
    """
    Показує конфігурацію системи.
    """
    print("⚙️  System Configuration")
    print("=" * 50)
    print(f"Storage:              {settings.storage_backend}")
    print(f"S3 Endpoint:          {settings.s3_endpoint}")
    print(f"S3 Bucket:            {settings.s3_bucket}")
    print(f"Database:             {settings.db_dsn.split('@')[-1] if '@' in settings.db_dsn else 'local'}")
    print(f"Redis:                {settings.redis_dsn}")
    print()
    print("Crawler:")
    print(f"  Rotation interval:  {settings.channel_rotation_interval_hours}h")
    print(f"  Backfill since:     {settings.crawl_backfill_since}")
    print()
    print("ML Worker:")
    print(f"  Model path:         {settings.ml_model_path}")
    print(f"  Frames per second:  {settings.ml_frames_per_second}")
    print(f"  Positive threshold: {settings.ml_positive_threshold} frames")
    print(f"  Batch size:         {settings.ml_batch_size}")


if __name__ == "__main__":
    app()
