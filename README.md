# FPV‑краулер відео з Telegram

> MTProto via Pyrogram. Incremental crawl з чекпойнтами, рання дедуплікація без завантаження, S3/FS storage, manifest export із `source_url`.

## Опис проєкту

Система інкрементального збору відео з FPV‑дронів із мережі публічних Telegram‑каналів/чатів через **MTProto (Pyrogram, не Bot API)**. Інструмент обходить повідомлення, розпізнає форварди (будує граф джерел), фільтрує потенційні FPV‑відео за евристиками, **не завантажує дублікати** (рання перевірка за Telegram `file_unique_id`), зберігає валідні медіа у локальне FS або S3‑сумісне сховище та фіксує метадані в БД. Підтримує експорт маніфесту з прямим посиланням на оригінальне повідомлення в Telegram (`source_url`).

**Ключові можливості**

* Інкрементальний краулінг з чекпойнтами per‑channel (`last_scanned_id`), режими: `latest` · `backfill` · `resume`.
* Побудова графа джерел за форвардами та інкрементальний обхід.
* Рання дедуплікація **до завантаження** за `file_unique_id` + контент‑дедуп за SHA‑256.
* Збереження за шаблоном `telegram/{channel}/{yyyy}/{mm}/{sha8}.ext` у FS або S3.
* Маніфест CSV/Parquet з `source_url` на оригінал у Telegram.
* Ідемпотентність, ретраї з backoff, `dead_letter` реєстр.

> **Важливо:** *Docker App* поки що **не використовувати**. Для інфраструктури можна піднімати лише **DB та S3** через `docker-compose`. Застосунок запускайте локально через `pip`/`venv`.

---

## Зміст

* [Залежності](#залежності)
* [Встановлення через pip](#встановлення-через-pip)
* [Конфігурація (.env)](#конфігурація-env)
* [Команди CLI](#команди-cli)
* [Чекпойнти та режими обходу](#чекпойнти-та-режими-обходу)
* [Сховище та шляхи](#сховище-та-шляхи)
* [Дедуплікація](#дедуплікація)
* [Евристики FPV](#евристики-fpv)
* [Схема БД та індекси](#схема-бд-та-індекси)
* [Маніфест датасету](#маніфест-датасету)
* [Звіти](#звіти)
* [Усунення проблем](#усунення-проблем)
* [Roadmap](#roadmap)
* [Структура репозиторію](#структура-репозиторію)

---

## Залежності

* **Python 3.12**
* **Telegram MTProto credentials**: `TG_API_ID`, `TG_API_HASH`, `TG_PHONE_NUMBER` (для OTP логіну) або `TG_SESSION`
* **DB**: PostgreSQL (`postgresql+psycopg2://…`) або SQLite для PoC
* **Storage**: локальний FS або S3‑сумісне (MinIO / AWS S3)
* **FFmpeg/ffprobe** у `$PATH` для витягування метаданих
* ОС: Linux або Windows (консоль для OTP під час першого логіну)

## Встановлення через pip

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env  # заповніть ключі
```

> **Docker:** піднімайте тільки інфраструктуру: `docker-compose up -d db minio`.

## Конфігурація (.env)

Основні параметри (через `pydantic-settings`).

```env
# Telegram MTProto
TG_API_ID=123456
TG_API_HASH=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TG_PHONE_NUMBER=+380xxxxxxxxx       # опційно, для OTP
TG_SESSION=fpv_session              # назва/файл сесії
TG_WORKDIR=.tg_sessions             # текa для сесій
TG_LANG=uk
TG_RPS=20
TG_RPS_BURST=30
TG_SLEEP_JITTER_MS=250

# Database
DB_DSN=postgresql+psycopg2://user:pass@host:5432/db

# Storage
STORAGE_BACKEND=fs                  # fs | s3
FS_ROOT=./data
S3_ENDPOINT=http://127.0.0.1:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=fpv
S3_REGION=us-east-1
S3_SECURE=false

# FPV filter
FPV_MIN_DURATION_S=5
FPV_MAX_DURATION_S=1200
FPV_KEYWORDS=#fpv,#drone,виліт,дрон
FPV_MIN_CONFIDENCE=0.3

# Crawl
CRAWL_BACKFILL_SINCE=2024-01-01
CRAWL_QUEUE_MAXSIZE=2000
CRAWL_CONCURRENCY=4

# Reports
REPORTS_DAILY_HOUR=09
```

## Команди CLI

Усі команди через **Typer**: `python -m src.cli.main <cmd>`

```bash
# Підняти інфраструктуру (тільки DB + S3)
docker-compose up -d db minio

# 1)  Ініціалізація БД / авто-міграції та перший логін у Telegram (OTP)
python -m src.cli.main init

# 2)  Додати seed-канали з файлу (по одному username без @ на рядок)
python -m src.cli.main add-seeds --file seeds.txt

# 3)  Краулінг
# 3.1) Лише нове вище чекпойнта
python -m src.cli.main crawl --mode latest
# 3.2) Бекфіл від дати (зупинка на since з .env файлу)
python -m src.cli.main crawl --mode backfill

# 4)  Експорт маніфесту з source_url
python -m src.cli.main export-manifest --out manifest.parquet

# 5)  Базова щоденна статистика
python -m src.cli.main stats
```

## Чекпойнти та режими обходу

* Під час обробки повідомлень оновлюється `channels.last_scanned_id` та `last_scanned_at`.
* `resume` читає історію зверху вниз і **перериває** обхід, коли зустрічає `message_id ≤ last_scanned_id`.
* `latest` працює аналогічно, але орієнтований на нові події від останнього скану.
* `backfill --since YYYY-MM-DD` читає вглиб до заданої дати та зупиняється.

## Сховище та шляхи

Шаблон ключа/шляху:

```
telegram/{channel_username|id}/{yyyy}/{mm}/{sha256_prefix8}.{ext}
```

Приклади:

* FS: `./data/telegram/voynareal_ua/2025/10/a1b2c3d4.mp4`
* S3: `s3://fpv/telegram/-1001234567890/2025/10/a1b2c3d4.mov`

## Дедуплікація

1. **Рання без завантаження**: якщо у повідомленні відео має Telegram `file_unique_id`, система перевіряє у БД та **пропускає завантаження**, якщо такий UID уже зафіксований.
2. **Контент‑дедуп**: після скачування рахується SHA‑256, `media.sha256` — PK. Якщо збіг — запис не дублюється.
3. **Ідемпотентність повідомлень**: унікальний `(channel_id, message_id)`; повторна обробка безпечно завершиться без дубляжу.

## Евристики FPV

`fpv_confidence ∈ [0,1]` формується з:

* тексту/хештегів (ключові слова з `FPV_KEYWORDS`),
* тривалості в межах `[FPV_MIN_DURATION_S, FPV_MAX_DURATION_S]`,
* орієнтації кадру (портрет/альбом),
* уточнених метаданих `ffprobe` після попереднього фільтру.

Підтримувані контейнери: `mp4`, `mov`, `mkv`.

## Схема БД та індекси

**channels**: `id`, `username`, `title`, `first_seen_at`, `last_scanned_id`, `last_scanned_at`

**edges**: `src_channel_id`, `dst_channel_id`, `kind='FORWARDED_FROM'`, `first_seen_at`

**messages**: `id`, `channel_id`, `message_id`, `date`, `has_media`, `is_fwd`, `fwd_src_channel_id`, `text_hash`, `lang`, `collected_at`

**media**: `sha256` (PK), `message_pk` → `messages.id`, `tg_file_unique_id` (UNIQUE), `mime`, `size`, `duration_s`, `width`, `height`, `s3_path`, `fpv_confidence`

**runs**: `id`, `started_at`, `finished_at`, `mode`, `stats_json`

**dead_letter**: `id`, `channel_id`, `message_id`, `error`, `first_seen_at`, `retries`

**Індекси/унікальні ключі**

* `UNIQUE messages(channel_id, message_id)`
* `UNIQUE media(tg_file_unique_id)`
* `PK media.sha256`
* `UNIQUE edges(src_channel_id, dst_channel_id, kind)`

## Маніфест датасету

Експорт: `python -m src.cli.main export-manifest --out manifest.parquet`

**Обов’язкові колонки**

```
channel_id, message_id, date, is_fwd, fwd_src_channel_id,
mime, size, duration_s, width, height, s3_path, fpv_confidence,
source_url
```

**Формування `source_url`**

* Публічний канал (є `username`):
  `https://t.me/{username}/{message_id}`
* Приватний/без username (`channel_id` відʼємний):
  `https://t.me/c/{abs(channel_id) - 1000000000000}/{message_id}`

## Звіти

`python -m src.cli.main stats` — добові агрегати: нові канали, повідомлення з/без медіа, кількість відео.

## Усунення проблем

* **OTP/логін**: перший запуск `init` попросить код у консолі.
* **`ffprobe` not found**: встановіть FFmpeg та додайте у `PATH`.
* **Windows/TgCrypto**: за потреби поставте Microsoft Build Tools або використайте попередньозібрані колеса PyPI.
* **FloodWait**: відрегулюйте `TG_RPS` і `TG_SLEEP_JITTER_MS`.
* **Duplicate SHA**: завдяки ранній дедуплікації за `file_unique_id` трапляється рідше; у випадку гонки — ловиться `IntegrityError` і дає чистий повтор.

## Roadmap

* BFS обхід графа форвардів з RPS‑обмеженнями.
* Дашборд (FastAPI, read‑only) + метрики.
* Адекватна відбірка відео з fpv дронами
* Превʼю кадру, вебхуки `on_new_channel` / `on_new_video`.
* `verify` для перевірки консистентності даних.

## Структура репозиторію

```
/src
  /cli            # Typer CLI
  /crawler        # інкрементальний обхід, dead-letter
  /tg_client      # Pyrogram клієнт, ліміти, ретраї
  /extractors     # ffprobe метадані
  /store          # FS/S3 writer, SHA-256
  /db             # SQLAlchemy моделі й сесія
  /filters        # FPV-евристики
  /reports        # експорт маніфесту, статистика
/docs
seeds.txt
requirements.txt
pyproject.toml
```

## Швидкий старт

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env && nano .env                    # TG_API_ID/TG_API_HASH/TG_PHONE_NUMBER

# інфраструктура: лише DB + S3
docker-compose up -d db minio

python -m src.cli.main init
python -m src.cli.main add-seeds --file seeds.txt
python -m src.cli.main crawl --mode resume
python -m src.cli.main export-manifest --out manifest.parquet
```
