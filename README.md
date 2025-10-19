# FPV-краулер відео з Telegram

> MTProto via Pyrogram. Інкрементальний обхід з чекпойнтами, рання дедуплікація без завантаження, S3/FS-сховище, експорт маніфесту з `source_url`. Підтримка сегментації на 5-секундні фрагменти та Telegram-бота для розмітки.

## Опис проєкту

Система інкрементального збору відео з FPV-дронів із мережі публічних Telegram-каналів/чатів через **MTProto (Pyrogram)**. Обхід повідомлень, евристичний фільтр FPV, **рання недопуска дублікатів** за `file_unique_id` до завантаження, збереження валідних медіа у локальне FS або S3-сумісне сховище, фіксація метаданих у БД. Є фонове **розбиття відео на сегменти** та **Telegram-бот** для ручної розмітки цих сегментів.

**Ключові можливості**

* Інкрементальний краулінг з чекпойнтами per-channel (`last_scanned_id`), режими: `latest` · `backfill` · `resume`.
* Рання дедуплікація **до завантаження** за `file_unique_id` + контент-дедуп за SHA-256.
* Збереження за шаблоном `telegram/{channel}/{yyyy}/{mm}/{sha8}.ext` у S3.
* Сегментація кожного відео на 5-секундні фрагменти під `segments/…`.
* Ідемпотентність, ретраї з backoff, `dead_letter` реєстр.
* Telegram-бот на aiogram для розмітки сегментів (Redis-координація, Postgres-збереження).

> **Важливо:** *Docker застосунок* зараз не використовувати. Підіймайте **лише інфраструктуру**: БД, S3 (та Redis для бота) через `docker-compose`. Сам застосунок запускайте локально через `pip`/`venv`.

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
* [Сегментація відео](#сегментація-відео)
* [Telegram-бот для розмітки сегментів](#telegram-бот-для-розмітки-сегментів)
* [Звіти](#звіти)
* [Усунення проблем](#усунення-проблем)
* [Roadmap](#roadmap)
* [Структура репозиторію](#структура-репозиторію)
* [Швидкий старт](#швидкий-старт)

## Залежності

* **Python 3.12**
* **Telegram MTProto**: `TG_API_ID`, `TG_API_HASH`, `TG_PHONE_NUMBER` (для OTP) або готова `TG_SESSION`
* **DB**: PostgreSQL (`postgresql+psycopg2://…`)
* **Storage**: S3-сумісне (MinIO / AWS S3)
* **FFmpeg/ffprobe** у `$PATH` (метадані, стиск)
* **OpenCV (cv2)** для сегментації відео
* **Redis** для Telegram-бота
* ОС: Linux або Windows (консоль для OTP на першому логіні)

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

> **Docker інфраструктура:** `docker-compose up -d db minio redis`.

## Конфігурація (.env)

Основні параметри (через `pydantic-settings`).

```env
# Telegram MTProto
TG_API_ID=123456
TG_API_HASH=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TG_PHONE_NUMBER=+380xxxxxxxxx       # опційно, для OTP
TG_SESSION=fpv_session              # назва/файл сесії
TG_WORKDIR=.tg_sessions             # тека для сесій
TG_LANG=uk
TG_RPS=20
TG_RPS_BURST=30
TG_SLEEP_JITTER_MS=250
TG_BOT_TOKEN=xxxxxxxxxxxxxxxxxxxx   # для Telegram-бота розмітки

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

# Redis (для бота)
REDIS_DSN=redis://localhost:6379/0

# FPV filter
FPV_MIN_DURATION_S=5
FPV_MAX_DURATION_S=1200
FPV_KEYWORDS=#fpv,#drone,виліт,дрон
FPV_MIN_CONFIDENCE=0.3

# Encode / ffmpeg
ENCODE_TIMEOUT_S=120

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
# Інфраструктура (рекомендовано лише DB + S3 + Redis)
docker-compose up -d db minio redis

# 1)  Ініціалізація БД / авто-міграції та перший логін у Telegram (OTP)
python -m src.cli.main init

# 2)  Додати seed-канали з файлу (по одному username без @ на рядок)
python -m src.cli.main add-seeds --file seeds.txt

# 3)  Краулінг
python -m src.cli.main crawl # Оброблює спочатку нові відео, потім проходить по історії до кінцевої дати

# 4)  Сегментація наявних відео в S3
python -m src.cli.main partition

# 5)  Базова щоденна статистика
python -m src.cli.main stats

# 6)  Telegram-бот для розмітки сегментів
python -m src.cli.main bot
```

## Чекпойнти та режими обходу

* Під час обробки повідомлень оновлюється `channels.last_scanned_id` та `last_scanned_at`.

## Сховище та шляхи

Шаблон ключа/шляху для оригіналів:

```
telegram/{channel_username|id}/{yyyy}/{mm}/{sha256_prefix8}.{ext}
```

Приклад:

* S3: `s3://fpv/telegram/-1001234567890/2025/10/a1b2c3d4.mov`

**Сегменти** зберігаються під:

```
segments/{telegram/{channel}/{yyyy}/{mm}}/{basename}_{0000..}.mp4
```

## Дедуплікація

1. **Рання без завантаження**: якщо відео має Telegram `file_unique_id`, перевіряється у БД; за наявності — **скачування пропускається**.
2. **Контент-дедуп**: після завантаження рахується SHA-256, `media.sha256` — PK. Якщо збіг — запис не дублюється.
3. **Ідемпотентність повідомлень**: унікальний `(channel_id, message_id)`; повторна обробка не створює дублів.

## Евристики FPV

`fpv_confidence ∈ [0,1]` формується з:

* тексту/хештегів (`FPV_KEYWORDS`, RapidFuzz partial ratio),
* тривалості в межах `[FPV_MIN_DURATION_S, FPV_MAX_DURATION_S]`,
* орієнтації кадру (портрет часто трапляється у FPV-публікаціях),
* уточнених метаданих `ffprobe` після стиску.

Підтримувані контейнери: `mp4`, `mov`, `mkv`.

## Схема БД та індекси

**channels**: `id`, `username`, `title`, `first_seen_at`, `last_scanned_id`, `last_scanned_at`
**edges**: `src_channel_id`, `dst_channel_id`, `kind='FORWARDED_FROM'`, `first_seen_at`
**messages**: `id`, `channel_id`, `message_id`, `date`, `has_media`, `is_fwd`, `fwd_src_channel_id`, `text_hash`, `lang`, `collected_at`
**media**: `sha256` (PK), `message_pk` → `messages.id`, `tg_file_unique_id`, `mime`, `size`, `duration_s`, `width`, `height`, `s3_path`, `fpv_confidence`, `is_segment`
**runs**: `id`, `started_at`, `finished_at`, `mode`, `stats_json`
**dead_letter**: `id`, `channel_id`, `message_id`, `error`, `first_seen_at`, `retries`
**labels**: `id`, `user_id`, `segment_path`, `decision`, `created_at`

**Індекси/унікальні ключі**

* `UNIQUE messages(channel_id, message_id)`
* `UNIQUE edges(src_channel_id, dst_channel_id, kind)`
* `UNIQUE media(tg_file_unique_id)`
* `PK media.sha256`
* Допоміжні індекси на `messages(channel_id, message_id)`, `media(message_pk)`

## Сегментація відео

Модуль `src/utils/split_existing_videos.py` автоматично **розбиває відео на сегменти по 5 секунд** і завантажує їх у S3 під префіксом:

```
segments/{original_path_dir}/{basename}_{0000..}.mp4
```

> Після успішної сегментації **оригінальний файл видаляється** для економії місця.

Команда для ручного запуску:

```bash
python -m src.cli.main partition
```

Технічні деталі:

* Розбиття — `OpenCV`.
* Швидкий стиск — `ffmpeg` (`h264_nvenc` за наявності NVIDIA GPU, інакше `libx264`), `-preset fast|ultrafast`, `-crf 23`, `-movflags +faststart`.

## Telegram-бот для розмітки сегментів

Папка `src/label_bot` містить асинхронного бота на **aiogram**, який роздає користувачам 5-секундні сегменти для ручної розмітки.

**Можливості**

* Вибір нерозмічених сегментів із `segments/` у S3.
* Гарантія унікальної видачі між користувачами через Redis-множину `fpv:assigned` і тимчасову мапу callback-кнопок.
* Збереження міток у таблиці `labels` з полями `user_id`, `segment_path`, `decision (1|0)`, `created_at`.
* Команди `/start`, `/next`, `/info`. Під час старту та зупинки надсилає широкомовні нотифікації всім, хто вже розмічав.

**Запуск**

```bash
python -m src.cli.main bot
```

Перед запуском переконайтесь, що у `.env` задано `TG_BOT_TOKEN` і доступний Redis за `REDIS_DSN`.

**Критерії розмітки** (`/info`):

- **Підходить**: зняте з дрона, вид згори, помітний політ/паралакс, ознаки FPV (широкий кут, крени, OSD, тінь, «желе»), проліт/обліт/ dive, ≥2.5 c релевантного кадру.
- **Не підходить**: від першої особи (руки, шолом), заставка/нарізка без польоту, зйомка з землі/телефону, немає виду згори.

**Обов’язкові колонки**

```
channel_id, message_id, date, is_fwd, fwd_src_channel_id,
mime, size, duration_s, width, height, s3_path, fpv_confidence,
source_url
```

**Формування `source_url`**

* Публічний канал (є `username`): `https://t.me/{username}/{message_id}`
* Без `username` (приватний/від’ємний `channel_id`):
  `https://t.me/c/{abs(channel_id) - 1000000000000}/{message_id}`

## Звіти

```bash
python -m src.cli.main stats
```

Повертає добові агрегати: повідомлення з/без медіа, кількість відео.

## Усунення проблем

* **OTP/логін**: перший запуск `init` попросить код у консолі.
* **`ffprobe` not found**: встановіть FFmpeg і додайте в `PATH`.
* **Windows/TgCrypto**: за потреби поставте Microsoft Build Tools або використовуйте готові колеса PyPI.
* **FloodWait**: відрегулюйте `TG_RPS` і `TG_SLEEP_JITTER_MS`.
* **Duplicate SHA**: завдяки перевірці `file_unique_id` трапляється рідше; при гонці ловиться `IntegrityError` і обробка продовжується.
* **Redis/бот**: перевірте `REDIS_DSN` і `TG_BOT_TOKEN`; при помилці кнопок оновіть чат `/next`.

## Roadmap

* Побудова графа форвардів та BFS-обхід з RPS-лімітами.
* Розширення FPV-евристик, класифікації і попередньої фільтрації.
* Дашборд (FastAPI, read-only) + метрики.
* Превʼю кадру, вебхуки `on_new_channel` / `on_new_video`.
* `verify` для перевірки консистентності даних.
* Повноцінний режим паралельного краулінгу каналів.

## Структура репозиторію

```
/src
  /cli            # Typer CLI
  /crawler        # інкрементальний обхід, dead-letter, чекпойнти
  /tg_client      # Pyrogram клієнт, ліміти
  /extractors     # ffprobe/ffmpeg, асинхронне кодування
  /store          # S3 writer, SHA-256, Redis, MinIO
  /db             # SQLAlchemy моделі та сесія
  /filters        # FPV-евристики
  /reports        # експорт маніфесту, статистика
  /label_bot      # Telegram-бот для розмітки сегментів
  /utils          # сегментація існуючих відео (OpenCV)
/docs
seeds.txt
requirements.txt
pyproject.toml
```

## Швидкий старт

```bash
# 1) Python env
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env && nano .env                    # TG_* / DB_DSN / S3_* / REDIS_DSN

# 2) Інфраструктура: DB + S3 (+ Redis для бота)
docker-compose up -d db minio redis

# 3) Telegram MTProto сесія і БД
python -m src.cli.main init

# 4) Додайте канали у seeds.txt (по одному username на рядок), наприклад:
# voynareal_ua
# nebo_peremogy
# yigal_levin

# 5) Краулінг
python -m src.cli.main crawl

# 6) Сегментація наявних відео у S3 (створить segments/)
python -m src.cli.main partition

# 7) (Опційно) Бот для розмітки
python -m src.cli.main bot
```
