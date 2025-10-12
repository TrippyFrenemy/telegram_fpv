# FPV-краулер відео з Telegram

> MTProto via Pyrogram. Incremental crawl, FPV heuristics, S3/FS storage, manifest export.

## Опис проєкту

Система інкрементального збору відео з FPV-дронів із мережі публічних Telegram‑каналів/чатів через **MTProto (Pyrogram, не Bot API)**. Інструмент обходить повідомлення, розпізнає форварди (будує граф джерел), відбирає потенційні FPV‑відео за евристиками, зберігає медіа в локальне FS або S3‑сумісне сховище та фіксує метадані в БД. Підтримує маніфест датасету та базові звіти.

Ключові можливості:
- Інкрементальний краулінг з чекпойнтами на рівні каналу (`last_message_id`), режими `latest`/`backfill`/`resume`.
- Виявлення ланцюжків форвардів і побудова графа джерел.
- Відбір FPV за текстом/хештегами, тривалістю і орієнтацією кадру з **оцінкою `fpv_confidence ∈ [0,1]`**. **ТРЕБА ЗАМІНИТИ**
- Дедуплікація по SHA‑256 файлу та Telegram‑ідентифікаторам.
- Збереження у структуру шляху `telegram/{channel}/{yyyy}/{mm}/{sha8}.ext` на FS або S3.
<!-- - Маніфест CSV/Parquet з обов’язковими полями та щоденні звіти. -->

> **Важливо: запуск в Docker App наразі не підтримується.** Не використовуйте режим *Docker App* або розгортання «Docker в Docker». Запускайте локально через `pip`/`venv`. Розділ із `docker-compose` в репозиторії є експериментальним і використовується лище для DB та S3, у подальшому зміниться.


## Вміст

- [Залежності](#requirements--залежності)
- [Встановлення через pip](#install-via-pip--встановлення-через-pip)
- [Конфігурація (.env)](#configuration-env--конфігурація-env)
- [Команди](#cli--команди)
- [Початкові канали](#seeds--початкові-канали)
- [Сховище](#storage-layout--сховище)
- [Евристики FPV](#heuristics--евристики-fpv)
- [Схема БД](#database-schema--схема-бд)
- [Маніфест](#manifest--маніфест)
- [Звіти](#stats--звіти)
- [Усунення проблем](#troubleshooting--усунення-проблем)
- [Roadmap](#roadmap)


## Залежності

- **Python 3.12**
- **Telegram MTProto credentials**: `TG_API_ID`, `TG_API_HASH`, `TG_PHONE_NUMBER`.
- **DB**: PostgreSQL (`DB_DSN=postgresql+psycopg2://...`)
- **Storage**: локальне FS або S3‑сумісне (MinIO, AWS S3).
- **FFmpeg/ffprobe** в `$PATH` для вилучення метаданих відео.
- ОС: Linux/Windows (Windows потребує консоль для OTP‑логіну, Pyrogram сумісний).


## Встановлення через pip

```bash
# 1) Create venv / Створити середовище
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2) Install deps / Встановити залежності
pip install -r requirements.txt
# або editable mode:
# pip install -e .

# 3) Prepare .env / Підготуйте .env
cp .env.example .env  # або створіть вручну як нижче
```

> **Docker:** не використовуйте *Docker App* поки що.


## Конфігурація (.env)

Налаштування зчитуються через `pydantic-settings`. Приклади ключів і дефолтів.

```env
# Telegram MTProto
TG_API_ID=123456
TG_API_HASH=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TG_PHONE_NUMBER=+380xxxxxxxxx            # опційно
TG_SESSION=fpv_session                   # назва сесії
TG_WORKDIR=.tg_sessions                  # папка для сесій
TG_LANG=uk
TG_RPS=20                                # запитів/сек
TG_RPS_BURST=30
TG_SLEEP_JITTER_MS=250

# Database
DB_DSN=sqlite:///./fpv.db                 # або postgres://...

# Storage
STORAGE_BACKEND=fs                        # fs|s3
FS_ROOT=./data                            # корінь FS
S3_ENDPOINT=http://127.0.0.1:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=fpv
S3_REGION=us-east-1
S3_SECURE=false

# FPV filter
FPV_MIN_DURATION_S=5
FPV_MAX_DURATION_S=1200
FPV_KEYWORDS=#fpv,#drone
FPV_MIN_CONFIDENCE=0.3

# Crawl
CRAWL_BACKFILL_SINCE=2024-01-01
CRAWL_QUEUE_MAXSIZE=2000
CRAWL_CONCURRENCY=4

# Reports
REPORTS_DAILY_HOUR=09
```

## Команди

Бінарій: `python -m src.cli.main` (Typer CLI).

```bash
# Ініціалізація БД та первинний логін у Telegram (OTP у консолі)
python -m src.cli.main init

# Додати seed-канали зі списку
python -m src.cli.main add-seeds --file seeds.txt

# Краул: режими backfill
python -m src.cli.main crawl 

# Експорт маніфесту (CSV/Parquet за розширенням)
python -m src.cli.main export-manifest --out manifest.parquet

# Огляд щоденної статистики
python -m src.cli.main stats
```

Поведінка:
- `init`: створює таблиці, індекси, готує сесію Pyrogram.
- `add-seeds`: записує початкові канали, їх `id` оновляться під час першого обходу.
- `crawl`: проходить історію seed‑каналів. У `backfill` зупиняється на `CRAWL_BACKFILL_SINCE`.
- `resume`: синонім `crawl` із дефолтами.
- `export-manifest`: формує маніфест з полями нижче.
- `stats`: останні 30 днів агрегатів.

## Початкові канали

Файл `seeds.txt`, по одному `username` на рядок, без `@`:

```
voynareal_ua
nebo_peremogy
```


## Сховище

Файл кладеться за шаблоном:

```
telegram/{channel_username|id}/{yyyy}/{mm}/{sha256_prefix8}.{ext}
```

Приклади:
- FS: `./data/telegram/voynareal_ua/2025/10/a1b2c3d4.mp4`
- S3: `s3://fpv/telegram/-1001234567890/2025/10/a1b2c3d4.mov`

Дедуплікація:
- По вмісту: SHA‑256 байтів файлу.
- По сутності: `(channel_id, message_id)` і прив’язка `media.message_pk`.



## Евристики FPV (Працює як написано, потрібно переробити на адекватне)

Оцінка `fpv_confidence` будується з:
- Текст/хештеги: частковий fuzzy‑match ключових слів (`FPV_KEYWORDS`).
- Тривалість: у межах `[FPV_MIN_DURATION_S, FPV_MAX_DURATION_S]`.
- Орієнтація кадру: портретна (`height >= width`).

Типові бали: текст ≈ 0.3, тривалість ≈ 0.3, портрет ≈ 0.1. Обрізання на `1.0`.
Попередній фільтр до завантаження, потім точні метадані через `ffprobe` і повторна оцінка.

Підтримувані контейнери: `mp4`, `mov`, `mkv`.


## Схема БД

Таблиці (мінімум):

**channels**
- `id`(PK, tg id), `username`, `title`, `first_seen_at`, `last_scanned_id`, `last_scanned_at`

**edges**
- `src_channel_id` → `channels.id`
- `dst_channel_id` → `channels.id`
- `kind='FORWARDED_FROM'`, `first_seen_at`

**messages**
- `id`(PK), `channel_id` → `channels.id`
- `message_id`, `date`, `has_media`, `is_fwd`, `fwd_src_channel_id`, `text_hash`, `lang`, `collected_at`

**media**
- `sha256`(PK), `message_pk` → `messages.id`
- `mime`, `size`, `duration_s`, `width`, `height`, `s3_path`, `fpv_confidence`

**runs**
- `id`, `started_at`, `finished_at`, `mode`, `stats_json`

**dead_letter**
- `id`, `channel_id`, `message_id`, `error`, `first_seen_at`, `retries`

Індекси:
- `messages(channel_id, message_id)`
- `media(message_pk)`


## Маніфест

Обов’язкові колонки експорту:
```
channel_id, message_id, date, is_fwd, fwd_src_channel_id,
mime, size, duration_s, width, height, s3_path, fpv_confidence
```
Формати: `.parquet` або `.csv` (визначається за розширенням).


## Звіти

`python -m src.cli.main stats` виводить за днями: кількість повідомлень з/без медіа та число відео.


## Усунення проблем

- **OTP/логін**: перший старт `python -m src.cli.main init` відкриє сесію та попросить код у консолі.
- **`ffprobe` not found**: встановіть FFmpeg та переконайтесь, що `ffprobe` у `PATH`.
- **Windows**: для Pyrogram інколи потрібна політика циклу подій; вона виставляється в коді автоматично. За помилок з `TgCrypto` поставте актуальні Build Tools або використовуйте попередньозібрані колеса.
- **FloodWait**: клієнт очікує і ретраїть автоматично; збалансуйте `TG_RPS` та `TG_SLEEP_JITTER_MS`.
- **FK violations**: запускайте `python -m src.cli.main init` до краулу, використовуйте `add-seeds` перед `crawl`.


## Roadmap

- BFS обхід графа форвардів з лімітами RPS.
- Повноцінний дашборд із метриками (FastAPI, read‑only).
- Розширені евристики FPV та активне навчання.
- Перевірка цілісності (`verify`), прев’ю кадрів, вебхуки подій.
- Продукційний контейнер після стабілізації.


## Структура репозиторію

```
/src
  /cli            # Typer CLI
  /crawler        # інкрементальний обхід, обробка повідомлень, DeadLetter
  /graph          # модель графа каналів (edges)
  /tg_client      # Pyrogram клієнт, RPS-лімітер
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
cp .env.example .env && nano .env                    # заповніть TG_API_ID/TG_API_HASH/TG_PHONE_NUMBER
echo -e "voynareal_ua\nnebo_peremogy" > seeds.txt
python -m src.cli.main init
python -m src.cli.main add-seeds --file seeds.txt
python -m src.cli.main crawl
python -m src.cli.main export-manifest --out manifest.parquet
```

> **Docker App — не використовувати наразі.** Користуйтеся `pip`/`venv`.
