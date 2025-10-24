import os
import io
import time
import tempfile
import subprocess
import signal
import hashlib
from pathlib import Path
from typing import List, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from src.config import settings
from src.store.storage_backend import get_storage_backend
from src.store.redis import _client as redis
from src.logger import log
from src.ml.frame_classifier import load_model


# Redis ключі
ML_QUEUE = "fpv:ml_queue"
ML_PROCESSING = "fpv:processing"
ML_STATS = "fpv:ml_stats"

# Глобальна змінна для graceful shutdown
_shutdown_requested = False


def extract_frames_ffmpeg(video_path: str, fps: float) -> List[bytes]:
    """
    Витягує кадри з відео за допомогою ffmpeg.
    
    Args:
        video_path: Шлях до відео файлу
        fps: Кількість кадрів на секунду
        
    Returns:
        Список байтів JPEG-кодованих кадрів
    """
    tmpdir = Path(tempfile.mkdtemp())
    pattern = tmpdir / "frame_%08d.jpg"
    
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-r", str(fps),  # Частота кадрів
            "-q:v", "2",     # Якість JPEG (2 = високо)
            str(pattern)
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            timeout=settings.encode_timeout_s
        )
        
        if result.returncode != 0:
            log.error("ffmpeg_extraction_failed", 
                     stderr=result.stderr.decode()[:500])
            return []
        
        # Читаємо згенеровані кадри
        frames = []
        for jpg_path in sorted(tmpdir.glob("frame_*.jpg")):
            frames.append(jpg_path.read_bytes())
            
        log.info("frames_extracted", count=len(frames), fps=fps)
        return frames
        
    except subprocess.TimeoutExpired:
        log.error("ffmpeg_timeout", video_path=video_path)
        return []
    except Exception as e:
        log.error("frame_extraction_error", error=str(e))
        return []
    finally:
        # Очищення тимчасових файлів
        for f in tmpdir.glob("*"):
            try:
                f.unlink()
            except:
                pass
        try:
            tmpdir.rmdir()
        except:
            pass


class FrameClassifier:
    """
    Обгортка навколо ML моделі для класифікації кадрів.
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Шлях до checkpoint моделі
            device: cuda/cpu/auto
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        log.info("loading_model", path=model_path, device=str(self.device))
        
        self.model, self.config = load_model(model_path, self.device)
        self.model.eval()
        
        # Transform для кадрів (ImageNet нормалізація)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        log.info("model_loaded", 
                params=sum(p.numel() for p in self.model.parameters()) / 1e6)
    
    @torch.no_grad()
    def classify_frames(self, frame_bytes: List[bytes]) -> List[float]:
        """
        Класифікує список кадрів.
        
        Args:
            frame_bytes: Список JPEG байтів
            
        Returns:
            Список ймовірностей класу 1 (FPV підходить)
        """
        if not frame_bytes:
            return []
            
        # Конвертуємо байти в тензори
        frames = []
        for fb in frame_bytes:
            try:
                img = Image.open(io.BytesIO(fb)).convert("RGB")
                tensor = self.transform(img)
                frames.append(tensor)
            except Exception as e:
                log.warning("frame_conversion_failed", error=str(e))
                continue
        
        if not frames:
            return []
        
        # Batch inference
        batch = torch.stack(frames).to(self.device)
        
        logits = self.model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        
        return probs.tolist()
    
    def classify_in_batches(
        self, 
        frame_bytes: List[bytes], 
        batch_size: int
    ) -> List[float]:
        """
        Класифікує кадри по батчах для економії памʼяті.
        
        Args:
            frame_bytes: Список JPEG байтів
            batch_size: Розмір батча
            
        Returns:
            Список ймовірностей
        """
        all_probs = []
        
        for i in range(0, len(frame_bytes), batch_size):
            batch = frame_bytes[i:i + batch_size]
            probs = self.classify_frames(batch)
            all_probs.extend(probs)
            
        return all_probs


def check_positive_sequence(
    probs: List[float], 
    threshold: float = 0.5,
    required_consecutive: int = None
) -> bool:
    """
    Перевіряє чи є у відео N підряд позитивних кадрів.
    
    Args:
        probs: Список ймовірностей для кожного кадру
        threshold: Поріг для класифікації як позитивний
        required_consecutive: Кількість підряд позитивних (з config)
        
    Returns:
        True якщо знайдено потрібну послідовність
    """
    if required_consecutive is None:
        required_consecutive = settings.ml_positive_threshold
    
    consecutive_count = 0
    
    for prob in probs:
        if prob >= threshold:
            consecutive_count += 1
            if consecutive_count >= required_consecutive:
                return True
        else:
            consecutive_count = 0
    
    return False


def move_to_dataset(storage_path: str, sha256: str, label: int) -> bool:
    """
    Переміщує відео з telegram/ в dataset/{0,1}/.
    REFACTORED: Uses storage backend instead of direct S3.
    
    Args:
        storage_path: Поточний шлях в storage
        sha256: SHA-256 хеш
        label: 0 або 1
        
    Returns:
        True якщо успішно перемістили
    """
    try:
        storage = get_storage_backend()
        dst_path = f"dataset/{label}/{sha256}.mp4"
        
        # Перевіряємо чи вже існує в dataset
        if storage.exists(dst_path):
            log.info("already_in_dataset", dst_path=dst_path)
            # Видаляємо оригінал
            storage.delete(storage_path)
            return True
        
        # Копіюємо в dataset
        storage.copy(storage_path, dst_path)
        
        # Видаляємо оригінал з telegram/
        storage.delete(storage_path)
        
        log.info("moved_to_dataset", 
                src=storage_path, dst=dst_path, label=label)
        return True
        
    except Exception as e:
        log.error("move_failed", error=str(e), storage_path=storage_path)
        return False


def process_video(
    storage_path: str, 
    sha256: str, 
    duration_s: float,
    classifier: FrameClassifier
) -> Tuple[bool, dict]:
    """
    Обробляє одне відео: витягує кадри, класифікує, приймає рішення.
    REFACTORED: Uses storage backend instead of direct S3.
    
    Args:
        storage_path: Шлях до відео в storage
        sha256: SHA-256 хеш
        duration_s: Тривалість відео
        classifier: Екземпляр FrameClassifier
        
    Returns:
        (success: bool, stats: dict)
    """
    start_time = time.time()
    storage = get_storage_backend()
    
    try:
        # 1. Завантажуємо відео зі storage у тимчасовий файл
        tmp_video = Path(tempfile.gettempdir()) / f"{sha256}.mp4"
        
        log.info("downloading_video", storage_path=storage_path)
        storage.download_to_file(storage_path, str(tmp_video))
        
        # 2. Витягуємо кадри
        fps = settings.ml_frames_per_second
        frames = extract_frames_ffmpeg(str(tmp_video), fps=fps)
        
        if not frames:
            log.warning("no_frames_extracted", storage_path=storage_path)
            tmp_video.unlink(missing_ok=True)
            return False, {"error": "no_frames"}
        
        # 3. Класифікуємо кадри
        log.info("classifying_frames", count=len(frames))
        probs = classifier.classify_in_batches(
            frames, 
            batch_size=settings.ml_batch_size
        )
        
        if not probs:
            log.warning("classification_failed", storage_path=storage_path)
            tmp_video.unlink(missing_ok=True)
            return False, {"error": "classification_failed"}
        
        # 4. Перевіряємо послідовність позитивних кадрів
        is_positive = check_positive_sequence(probs)
        
        # Статистика
        positive_count = sum(1 for p in probs if p >= 0.5)
        avg_prob = np.mean(probs)
        max_prob = np.max(probs)
        
        stats = {
            "frames_total": len(frames),
            "frames_positive": positive_count,
            "avg_probability": float(avg_prob),
            "max_probability": float(max_prob),
            "decision": 1 if is_positive else 0,
            "processing_time": time.time() - start_time,
        }
        
        # 5. Переміщуємо в dataset якщо підходить
        label = 1 if is_positive else 0
        move_success = move_to_dataset(storage_path, sha256, label)
        
        # Очищення
        tmp_video.unlink(missing_ok=True)
        
        log.info("video_processed", 
                storage_path=storage_path, 
                label=label,
                **stats)
        
        return move_success, stats
        
    except Exception as e:
        log.error("processing_error", error=str(e), storage_path=storage_path)
        return False, {"error": str(e)}


def update_stats(stats: dict) -> None:
    """
    Оновлює глобальну статистику в Redis.
    
    Args:
        stats: Словник зі статистикою обробки
    """
    try:
        # Інкрементуємо лічильники
        redis.hincrby(ML_STATS, "total_processed", 1)
        
        if stats.get("decision") == 1:
            redis.hincrby(ML_STATS, "positive_videos", 1)
        else:
            redis.hincrby(ML_STATS, "negative_videos", 1)
        
        # Зберігаємо середній час обробки
        current_avg = float(redis.hget(ML_STATS, "avg_processing_time") or 0)
        total = int(redis.hget(ML_STATS, "total_processed") or 1)
        
        new_avg = (current_avg * (total - 1) + stats["processing_time"]) / total
        redis.hset(ML_STATS, "avg_processing_time", new_avg)
        
    except Exception as e:
        log.error("stats_update_failed", error=str(e))


def worker_loop():
    """
    Основний цикл ML воркера.
    
    - Слухає Redis чергу
    - Обробляє відео
    - Оновлює статистику
    """
    global _shutdown_requested
    
    log.info("ml_worker_started", 
            model_path=settings.ml_model_path,
            fps=settings.ml_frames_per_second,
            positive_threshold=settings.ml_positive_threshold)
    
    # Ініціалізація класифікатора
    classifier = FrameClassifier(settings.ml_model_path)
    
    # Статистика воркера
    processed_count = 0
    error_count = 0
    
    while not _shutdown_requested:
        try:
            # Блокуюче читання з черги (timeout 5 сек)
            result = redis.blpop(ML_QUEUE, timeout=15)
            
            if result is None:
                # Timeout - продовжуємо цикл
                continue
            
            _, task_data = result
            task_data = task_data.decode() if isinstance(task_data, bytes) else task_data
            
            # Парсимо дані задачі: "storage_path|sha256|duration"
            try:
                storage_path, sha256, duration_str = task_data.split("|")
                duration = 0.0
            except ValueError:
                log.error("invalid_task_format", task=task_data)
                continue
            
            # Перевіряємо чи не обробляється вже
            if redis.sismember(ML_PROCESSING, storage_path):
                log.warning("already_processing", storage_path=storage_path)
                continue
            
            # Додаємо в set активних
            redis.sadd(ML_PROCESSING, storage_path)
            
            try:
                # Обробляємо відео
                success, stats = process_video(
                    storage_path, sha256, duration, classifier
                )
                
                if success:
                    processed_count += 1
                    update_stats(stats)
                else:
                    error_count += 1
                
            finally:
                # Видаляємо з активних
                redis.srem(ML_PROCESSING, storage_path)
            
            # Періодичний лог статистики
            if (processed_count + error_count) % 10 == 0:
                log.info("worker_stats", 
                        processed=processed_count,
                        errors=error_count,
                        queue_size=redis.llen(ML_QUEUE))
        
        except KeyboardInterrupt:
            log.info("worker_interrupted")
            break
        except Exception as e:
            log.error("worker_loop_error", error=str(e))
            error_count += 1
            time.sleep(1)  # Пауза при помилці
    
    log.info("ml_worker_stopped", 
            total_processed=processed_count,
            total_errors=error_count)


def signal_handler(signum, frame):
    """Обробник сигналів для graceful shutdown."""
    global _shutdown_requested
    log.info("shutdown_signal_received", signal=signum)
    _shutdown_requested = True


def main():
    """Точка входу для ML воркера."""
    # Реєстрація обробників сигналів
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Перевірка моделі
    model_path = Path(settings.ml_model_path)
    if not model_path.exists():
        log.error("model_not_found", path=str(model_path))
        print(f"ERROR: Модель не знайдено за шляхом: {model_path}")
        print("Запустіть спочатку навчання моделі:")
        print("  python -m src.ml.frame_classifier")
        return
    
    # Ініціалізація статистики
    if not redis.exists(ML_STATS):
        redis.hset(ML_STATS, "total_processed", 0)
        redis.hset(ML_STATS, "positive_videos", 0)
        redis.hset(ML_STATS, "negative_videos", 0)
        redis.hset(ML_STATS, "avg_processing_time", 0)
    
    # Запуск основного циклу
    worker_loop()


if __name__ == "__main__":
    main()
