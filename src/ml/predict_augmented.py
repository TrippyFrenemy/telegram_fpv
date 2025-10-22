import os
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import torchvision.models as models

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns

from src.ml.frame_classifier import load_model

# --- Конфигурация ---
CHECKPOINT_PATH = "data/best_model_rn.pt"  # или fpv_frame_cls_torch.pt
TEST_MANIFEST = "data/frames_manifest.csv"  # полный датасет
TEST_CACHE_DIR = "data/frames_local"  # локальный кэш с кадрами
IMG_SIZE = (224, 224)
BATCH_SIZE = 64  # Можно больше для inference
NUM_WORKERS = 8
SEED = 42

# TTA (Test-Time Augmentation) конфигурация
TTA_ENABLED = False  # Включить/выключить TTA
TTA_NUM_CROPS = 5  # Количество рандомных кропов
TTA_FLIPS = True   # Горизонтальные flip'ы


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TTADataset(Dataset):
    """Dataset с множественными аугментациями для TTA"""
    
    def __init__(self, root, tta_transforms_list):
        self.base_dataset = datasets.ImageFolder(root=str(root))
        self.tta_transforms_list = tta_transforms_list
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # Применяем все TTA трансформации
        tta_images = []
        for tf in self.tta_transforms_list:
            tta_images.append(tf(img))
        
        # Возвращаем стек всех аугментаций
        return torch.stack(tta_images), label


def create_tta_transforms(img_size=IMG_SIZE, num_crops=5, enable_flips=True):
    """
    Создаёт список трансформаций для TTA:
    - Центральный crop
    - N рандомных кропов
    - Flip варианты
    - Вариации яркости/контраста
    """
    base_tf = [
        transforms.Resize(int(img_size[0] * 1.15)),  # Чуть больше для crop'ов
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    
    tta_list = []
    
    # 1. Центральный crop (baseline)
    tta_list.append(transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    
    # 2. Горизонтальный flip центрального crop
    if enable_flips:
        tta_list.append(transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
    
    # 3. Рандомные кропы (детерминированные через разные seeds)
    for i in range(num_crops):
        tta_list.append(transforms.Compose([
            transforms.Resize(int(img_size[0] * 1.15)),
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
    
    # 4. Вариации яркости (светлее/темнее)
    for brightness_factor in [0.2, 0.9]:
        tta_list.append(transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ColorJitter(brightness=1.0-brightness_factor),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
    
    # 5. Вариации контраста
    for contrast_factor in [0.2, 0.9]:
        tta_list.append(transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ColorJitter(contrast=1.0-contrast_factor),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
    
    # 6. Небольшие повороты
    for angle in [-5, 5]:
        tta_list.append(transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.RandomRotation(degrees=(angle, angle)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
    
    print(f"[TTA] Создано {len(tta_list)} трансформаций")
    return tta_list


@torch.no_grad()
def predict_with_tta(model, loader, device, tta_enabled=True):
    """
    Предсказания с TTA (усреднение по аугментациям)
    
    Returns:
        probs: np.array (N,) - вероятности класса 1
        labels: np.array (N,) - истинные метки
        logits_per_aug: list of arrays - логиты для каждой аугментации (для анализа)
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    all_logits_per_aug = defaultdict(list)
    
    for batch_imgs, batch_labels in tqdm(loader, desc="Inference"):
        # batch_imgs: (B, num_tta, C, H, W)
        # batch_labels: (B,)
        
        B, num_tta, C, H, W = batch_imgs.shape
        
        if tta_enabled and num_tta > 1:
            # Прогоняем каждую аугментацию через модель
            batch_imgs = batch_imgs.view(B * num_tta, C, H, W).to(device, non_blocking=True)
            logits = model(batch_imgs)  # (B * num_tta,)
            logits = logits.view(B, num_tta)  # (B, num_tta)
            
            # Сохраняем логиты каждой аугментации для анализа
            for aug_idx in range(num_tta):
                all_logits_per_aug[aug_idx].append(logits[:, aug_idx].cpu().numpy())
            
            # Усредняем логиты (можно также усреднять probs)
            logits_mean = logits.mean(dim=1)  # (B,)
            probs = torch.sigmoid(logits_mean).cpu().numpy()
        else:
            # Без TTA - берём первую трансформацию
            batch_imgs = batch_imgs[:, 0].to(device, non_blocking=True)
            logits = model(batch_imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        all_probs.append(probs)
        all_labels.append(batch_labels.numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Конвертируем logits_per_aug в массивы
    logits_per_aug = {k: np.concatenate(v) for k, v in all_logits_per_aug.items()}
    
    return all_probs, all_labels, logits_per_aug


def calculate_all_metrics(y_true, y_probs, threshold=0.5):
    """
    Вычисление ВСЕХ возможных метрик для бинарной классификации
    """
    y_pred = (y_probs >= threshold).astype(int)
    
    metrics = {}
    
    # Базовые метрики
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["specificity"] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    # AUC метрики
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_probs)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    
    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_probs)
    except ValueError:
        metrics["pr_auc"] = float("nan")
    
    # Confusion matrix элементы
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    
    # Дополнительные метрики
    metrics["matthews_corrcoef"] = matthews_corrcoef(y_true, y_pred)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    
    # NPV (Negative Predictive Value)
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    # Diagnostic odds ratio
    metrics["diagnostic_odds_ratio"] = (tp * tn) / (fp * fn) if (fp * fn) > 0 else float("inf")
    
    # Prevalence
    metrics["prevalence"] = np.mean(y_true)
    
    # Informedness (Youden's J statistic)
    metrics["informedness"] = metrics["recall"] + metrics["specificity"] - 1
    
    # Markedness
    metrics["markedness"] = metrics["precision"] + metrics["npv"] - 1
    
    return metrics


def find_optimal_threshold(y_true, y_probs):
    """Поиск оптимального порога по максимуму F1"""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1


def plot_all_metrics(y_true, y_probs, save_dir="data/test_results"):
    """Визуализация всех метрик"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. ROC Curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 2. Precision-Recall Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}", linewidth=2)
    plt.axhline(y=np.mean(y_true), color="k", linestyle="--", label=f"Baseline (prevalence={np.mean(y_true):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_dir / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 3. Confusion Matrix (оптимальный порог)
    optimal_thresh, _ = find_optimal_threshold(y_true, y_probs)
    y_pred = (y_probs >= optimal_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=["Class 0", "Class 1"],
                yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (threshold={optimal_thresh:.3f})")
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 4. Распределение вероятностей по классам
    plt.figure(figsize=(10, 6))
    plt.hist(y_probs[y_true == 0], bins=50, alpha=0.6, label="Class 0 (negative)", color="blue", density=True)
    plt.hist(y_probs[y_true == 1], bins=50, alpha=0.6, label="Class 1 (positive)", color="red", density=True)
    plt.axvline(x=optimal_thresh, color="green", linestyle="--", linewidth=2, label=f"Optimal threshold={optimal_thresh:.3f}")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.title("Probability Distribution by Class")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_dir / "probability_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 5. F1 Score vs Threshold
    thresholds_f1 = np.linspace(0.1, 0.9, 81)
    f1_scores = [f1_score(y_true, (y_probs >= t).astype(int), zero_division=0) for t in thresholds_f1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_f1, f1_scores, linewidth=2)
    plt.axvline(x=optimal_thresh, color="red", linestyle="--", label=f"Optimal={optimal_thresh:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Classification Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_dir / "f1_vs_threshold.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"[plots] Все графики сохранены в {save_dir}")


def analyze_tta_variance(logits_per_aug, y_true, save_dir="data/test_results"):
    """Анализ вариации предсказаний между аугментациями"""
    if not logits_per_aug:
        return
    
    save_dir = Path(save_dir)
    
    # Собираем все логиты в матрицу (N, num_aug)
    logits_matrix = np.stack([logits_per_aug[i] for i in sorted(logits_per_aug.keys())], axis=1)
    probs_matrix = 1 / (1 + np.exp(-logits_matrix))  # sigmoid
    
    # Вычисляем std по аугментациям для каждого примера
    std_per_sample = probs_matrix.std(axis=1)
    
    # Группируем по классам
    std_class_0 = std_per_sample[y_true == 0]
    std_class_1 = std_per_sample[y_true == 1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(std_class_0, bins=50, alpha=0.6, label="Class 0", color="blue", density=True)
    plt.hist(std_class_1, bins=50, alpha=0.6, label="Class 1", color="red", density=True)
    plt.xlabel("Std Dev of Predictions across TTA")
    plt.ylabel("Density")
    plt.title("TTA Prediction Variance by Class")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_dir / "tta_variance.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"[TTA analysis] Средняя вариация: class_0={std_class_0.mean():.4f}, class_1={std_class_1.mean():.4f}")


def main():
    set_seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    
    # 1. Загрузка модели
    if not Path(CHECKPOINT_PATH).exists():
        print(f"[ERROR] Checkpoint не найден: {CHECKPOINT_PATH}")
        return
    
    model, config = load_model(CHECKPOINT_PATH, device)
    
    # 2. Подготовка TTA трансформаций
    tta_transforms = create_tta_transforms(
        img_size=IMG_SIZE,
        num_crops=TTA_NUM_CROPS,
        enable_flips=TTA_FLIPS
    ) if TTA_ENABLED else [transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])]
    
    # 3. Загрузка тестового датасета
    print(f"[data] Загрузка из {TEST_CACHE_DIR}")
    
    # Если нужно скачать данные (как в train)
    # from frame_classifier import sync_subset
    # test_root = sync_subset(TEST_MANIFEST, TEST_CACHE_DIR, max_per_class=999999)
    
    test_root = Path(TEST_CACHE_DIR)
    if not test_root.exists():
        print(f"[ERROR] Тестовый датасет не найден: {test_root}")
        print("Запустите сначала sync_subset() для скачивания данных")
        return
    
    test_dataset = TTADataset(test_root, tta_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"[data] Тестовых примеров: {len(test_dataset)}")
    
    # 4. Inference с TTA
    print(f"[inference] Запуск предсказаний (TTA={'enabled' if TTA_ENABLED else 'disabled'})...")
    t0 = time.time()
    
    y_probs, y_true, logits_per_aug = predict_with_tta(
        model, test_loader, device, tta_enabled=TTA_ENABLED
    )
    
    inference_time = time.time() - t0
    print(f"[inference] Завершено за {inference_time:.1f}s ({len(y_true)/inference_time:.1f} img/s)")
    
    # 5. Вычисление метрик
    print("\n" + "="*60)
    print("МЕТРИКИ (threshold=0.5)")
    print("="*60)
    
    metrics = calculate_all_metrics(y_true, y_probs, threshold=0.5)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:30s}: {v:.4f}")
        else:
            print(f"{k:30s}: {v}")
    
    # 6. Оптимальный порог
    optimal_thresh, optimal_f1 = find_optimal_threshold(y_true, y_probs)
    print(f"\n{'='*60}")
    print(f"ОПТИМАЛЬНЫЙ ПОРОГ (по max F1)")
    print(f"{'='*60}")
    print(f"Threshold: {optimal_thresh:.4f}")
    print(f"F1 Score : {optimal_f1:.4f}")
    
    metrics_optimal = calculate_all_metrics(y_true, y_probs, threshold=optimal_thresh)
    print(f"Accuracy : {metrics_optimal['accuracy']:.4f}")
    print(f"Precision: {metrics_optimal['precision']:.4f}")
    print(f"Recall   : {metrics_optimal['recall']:.4f}")
    
    # 7. Classification report (детальный)
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT (threshold=0.5)")
    print(f"{'='*60}")
    y_pred = (y_probs >= 0.5).astype(int)
    print(classification_report(y_true, y_pred, target_names=["Class 0", "Class 1"], digits=4))
    
    # 8. Визуализация
    print(f"\n[plots] Создание графиков...")
    plot_all_metrics(y_true, y_probs, save_dir="data/test_results")
    
    # 9. Анализ TTA вариации
    if TTA_ENABLED and logits_per_aug:
        analyze_tta_variance(logits_per_aug, y_true, save_dir="data/test_results")
    
    # 10. Сохранение результатов в JSON
    results = {
        "config": {
            "checkpoint": str(CHECKPOINT_PATH),
            "test_dataset": str(TEST_CACHE_DIR),
            "num_samples": int(len(y_true)),
            "tta_enabled": TTA_ENABLED,
            "tta_num_augmentations": len(tta_transforms),
            "model_config": config,
        },
        "metrics_threshold_0.5": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                                    for k, v in metrics.items()},
        "optimal_threshold": {
            "threshold": float(optimal_thresh),
            "f1_score": float(optimal_f1),
            "metrics": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                        for k, v in metrics_optimal.items()},
        },
        "inference_time_seconds": inference_time,
        "throughput_img_per_sec": len(y_true) / inference_time,
    }
    
    results_path = Path("data/test_results/results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[save] Результаты сохранены в {results_path}")
    print(f"\n{'='*60}")
    print("ЗАВЕРШЕНО")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
