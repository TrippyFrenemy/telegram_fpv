import hashlib
import os
import time

import numpy as np
from collections import Counter
from pathlib import Path
import shutil

from minio import S3Error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.models as models

import matplotlib.pyplot as plt

from src.config import settings
from src.store.s3 import client as s3

# --- константы ---
BUCKET = settings.s3_bucket
BASE_LR = 3e-4  # Поднято с 1e-6
WARMUP_EPOCHS = 5
EPOCHS = 25
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
MANIFEST_PATH = "data/frames_manifest.csv"
BEST_DATASET_PATH = "data/best_model_30k.pt"
DATASET_PATH = "data/fpv_frame_cls_torch.pt"
MAX_PER_CLASS = 100_000_000

def _device_info():
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        return {
            "device": "cuda",
            "name": torch.cuda.get_device_name(0),
            "cap": f"{torch.cuda.get_device_capability(0)}",
            "mem_total_gb": round(total/1024**3, 2),
        }
    return {"device": "cpu"}

# --- утилиты ---
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sync_subset(manifest_path: str, out_dir: str, max_per_class: int = MAX_PER_CLASS):
    df = pd.read_parquet(manifest_path) if manifest_path.endswith(".parquet") else pd.read_csv(manifest_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for cls in (0, 1):
        sub = df[df["label"] == cls].sample(n=min(max_per_class, (df["label"] == cls).sum()), random_state=SEED)
        dst = out / str(cls)
        dst.mkdir(parents=True, exist_ok=True)
        for _, r in sub.iterrows():
            key = r["s3_path"]
            fname = key.split("/")[-1]
            loc = dst / fname
            if not loc.exists():
                try:
                    s3.fget_object(BUCKET, key, str(loc))
                except S3Error:
                    continue
    return out

def build_loaders(root: str | Path, img_size=IMG_SIZE, batch=BATCH_SIZE):
    """
    DataLoader с ImageNet нормализацией, persistent workers, prefetch
    """
    # Аугментации для train (добавлен ColorJitter, RandomResizedCrop)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # Вместо Resize
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base = datasets.ImageFolder(root=str(root))
    targets = np.array(base.targets)
    
    idx_train, idx_val = train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        random_state=SEED,
        stratify=targets
    )

    ds_train = datasets.ImageFolder(root=str(root), transform=train_tf)
    ds_val = datasets.ImageFolder(root=str(root), transform=val_tf)
    ds_train = Subset(ds_train, idx_train.tolist())
    ds_val = Subset(ds_val, idx_val.tolist())

    # Подсчёт классов на train для pos_weight
    y_train = targets[idx_train]
    cnt = Counter(y_train.tolist())
    neg, pos = cnt.get(0, 0), cnt.get(1, 0)
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)

    num_workers = min(8, os.cpu_count() or 4)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        ds_train, 
        batch_size=batch, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        ds_val, 
        batch_size=batch, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4
    )
    
    return train_loader, val_loader, pos_weight, cnt
    
def build_model(num_classes=1, pretrained=True, freeze_backbone_epochs=5):
    """
    ResNet18 с pretrained весами (вместо SmallCNN)
    """
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, num_classes),
        nn.Flatten(0)  # (B, 1) -> (B,)
    )
    return model

def unfreeze_backbone(model):
    """Разморозить весь backbone после warmup"""
    for param in model.parameters():
        param.requires_grad = True
    print("[model] Backbone разморожен")

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, epoch=0):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    
    t0 = time.perf_counter()
    
    for i, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, labels)
            
            # Проверка на NaN/inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] epoch={epoch}, batch={i}: loss={loss.item()}")
                continue
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            total_correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
        
        # Лог первого батча для отладки I/O
        if i == 0:
            print(f"[timing] first batch: {time.perf_counter()-t0:.2f}s")

    acc = total_correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)
    epoch_time = time.perf_counter() - t0
    
    return avg_loss, acc, epoch_time

def load_model(checkpoint_path, device):
    """Загрузка обученной модели"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Определяем архитектуру из checkpoint (если сохранена)
    config = checkpoint.get("config", {})
    model_arch = config.get("model", "resnet18")
    
    print(f"[model] Загрузка {model_arch} из {checkpoint_path}")
    
    # Создаём модель
    if model_arch == "resnet18":
        base_model = models.resnet18(weights=None)
        # Оборачиваем в nn.Sequential, чтобы соответствовать структуре обучения
        base_model.fc = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 1)
        )
    elif model_arch == "resnet34":
        base_model = models.resnet34(weights=None)
        # Оборачиваем в nn.Sequential, чтобы соответствовать структуре обучения
        base_model.fc = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 1)
        )
    elif model_arch == "mobilenet_v3_large":
        base_model = models.mobilenet_v3_large(weights=None)
        base_model.classifier[3] = nn.Linear(base_model.classifier[3].in_features, 1)
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")
    
    # Загружаем веса
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Проверяем есть ли обёртка в state_dict
    has_wrapper = any(k.startswith("model.") for k in state_dict.keys())
    
    if has_wrapper:
        # Чекпоинт сохранён с BinaryWrapper
        class BinaryWrapper(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.model = base
            
            def forward(self, x):
                out = self.model(x)
                return out.squeeze(1) if out.dim() > 1 and out.size(1) == 1 else out
        
        model = BinaryWrapper(base_model)
        model.load_state_dict(state_dict)
    else:
        # Чекпоинт сохранён БЕЗ обёртки (напрямую ResNet)
        # Убираем префикс "model." если есть (старые чекпоинты)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned_state_dict[k[6:]] = v  # Убираем "model."
            else:
                cleaned_state_dict[k] = v
        
        base_model.load_state_dict(cleaned_state_dict)
        
        # Создаём обёртку для squeeze
        class BinaryWrapper(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.model = base
            
            def forward(self, x):
                out = self.model(x)
                return out.squeeze(1) if out.dim() > 1 and out.size(1) == 1 else out
        
        model = BinaryWrapper(base_model)
    
    model.to(device)
    model.eval()
    
    print(f"[model] Модель загружена, параметры: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model, config

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        preds = (probs >= 0.5).long()
        total_correct += (preds == labels.long()).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    acc = total_correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)
    
    try:
        auc = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_probs))
    except Exception:
        auc = float("nan")
    
    return avg_loss, acc, auc

def plot_history(history, out_path="data/training_validation_plots.png"):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8))

    ax1.set_title("Loss")
    ax1.plot(history["loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    ax1.set_ylabel("Loss")
    max_loss = max(history["loss"] + history["val_loss"])
    ax1.set_ylim([0, np.ceil(max_loss)])
    ax1.set_xlabel("Epoch")
    ax1.legend(["Train", "Validation"])

    ax2.set_title("Accuracy")
    ax2.plot(history["acc"], label="train")
    ax2.plot(history["val_acc"], label="val")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1])
    ax2.set_xlabel("Epoch")
    ax2.legend(["Train", "Validation"])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[plot] Сохранено в {out_path}")

def main(manifest="data/frames_manifest.csv", cache_dir="data/frames_local", epochs=EPOCHS):
    set_seed(SEED)
    device_info = _device_info()
    print(f"[info] device: {device_info}")
    
    # Загрузка данных
    root = Path("D:\\Projects\\Python\\telegram_fpv\\data\\frames_local") # sync_subset(manifest, cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    train_loader, val_loader, pos_weight, cnt = build_loaders(root, IMG_SIZE, BATCH_SIZE)
    print(f"[data] train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, class_dist={cnt}")

    # Модель: ResNet18 pretrained
    model = build_model(num_classes=1, pretrained=True).to(device)
    
    # Оптимизатор и scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)
    
    # Cosine scheduler с warmup
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": [], "val_auc": [], "epoch_time": []}
    
    # Early stopping
    best_auc = 0.0
    patience = 7
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc, epoch_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["loss"].append(train_loss)
        history["acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["epoch_time"].append(epoch_time)

        print(f"epoch {epoch+1:02d}/{epochs} "
              f"lr={optimizer.param_groups[0]['lr']:.2e} "
              f"loss={train_loss:.4f} acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f} "
              f"time={epoch_time:.1f}s")
        
        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "auc": best_auc,
            }, BEST_DATASET_PATH)
            print(f"  [checkpoint] Лучшая модель сохранена (AUC={best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[early_stop] Остановка на эпохе {epoch+1}, лучший AUC={best_auc:.4f}")
                break

    # Сохранение финальной модели с метаданными
    Path("data").mkdir(parents=True, exist_ok=True)
    
    manifest_hash = hashlib.md5(open(manifest, "rb").read()).hexdigest()
    
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "model": "resnet18",
            "pretrained": True,
            "epochs": epochs,
            "lr": BASE_LR,
            "batch_size": BATCH_SIZE,
            "img_size": IMG_SIZE,
            "seed": SEED,
            "classes": ("0", "1"),
            "best_val_auc": best_auc,
        },
        "manifest_hash": manifest_hash,
        "pytorch_version": torch.__version__,
    }, DATASET_PATH)
    
    print(f"[save] Модель сохранена в data/fpv_frame_cls_torch.pt")
    print(f"[info] Лучший val_auc={best_auc:.4f}")
    print(f"[info] Среднее время эпохи: {np.mean(history['epoch_time']):.1f}s")

    plot_history(history, "data/training_validation_plots.png")

if __name__ == "__main__":
    main()
    