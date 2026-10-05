"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.
Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
"""
from __future__ import annotations

import os
import sys
import time
import copy
import json
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR

# Tìm và import eval.py, dataset, model, losses
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(current_dir))

from eval import save_predictions, compute_metrics
import dataset as ds
import model as mdl
import losses as lss


def softmax(z: np.ndarray) -> np.ndarray:
    """Tính xác suất softmax ổn định số học từ logit numpy."""
    z_max = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - z_max)
    return e / np.sum(e, axis=-1, keepdims=True)


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 32              # Phù hợp GPU 4GB VRAM local và T4 Colab
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW với 3 nhóm tham số."""
    groups = mdl.param_groups(
        model=model,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        weight_decay=cfg.weight_decay
    )
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int) -> LambdaLR:
    """Warmup tuyến tính rồi cosine về gần 0."""
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(1e-6, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số (Model Exponential Moving Average)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for p_ema, p in zip(self.ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        for b_ema, b in zip(self.ema_model.buffers(), model.buffers()):
            b_ema.data.copy_(b.data)


def train_one_epoch(model: nn.Module, loader, criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: LambdaLR, scaler: GradScaler, cfg: Config,
                    device: torch.device, ema: Optional[EMA] = None) -> Dict[str, float]:
    """Huấn luyện một epoch."""
    model.train()
    if cfg.init == "frozen":
        mdl.freeze_backbone(model)

    running_loss = 0.0
    total_samples = 0

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        # Mixup / CutMix
        if cfg.mix and cfg.mix != "none":
            images_mixed, mix_targets = lss.mix_batch(images, targets, alpha=cfg.mix_alpha, mode=cfg.mix)
            with autocast("cuda", enabled=cfg.amp):
                logits = model(images_mixed)
                loss = lss.mixed_loss(criterion, logits, mix_targets)
        else:
            with autocast("cuda", enabled=cfg.amp):
                logits = model(images)
                loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / max(1, total_samples)
    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": epoch_loss, "lr": current_lr}


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module,
             device: torch.device) -> Tuple[List[str], np.ndarray, np.ndarray, float]:
    """Đánh giá model trên một loader."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_filenames = []
    all_targets = []
    all_logits = []

    for images, targets, filenames in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_filenames.extend(filenames)
        all_targets.append(targets.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

    mean_loss = running_loss / max(1, total_samples)
    y_true = np.concatenate(all_targets, axis=0)
    logits_arr = np.concatenate(all_logits, axis=0)

    return all_filenames, y_true, logits_arr, mean_loss


def plot_curves(history: List[Dict[str, Any]], path: str | Path, title: str) -> None:
    """Vẽ đường cong training và lưu vào curves/."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]
    val_acc = [h["val_top1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss
    ax1.plot(epochs, train_loss, label="Train Loss", marker="o", markersize=3)
    ax1.plot(epochs, val_loss, label="Val Loss", marker="s", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss Curves: {title}")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Metrics
    ax2.plot(epochs, val_f1, label="Val Macro-F1", color="green", marker="^", markersize=3)
    ax2.plot(epochs, val_acc, label="Val Top-1 Acc", color="orange", marker="d", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.set_title(f"Val Metrics: {title}")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def run(cfg: Config) -> Dict[str, Any]:
    """Quy trình huấn luyện dùng chung cho mọi cấu hình."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tạo thư mục đầu ra
    r_dir = run_dir(cfg)
    r_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.curves_dir).mkdir(parents=True, exist_ok=True)

    with open(r_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # 1. Đọc và kiểm tra split
    train_df, val_df, test_df = ds.load_split(cfg.labels_dir, fold=cfg.fold)
    ds.check_split(train_df, val_df, test_df, cfg.images_dir)

    # 2. Tạo DataLoaders
    train_trans = ds.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    eval_trans = ds.build_transforms(train=False, img_size=cfg.img_size, aug="basic")

    train_loader = ds.make_loader(train_df, cfg.images_dir, train_trans, cfg.batch_size, train=True, sampler=cfg.sampler, num_workers=cfg.num_workers)
    val_loader = ds.make_loader(val_df, cfg.images_dir, eval_trans, cfg.batch_size, train=False, num_workers=cfg.num_workers)

    # 3. Xây dựng Model
    model = mdl.build_model(
        name=cfg.backbone,
        pretrained=True,
        num_classes=ds.NUM_CLASSES,
        drop_rate=cfg.drop_rate,
        init=cfg.init
    ).to(device)

    # 4. Xây dựng Loss
    if cfg.loss == "ce_weighted":
        c_weights = lss.class_weights(train_df["Label"].value_counts().to_dict(), beta=cfg.class_weight_beta or 0.0).to(device)
        criterion = lss.build_criterion("ce_weighted", weight=c_weights)
    elif cfg.loss == "ls":
        criterion = lss.build_criterion("ls", smoothing=cfg.label_smoothing or 0.1)
    elif cfg.loss == "focal":
        criterion = lss.build_criterion("focal", gamma=cfg.focal_gamma)
    else:
        criterion = lss.build_criterion("ce")

    val_criterion = nn.CrossEntropyLoss()

    # 5. Optimizer & Scheduler
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = GradScaler("cuda", enabled=cfg.amp)
    ema = EMA(model, decay=cfg.ema_decay) if (cfg.ema_decay is not None) else None

    history = []
    best_f1 = -1.0
    best_epoch = -1
    best_weights = None
    epoch_times = []

    print(f"\n=======================================================")
    print(f"Bắt đầu thí nghiệm: {cfg.exp_id} | Backbone: {cfg.backbone} | Seed: {cfg.seed}")
    print(f"Params: {mdl.count_params(model)}M | GMACs: {mdl.count_gmacs(model, cfg.img_size)}")
    print(f"=======================================================")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        t_epoch = time.time() - t0
        epoch_times.append(t_epoch)

        # Đánh giá Val (dùng model EMA nếu có)
        eval_model = ema.ema_model if (ema is not None) else model
        filenames_val, y_val, logits_val, val_loss = evaluate(eval_model, val_loader, val_criterion, device)
        probs_val = softmax(logits_val)
        val_metrics = compute_metrics(probs_val, y_val)

        val_macro_f1 = val_metrics["macro_f1"]
        val_top1 = val_metrics["top1"]

        history.append({
            "epoch": epoch,
            "train_loss": train_res["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "val_top1": val_top1,
            "lr": train_res["lr"],
            "time_sec": t_epoch
        })

        print(f"Epoch {epoch:02d}/{cfg.epochs:02d} | Train Loss: {train_res['train_loss']:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_macro_f1:.4f} | Val Acc: {val_top1*100:.2f}% | {t_epoch:.1f}s")

        # Lưu checkpoint theo macro-F1 val (hòa lấy epoch sớm hơn)
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            best_epoch = epoch
            best_weights = copy.deepcopy(eval_model.state_dict())
            torch.save(best_weights, r_dir / "best_model.pt")

    # Nạp lại checkpoint tốt nhất để xuất dự đoán
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Xuất dự đoán Val
    fnames_val, y_val, logits_val, _ = evaluate(model, val_loader, val_criterion, device)
    probs_val = softmax(logits_val)
    save_predictions(pred_path(cfg, "val"), fnames_val, y_val, probs_val)

    # Nếu được phép (Bước 4): Chạy Test đúng 1 lần
    test_metrics = None
    if cfg.save_test_predictions:
        print(f"--> Đang chạy dự đoán trên tập TEST (Seed {cfg.seed})...")
        test_loader = ds.make_loader(test_df, cfg.images_dir, eval_trans, cfg.batch_size, train=False, num_workers=cfg.num_workers)
        fnames_test, y_test, logits_test, _ = evaluate(model, test_loader, val_criterion, device)
        probs_test = softmax(logits_test)
        
        # Nếu là chung kết F01: lưu bản uncalibrated và bản calibrated sau Temperature Scaling
        if cfg.exp_id.startswith("F"):
            import inference as inf
            uncal_path = Path(cfg.pred_dir) / f"{cfg.exp_id}_uncal_seed{cfg.seed}_test.csv"
            save_predictions(uncal_path, fnames_test, y_test, probs_test)
            
            # Khớp T trên Val (chuẩn S2, S4) và áp dụng sang Test
            T_val = inf.fit_temperature(logits_val, y_val)
            probs_test_cal = inf.apply_temperature(logits_test, T_val)
            save_predictions(pred_path(cfg, "test"), fnames_test, y_test, probs_test_cal)
            test_metrics = compute_metrics(probs_test_cal, y_test)
            print(f"--> KẾT QUẢ TEST (Calibrated T={T_val:.4f}): Top-1 Acc: {test_metrics['top1']*100:.2f}% | Macro-F1: {test_metrics['macro_f1']:.4f}")
        else:
            save_predictions(pred_path(cfg, "test"), fnames_test, y_test, probs_test)
            test_metrics = compute_metrics(probs_test, y_test)
            print(f"--> KẾT QUẢ TEST: Top-1 Acc: {test_metrics['top1']*100:.2f}% | Macro-F1: {test_metrics['macro_f1']:.4f}")

    # Lưu history và biểu đồ
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(r_dir / "history.csv", index=False)
    plot_path = Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, plot_path, title=f"{cfg.exp_id} ({cfg.backbone})")

    summary = {
        "exp_id": cfg.exp_id,
        "backbone": cfg.backbone,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "val_macro_f1": best_f1,
        "val_top1": history[best_epoch - 1]["val_top1"] if best_epoch > 0 else 0.0,
        "avg_train_time_per_epoch": float(np.mean(epoch_times)),
        "params_m": mdl.count_params(model),
        "gmacs": mdl.count_gmacs(model, cfg.img_size),
        "test_metrics": test_metrics
    }
    return summary


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """Chuyển ['backbone=resnet50', 'seed=1'] thành dict ép đúng kiểu field của Config."""
    cfg_defaults = asdict(Config())
    res = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k not in cfg_defaults:
            raise KeyError(f"Trường không hợp lệ trong Config: {k}")

        orig_val = cfg_defaults[k]
        if orig_val is None:
            # Đoán kiểu nếu giá trị mặc định là None
            if v.lower() == "none":
                res[k] = None
            elif v.isdigit():
                res[k] = int(v)
            else:
                try:
                    res[k] = float(v)
                except ValueError:
                    res[k] = v
        elif isinstance(orig_val, bool):
            res[k] = v.lower() in ("true", "1", "yes")
        elif isinstance(orig_val, int):
            res[k] = int(v)
        elif isinstance(orig_val, float):
            res[k] = float(v)
        else:
            res[k] = v
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình DeepWeeds")
    parser.add_argument("--set", nargs="*", default=[], help="Cặp KEY=VALUE để ghi đè Config")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    run(cfg)


if __name__ == "__main__":
    main()
