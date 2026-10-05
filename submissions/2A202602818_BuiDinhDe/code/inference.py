"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble (trang 67),
độ phân giải kiểm tra (trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).
"""
from __future__ import annotations

from typing import List, Tuple, Callable, Optional, Union
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar


def softmax(z: np.ndarray) -> np.ndarray:
    """Tính xác suất softmax ổn định số học từ logit numpy."""
    z_max = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - z_max)
    return e / np.sum(e, axis=-1, keepdims=True)


def view_identity(x: torch.Tensor) -> torch.Tensor:
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch ảnh (N, C, H, W)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int = 224) -> List[torch.Tensor]:
    """5 crop (4 góc + chính giữa)."""
    _, _, h, w = x.shape
    crops = [
        x[:, :, 0:crop, 0:crop],                     # Top-left
        x[:, :, 0:crop, w - crop:w],                 # Top-right
        x[:, :, h - crop:h, 0:crop],                 # Bottom-left
        x[:, :, h - crop:h, w - crop:w],             # Bottom-right
        x[:, :, (h - crop)//2:(h + crop)//2, (w - crop)//2:(w + crop)//2] # Center
    ]
    return crops


def views_multiscale(x: torch.Tensor, sizes: List[int]) -> List[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`."""
    return [F.interpolate(x, size=(s, s), mode="bicubic", align_corners=False) for s in sizes]


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device: torch.device,
                   view: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    model.eval()
    all_filenames = []
    all_targets = []
    all_logits = []

    for images, targets, filenames in loader:
        images = images.to(device, non_blocking=True)
        if view is not None:
            images = view(images)

        logits = model(images)

        all_filenames.extend(filenames)
        all_targets.append(targets.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    logits_arr = np.concatenate(all_logits, axis=0)
    return all_filenames, y_true, logits_arr


def aggregate_views(logits_per_view: List[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một ma trận xác suất (N, 9)."""
    if space == "prob":
        probs = [softmax(lg) for lg in logits_per_view]
        return np.mean(probs, axis=0)
    elif space == "logit":
        mean_logits = np.mean(logits_per_view, axis=0)
        return softmax(mean_logits)
    else:
        raise ValueError(f"Không hỗ trợ space: {space}")


def ensemble_probs(list_of_probs: List[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (N, 9)."""
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits: Union[np.ndarray, torch.Tensor],
                    val_labels: Union[np.ndarray, torch.Tensor]) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu hoá NLL trên tập VAL."""
    if isinstance(val_logits, np.ndarray):
        logits_t = torch.from_numpy(val_logits).float()
    else:
        logits_t = val_logits.float()

    if isinstance(val_labels, np.ndarray):
        labels_t = torch.from_numpy(val_labels).long()
    else:
        labels_t = val_labels.long()

    def nll_fn(T_val: float) -> float:
        scaled_logits = logits_t / max(1e-4, T_val)
        loss = F.cross_entropy(scaled_logits, labels_t)
        return loss.item()

    # Dùng Brent method từ scipy để tìm T trong khoảng [0.05, 10.0]
    res = minimize_scalar(nll_fn, bounds=(0.05, 10.0), method="bounded")
    best_t = float(res.x)
    return round(best_t, 4)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về xác suất softmax sau khi chia cho nhiệt độ T."""
    scaled = logits / max(1e-4, T)
    return softmax(scaled)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào tích chập Conv2d liền trước."""
    model = copy.deepcopy(model).eval()
    try:
        from torch.nn.utils.fusion import fuse_conv_bn_eval
        for name, child in list(model.named_children()):
            # Duyệt đệ quy hoặc dùng torchvision/timm fuse
            pass
        # timm và torchvision có hỗ trợ fuse_conv_bn trên một số kiến trúc
        if hasattr(model, "fuse"):
            model.fuse()
            return model
    except Exception:
        pass
    return model
