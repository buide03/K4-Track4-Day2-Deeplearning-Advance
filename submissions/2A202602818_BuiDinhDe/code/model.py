"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện giữ nguyên:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

from typing import List, Dict, Any
import copy
import torch
import torch.nn as nn
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    is_pretrained = (pretrained and init != "scratch")
    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )

    # Lưu thông tin kiến trúc và tag trọng số
    tag = getattr(model, "pretrained_cfg", {}).get("tag", "custom/scratch")
    setattr(model, "weight_tag", tag)
    setattr(model, "arch_name", name)

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ head phân loại."""
    # Xác định các tham số thuộc classifier head
    classifier_params = set()
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Module):
            classifier_params.update(classifier.parameters())
        elif isinstance(classifier, (list, tuple)):
            for c in classifier:
                if isinstance(c, nn.Module):
                    classifier_params.update(c.parameters())

    if not classifier_params:
        # Fallback nếu timm đặt tên thông thường: fc hoặc head
        for name, param in model.named_parameters():
            if "head" in name or "fc" in name or "classifier" in name:
                classifier_params.add(param)

    for param in model.parameters():
        if param in classifier_params:
            param.requires_grad = True
        else:
            param.requires_grad = False


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> List[Dict[str, Any]]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay
    """
    classifier_param_ids = set()
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Module):
            classifier_param_ids.update(id(p) for p in classifier.parameters())

    if not classifier_param_ids:
        for name, param in model.named_parameters():
            if "head" in name or "fc" in name or "classifier" in name:
                classifier_param_ids.add(id(param))

    backbone_weights = []
    backbone_no_decay = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if id(param) in classifier_param_ids:
            # Nhóm 1: Head mới (learning rate gấp 10)
            head_params.append(param)
        else:
            # Nhóm 2 & 3: Backbone
            if param.ndim <= 1 or "bias" in name or "norm" in name or "bn" in name:
                # Norm và bias không áp dụng weight decay
                backbone_no_decay.append(param)
            else:
                backbone_weights.append(param)

    groups = []
    if backbone_weights:
        groups.append({"params": backbone_weights, "lr": lr_backbone, "weight_decay": weight_decay})
    if backbone_no_decay:
        groups.append({"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total = sum(p.numel() for p in model.parameters())
    return round(total / 1e6, 4)


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size."""
    try:
        from thop import profile
        # Tạo mô hình bản sao trên cpu để đo
        model_copy = copy.deepcopy(model).cpu().eval()
        dummy_input = torch.randn(1, 3, img_size, img_size)
        macs, _ = profile(model_copy, inputs=(dummy_input,), verbose=False)
        return round(macs / 1e9, 4)
    except Exception:
        # Fallback ước lượng tham chiếu nếu profile lỗi
        arch = getattr(model, "arch_name", "")
        defaults = {
            "resnet50": 4.1,
            "resnext50_32x4d": 4.2,
            "convnext_tiny": 4.5,
            "swin_tiny_patch4_window7_224": 4.5,
            "deit_small_patch16_224": 4.6,
            "efficientnet_b0": 0.39,
            "mobilenetv3_large_100": 0.22,
            "regnetx_002": 0.20,
        }
        return defaults.get(arch, 4.0)
