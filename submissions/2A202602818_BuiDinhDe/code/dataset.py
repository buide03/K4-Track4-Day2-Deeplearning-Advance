"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.
Giao diện giữ nguyên:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1)."""
    labels_dir = Path(labels_dir)
    train_path = labels_dir / f"train_subset{fold}.csv"
    val_path = labels_dir / f"val_subset{fold}.csv"
    test_path = labels_dir / f"test_subset{fold}.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Đọc labels.csv để map Species nếu chưa có
    labels_path = labels_dir / "labels.csv"
    if labels_path.exists():
        full_labels = pd.read_csv(labels_path)
        if "Species" in full_labels.columns and "Label" in full_labels.columns:
            label2species = full_labels[["Label", "Species"]].drop_duplicates().set_index("Label")["Species"].to_dict()
            for df in (train_df, val_df, test_df):
                if "Species" not in df.columns and "Label" in df.columns:
                    df["Species"] = df["Label"].map(label2species)

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> Dict[str, Any]:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu."""
    images_dir = Path(images_dir)
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    total_samples = n_train + n_val + n_test

    s_train = set(train_df["Filename"])
    s_val = set(val_df["Filename"])
    s_test = set(test_df["Filename"])

    overlap_train_val = len(s_train & s_val)
    overlap_train_test = len(s_train & s_test)
    overlap_val_test = len(s_val & s_test)

    assert overlap_train_val == 0, f"LỖI S2: Trùng lặp Train và Val ({overlap_train_val} ảnh)"
    assert overlap_train_test == 0, f"LỖI S2: Trùng lặp Train và Test ({overlap_train_test} ảnh)"
    assert overlap_val_test == 0, f"LỖI S2: Trùng lặp Val và Test ({overlap_val_test} ảnh)"

    s_all = s_train | s_val | s_test
    assert len(s_all) == 17509, f"LỖI S3: Hợp 3 tập có {len(s_all)} ảnh, không đúng 17.509!"

    # Kiểm tra tồn tại trên đĩa
    missing_files = [f for f in s_all if not (images_dir / f).exists()]
    assert len(missing_files) == 0, f"LỖI S4: Có {len(missing_files)} file trong CSV thiếu trên đĩa!"

    # Đếm số lượng theo lớp
    per_class = {
        "train": train_df["Label"].value_counts().to_dict(),
        "val": val_df["Label"].value_counts().to_dict(),
        "test": test_df["Label"].value_counts().to_dict(),
    }

    results = {
        "n": {"train": n_train, "val": n_val, "test": n_test, "total": total_samples},
        "overlap": {
            "train_val": overlap_train_val,
            "train_test": overlap_train_test,
            "val_test": overlap_val_test
        },
        "per_class": per_class,
        "missing_count": len(missing_files)
    }
    return results


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic") -> transforms.Compose:
    """Tạo transform chuẩn hoá cho ảnh.

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: Resize(256) -> CenterCrop(img_size) + ToTensor + Normalize.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if not train:
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])

    # Augmentation cho train
    t_list = [transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC)]
    t_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if aug == "basic":
        pass
    elif aug == "color":
        t_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    elif aug == "randaug":
        t_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
    elif aug == "trivial":
        t_list.append(transforms.TrivialAugmentWide())
    else:
        # Nếu có tuỳ chọn kết hợp khác
        pass

    t_list.extend([
        transforms.ToTensor(),
        normalize,
    ])
    return transforms.Compose(t_list)


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ images_dir theo DataFrame (Filename, Label)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform: Optional[transforms.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].values
        self.labels = self.df["Label"].values.astype(np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, str]:
        fname = self.filenames[i]
        label = int(self.labels[i])
        img_path = self.images_dir / fname
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, fname


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform: transforms.Compose,
                batch_size: int, train: bool, sampler: Optional[str] = None,
                num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader."""
    dataset = DeepWeedsDataset(df=df, images_dir=images_dir, transform=transform)

    data_sampler = None
    shuffle = train

    if train and sampler == "balanced":
        # WeightedRandomSampler cân bằng 9 lớp
        class_counts = df["Label"].value_counts().to_dict()
        weights = [1.0 / class_counts[label] for label in df["Label"]]
        data_sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=data_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        worker_init_fn=seed_worker if train else None
    )
    return loader
