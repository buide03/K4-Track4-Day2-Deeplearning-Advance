"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() trước và sau đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99
  - ghi rõ GPU, dtype, batch, độ phân giải
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast


def bench(fn: Callable[[], Any], warmup: int = 10, iters: int = 100, sync: Optional[Callable[[], None]] = None) -> Dict[str, float]:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây (ms)."""
    # 1. Warmup
    for _ in range(warmup):
        fn()
    if sync:
        sync()

    # 2. Đo thật
    latencies = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    latencies_arr = np.array(latencies)
    return {
        "p50": round(float(np.percentile(latencies_arr, 50)), 3),
        "p95": round(float(np.percentile(latencies_arr, 95)), 3),
        "p99": round(float(np.percentile(latencies_arr, 99)), 3),
        "mean": round(float(np.mean(latencies_arr)), 3),
        "std": round(float(np.std(latencies_arr)), 3),
        "n": iters
    }


def latency_report(model: nn.Module, batch_size: int = 1, img_size: int = 224,
                   dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> Dict[str, Any]:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()

    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    sync_fn = torch.cuda.synchronize if dev.type == "cuda" else None

    if dtype == "fp16":
        model = model.half()
        dummy_input = dummy_input.half()

        @torch.inference_mode()
        def forward_fn():
            return model(dummy_input)

    elif dtype == "amp":
        @torch.inference_mode()
        def forward_fn():
            with autocast("cuda"):
                return model(dummy_input)

    else: # fp32
        @torch.inference_mode()
        def forward_fn():
            return model(dummy_input)

    res = bench(forward_fn, warmup=warmup, iters=iters, sync=sync_fn)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    throughput = round(batch_size / (res["p50"] / 1000.0), 2) if res["p50"] > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": res["p50"],
        "p95": res["p95"],
        "p99": res["p99"],
        "mean": res["mean"],
        "images_per_s": throughput,
        "torch": torch.__version__
    }


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1, img_size: int = 224,
                device: str = "cuda", warmup: int = 10, iters: int = 50) -> Dict[str, Any]:
    """Đo độ trễ TTA với K views (ví dụ 1 view thường + 1 view lật ngang)."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    sync_fn = torch.cuda.synchronize if dev.type == "cuda" else None

    @torch.inference_mode()
    def tta_fn():
        out1 = model(dummy_input)
        out2 = model(torch.flip(dummy_input, dims=[-1]))
        return (out1 + out2) * 0.5

    res = bench(tta_fn, warmup=warmup, iters=iters, sync=sync_fn)
    return {
        "k_views": k_views,
        "p50": res["p50"],
        "p95": res["p95"],
        "p99": res["p99"],
        "mean": res["mean"]
    }
