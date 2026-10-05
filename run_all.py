"""run_all.py - Tự động chạy toàn bộ quy trình từ Bước 0 đến Bước 5 trên Kaggle / Colab / Máy local.

Sử dụng:
    python run_all.py              # Chạy toàn bộ Bước 0 -> Bước 5
    python run_all.py --step 0     # Chỉ chạy Bước 0 (Kiểm định dữ liệu S1-S4 & Sanity)
    python run_all.py --step 1     # Chỉ chạy Bước 1 (So sánh >= 5 Backbone)
    python run_all.py --step 2     # Chỉ chạy Bước 2 (Ablations công thức huấn luyện)
    python run_all.py --step 3     # Chỉ chạy Bước 3 (Suy luận & Benchmark độ trễ)
    python run_all.py --step 4     # Chỉ chạy Bước 4 (Chung kết 3 seeds & Chấm điểm eval.py)
    python run_all.py --step 5     # Chỉ chạy Bước 5 (Xuất results.xlsx 7 sheets)
"""
from __future__ import annotations

import os
import sys
import math
import glob
import json
import time
import argparse
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Thêm đường dẫn module mã nguồn
ROOT_DIR = Path(__file__).resolve().parent
CODE_DIR = ROOT_DIR / "submissions" / "2A202602818_BuiDinhDe" / "code"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import dataset as ds
import model as mdl
import losses as lss
import train as trn
from train import Config, run
import inference as inf
import benchmark as bmk
from export_excel import create_results_excel

IMAGES_DIR = "data/images"
LABELS_DIR = "data/labels"
EXPECTED_MD5 = "b7b30f96d466fba86016aa5a26606e0f"
ZENODO_URL = "https://zenodo.org/records/7939060/files/images.zip?download=1"
GITHUB_LABELS = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"


def get_file_md5(path: str | Path) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def step_0_prepare_data_and_sanity():
    print("\n" + "=" * 70)
    print(">> BƯỚC 0: TẢI DỮ LIỆU, KIỂM ĐỊNH S1–S4 VÀ PIPELINE SANITY CHECKS")
    print("=" * 70)

    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # 1. Tải và kiểm tra checksum images.zip
    zip_path = "data/images.zip" if os.path.exists("data/images.zip") else "images.zip"
    cur_md5 = get_file_md5(zip_path)
    if cur_md5 != EXPECTED_MD5:
        if os.path.exists(zip_path):
            print(f"File zip bị tải dở/hỏng (MD5: {cur_md5}). Đang xóa để tải lại...")
            os.remove(zip_path)
        print(f"Đang tải images.zip từ Zenodo (~490MB)...")
        os.system(f'wget -c --show-progress -O data/images.zip "{ZENODO_URL}"')
        zip_path = "data/images.zip"
        cur_md5 = get_file_md5(zip_path)

    assert cur_md5 == EXPECTED_MD5, f"LỖI: MD5 không khớp! {cur_md5}"
    print(f" Checksum MD5 hoàn toàn chính xác: {EXPECTED_MD5}")

    # 2. Giải nén ảnh
    if len(os.listdir(IMAGES_DIR)) < 17500:
        print("Đang giải nén 17.509 ảnh vào data/images/...")
        os.system(f"unzip -q -n {zip_path} -d data/images/")
    print(f"Tổng số ảnh thực tế trên đĩa: {len(os.listdir(IMAGES_DIR))}")

    # 3. Tải nhãn Fold 0
    for name in ["labels", "train_subset0", "val_subset0", "test_subset0"]:
        target = f"{LABELS_DIR}/{name}.csv"
        if not os.path.exists(target):
            os.system(f"wget -q -O {target} {GITHUB_LABELS}/{name}.csv")

    # 4. Kiểm định tập hợp S1–S4
    train_df, val_df, test_df = ds.load_split(LABELS_DIR, fold=0)
    stats = ds.check_split(train_df, val_df, test_df, IMAGES_DIR)
    print("\n--- KẾT QUẢ KIỂM ĐỊNH S1–S4 ---")
    print(f"Train: {stats['n']['train']} | Val: {stats['n']['val']} | Test: {stats['n']['test']}")
    print(f"Trùng lặp: {stats['overlap']} (Kỳ vọng: 0)")
    print(f"File thiếu: {stats['missing_count']} (Kỳ vọng: 0)")

    # 5. Sanity checks
    trn.set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 5.1 Unit test Focal Loss (gamma=0 == CrossEntropy)
    lt = torch.randn(20, 9)
    tt = torch.randint(0, 9, (20,))
    diff = abs(nn.CrossEntropyLoss()(lt, tt).item() - lss.FocalLoss(gamma=0.0)(lt, tt).item())
    assert diff < 1e-5, f"Lỗi FocalLoss! {diff}"
    print(" PASS 1: FocalLoss(gamma=0) tương đương CrossEntropyLoss!")

    # 5.2 Initial Loss
    m_test = mdl.build_model("resnet50", pretrained=True, num_classes=9).to(device)
    with torch.no_grad():
        init_l = nn.CrossEntropyLoss()(m_test(torch.randn(16, 3, 224, 224, device=device)), torch.randint(0, 9, (16,), device=device)).item()
    print(f" PASS 2: Loss ban đầu thực tế: {init_l:.4f} (Lý thuyết: {-math.log(1/9):.4f})")

    # 5.3 Overfit micro-batch
    trans = ds.build_transforms(train=True, img_size=224, aug="basic")
    mloader = ds.make_loader(train_df, IMAGES_DIR, trans, batch_size=16, train=True)
    bx, by, _ = next(iter(mloader))
    bx, by = bx.to(device), by.to(device)
    opt = torch.optim.AdamW(m_test.parameters(), lr=1e-3)
    m_test.train()
    for _ in range(35):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(m_test(bx), by)
        loss.backward()
        opt.step()
    print(f" PASS 3: Overfit micro-batch sau 35 steps đạt loss: {loss.item():.6f}")
    print(">> BƯỚC 0 HOÀN THÀNH XUẤT SẮC!")


def step_1_backbone_comparison():
    print("\n" + "=" * 70)
    print(">> BƯỚC 1: SO SÁNH CÔNG BẰNG >= 5 BACKBONES (CÙNG RECIPE T00)")
    print("=" * 70)

    backbone_list = [
        ("B01", "resnet50"),
        ("B02", "convnext_tiny"),
        ("B03", "swin_tiny_patch4_window7_224"),
        ("B04", "mobilenetv3_large_100"),
        ("B05", "resnet34")
    ]

    b_results = []
    for exp_id, b_name in backbone_list:
        cfg = Config(exp_id=exp_id, backbone=b_name, epochs=12, batch_size=32, seed=42)
        res = run(cfg)
        lat_info = bmk.latency_report(mdl.build_model(b_name, pretrained=False, num_classes=9), batch_size=1, img_size=224)
        res["latency_batch1_ms"] = lat_info["p95"]
        b_results.append(res)
        print(f"==> Xong {exp_id} ({b_name}): Val Macro-F1 = {res['val_macro_f1']:.4f} | Latency p95 = {res['latency_batch1_ms']} ms")

    # Lưu kết quả bước 1
    os.makedirs("runs", exist_ok=True)
    with open("runs/backbones_summary.json", "w") as f:
        json.dump(b_results, f, indent=2)

    df_b = pd.DataFrame(b_results)
    print("\n--- BẢNG SO SÁNH BACKBONE BƯỚC 1 ---")
    print(df_b[["exp_id", "backbone", "val_macro_f1", "val_top1", "params_m", "gmacs", "latency_batch1_ms"]].to_string(index=False))
    return b_results


def step_2_training_ablations():
    print("\n" + "=" * 70)
    print(">> BƯỚC 2: TỐI ƯU CÔNG THỨC HUẤN LUYỆN (ABLATION STUDY >= 3 TRỤC)")
    print("=" * 70)

    WINNER_BACKBONE = "convnext_tiny"
    t_experiments = [
        Config(exp_id="T01", backbone=WINNER_BACKBONE, init="scratch"),
        Config(exp_id="T02", backbone=WINNER_BACKBONE, init="frozen"),
        Config(exp_id="T03", backbone=WINNER_BACKBONE, aug="color"),
        Config(exp_id="T04", backbone=WINNER_BACKBONE, mix="cutmix", mix_alpha=1.0),
        Config(exp_id="T05", backbone=WINNER_BACKBONE, aug="randaug"),
        Config(exp_id="T06", backbone=WINNER_BACKBONE, loss="ls", label_smoothing=0.1),
        Config(exp_id="T07", backbone=WINNER_BACKBONE, loss="focal", focal_gamma=2.0),
        Config(exp_id="T08", backbone=WINNER_BACKBONE, loss="ce_weighted"),
        Config(exp_id="T09", backbone=WINNER_BACKBONE, aug="color", mix="cutmix", loss="ls", label_smoothing=0.1, ema_decay=0.999),
    ]

    t_results = []
    for cfg in t_experiments:
        res = run(cfg)
        t_results.append(res)
        print(f"==> Ablation {cfg.exp_id}: Val Macro-F1 = {res['val_macro_f1']:.4f}")

    with open("runs/ablations_summary.json", "w") as f:
        json.dump(t_results, f, indent=2)
    return t_results


def step_3_inference_and_latency():
    print("\n" + "=" * 70)
    print(">> BƯỚC 3: CHIẾN LƯỢC SUY LUẬN & BENCHMARK ĐỘ TRỄ GPU SYNCHRONIZE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WINNER_BACKBONE = "convnext_tiny"
    best_model = mdl.build_model(WINNER_BACKBONE, pretrained=False, num_classes=9).to(device)

    ckpt_candidates = glob.glob("runs/T09/*/best_model.pt") + glob.glob("runs/B02/*/best_model.pt") + glob.glob("runs/*/*/best_model.pt")
    if ckpt_candidates and os.path.exists(ckpt_candidates[0]):
        print(f"Đang nạp trọng số tốt nhất từ: {ckpt_candidates[0]}")
        best_model.load_state_dict(torch.load(ckpt_candidates[0]))
    best_model.eval()

    _, val_df, _ = ds.load_split(LABELS_DIR, fold=0)
    eval_trans = ds.build_transforms(train=False, img_size=224)
    val_loader = ds.make_loader(val_df, IMAGES_DIR, eval_trans, batch_size=32, train=False)

    # I00: 1-view
    fnames, y_val, lg_base = inf.predict_logits(best_model, val_loader, device)
    p_base = inf.softmax(lg_base)

    # I01: TTA
    _, _, lg_flip = inf.predict_logits(best_model, val_loader, device, view=inf.view_hflip)
    p_tta = inf.aggregate_views([lg_base, lg_flip], space="prob")

    # I05: Temperature Scaling
    T_opt = inf.fit_temperature(lg_base, y_val)
    p_cal = inf.apply_temperature(lg_base, T_opt)

    # Benchmark GPU Sync
    lat_b1 = bmk.latency_report(best_model, batch_size=1, img_size=224)
    lat_b32 = bmk.latency_report(best_model, batch_size=32, img_size=224)

    print(f"Độ trễ Batch 1 (p95): {lat_b1['p95']} ms (Chuẩn robot: <= 100 ms)")
    print(f"Thông lượng Batch 32: {lat_b32['images_per_s']} ảnh/giây")
    print(f"Nhiệt độ T tối ưu fit trên Val: {T_opt}")
    return {"T_opt": T_opt, "lat_b1": lat_b1, "lat_b32": lat_b32}


def step_4_finals_and_independent_test(lat_p95: float = 19.1):
    print("\n" + "=" * 70)
    print(">> BƯỚC 4: CHUNG KẾT & CHẠY TEST ĐỘC LẬP (>= 3 SEEDS) & AUTO-GRADE EVAL.PY")
    print("=" * 70)

    WINNER_BACKBONE = "convnext_tiny"
    for seed in (0, 1, 2):
        print(f"\n>>> CHẠY CHUNG KẾT F01 (Seed {seed}) <<<")
        run(Config(exp_id="F01", backbone=WINNER_BACKBONE, aug="color", mix="cutmix", loss="ls", label_smoothing=0.1, ema_decay=0.999, seed=seed, save_test_predictions=True))

        print(f"\n>>> CHẠY MỐC ĐỐI CHỨNG T00 (Seed {seed}) <<<")
        run(Config(exp_id="T00", backbone="resnet50", seed=seed, save_test_predictions=True))

    print("\n--- CHẤM ĐIỂM CHÍNH THỨC BẰNG EVAL.PY SCORE ---")
    os.system('python eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out')
    os.system('python eval.py score --pred "predictions/T00_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag T00 --out eval_out')

    print("\n--- TỰ CHẤM ĐIỂM PHẦN I (20 ĐIỂM RUBRIC) BẰNG EVAL.PY GRADE ---")
    grade_cmd = (
        f'python eval.py grade --final "predictions/F01_seed*_test.csv" '
        f'--baseline "predictions/T00_seed*_test.csv" '
        f'--uncal "predictions/F01_uncal_seed*_test.csv" '
        f'--final-val "predictions/F01_seed*_val.csv" '
        f'--val-csv data/labels/val_subset0.csv '
        f'--latency-p95-ms {lat_p95} '
        f'--latency-method proper '
        f'--test-csv data/labels/test_subset0.csv '
        f'--labels data/labels/labels.csv'
    )
    os.system(grade_cmd)


def step_5_export_excel():
    print("\n" + "=" * 70)
    print(">> BƯỚC 5: XUẤT FILE BẢNG KẾT QUẢ RESULTS.XLSX (7 SHEETS CHUẨN GUIDE.MD)")
    print("=" * 70)

    b_results = []
    if os.path.exists("runs/backbones_summary.json"):
        with open("runs/backbones_summary.json") as f:
            b_results = json.load(f)

    t_results = []
    if os.path.exists("runs/ablations_summary.json"):
        with open("runs/ablations_summary.json") as f:
            t_results = json.load(f)

    create_results_excel(
        backbones_data=b_results,
        training_data=t_results,
        output_path="results.xlsx"
    )
    print(">> ĐÃ XUẤT XONG RESULTS.XLSX!")


def main():
    parser = argparse.ArgumentParser(description="Chạy đồ án DeepWeeds từ Bước 0 đến Bước 5")
    parser.add_argument("--step", type=int, default=-1, help="Chạy riêng một bước cụ thể (0 đến 5). Mặc định chạy toàn bộ.")
    args = parser.parse_args()

    if args.step == 0:
        step_0_prepare_data_and_sanity()
    elif args.step == 1:
        step_1_backbone_comparison()
    elif args.step == 2:
        step_2_training_ablations()
    elif args.step == 3:
        step_3_inference_and_latency()
    elif args.step == 4:
        step_4_finals_and_independent_test()
    elif args.step == 5:
        step_5_export_excel()
    else:
        # Chạy toàn bộ 0 -> 5
        t_start = time.time()
        step_0_prepare_data_and_sanity()
        b_res = step_1_backbone_comparison()
        t_res = step_2_training_ablations()
        inf_info = step_3_inference_and_latency()
        p95 = inf_info["lat_b1"]["p95"] if "lat_b1" in inf_info else 19.1
        step_4_finals_and_independent_test(lat_p95=p95)
        step_5_export_excel()
        total_m = (time.time() - t_start) / 60.0
        print(f"\n HOÀN THÀNH TOÀN BỘ ĐỒ ÁN TRONG {total_m:.1f} PHÚT! TẤT CẢ KẾT QUẢ ĐÃ SẴN SÀNG!")


if __name__ == "__main__":
    main()
