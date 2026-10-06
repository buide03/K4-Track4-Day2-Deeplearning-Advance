# Báo Cáo Kỹ Thuật & Hướng Dẫn Tái Lập Thí Nghiệm — Lab Day 2 (DeepWeeds)

* **Sinh viên**: Bùi Đình Đệ
* **Mã số sinh viên (MSSV)**: `2A202602818`
* **Lớp / Track**: Track 4 · Deep Learning Advance · Day 2
* **Bài lab**: Phân loại cỏ dại DeepWeeds 9 lớp cho robot nông nghiệp tự hành (Mục tiêu: 100/100)
* **Kết quả tự chấm `eval.py grade`**: **20 / 20 điểm tuyệt đối Phần I**

---

## 1. Môi Trường Thực Nghiệm & Phiên Bản Thư Viện

Thí nghiệm được thực hiện và kiểm chứng trên:
* **Hệ điều hành**: Linux Ubuntu / Kaggle & Google Colab (GPU Environment)
* **GPU**: NVIDIA Tesla T4 (16GB VRAM) trên Kaggle / Colab
* **CUDA**: 12.1 / 12.4
* **Python**: `3.10` / `3.12`
* **Các thư viện cốt lõi**:
  * `torch`: `2.4.0` / `2.6.0`
  * `torchvision`: `0.19.0` / `0.21.0`
  * `timm`: `1.0.9` / `1.0.30`
  * `pandas`: `2.2.2` / `3.0.6`
  * `scikit-learn`: `1.5.1` / `1.9.1`
  * `openpyxl`: `3.1.5`
  * `matplotlib`: `3.9.2` / `3.11.2`
  * `seaborn`: `0.13.2`

---

## 2. Hướng Dẫn Tái Lập Kết Quả (Reproducibility Guide)

### Cách 1: Chạy toàn bộ tự động bằng `run_all.py` hoặc Notebook
* **Script tự động**: `python run_all.py` (Chạy toàn trình từ Bước 0 đến Bước 5).
* **Notebook tái lập**: [`code/lab_day2.ipynb`](code/lab_day2.ipynb).
* **Quy trình chạy**:
  1. Mở file `code/lab_day2.ipynb` trên Google Colab / Kaggle.
  2. Chọn phần cứng: **GPU (T4 hoặc cao hơn)**.
  3. Bấm **Run All**.
  4. Hệ thống sẽ tự động:
     - Tải và kiểm tra MD5 checksum của `images.zip` (`b7b30f96d466fba86016aa5a26606e0f`).
     - Thực hiện kiểm tra tính toàn vẹn tập hợp dữ liệu S1–S4 và EDA.
     - Huấn luyện $\ge 5$ Backbone (`B01`–`B05`) theo công thức chuẩn `T00`.
     - Huấn luyện các nhánh Ablation (`T01`–`T09`).
     - Thực hiện các phương pháp suy luận (`I00`–`I05`), đo độ trễ p50/p95/p99 ở batch 1 và batch 32 với GPU synchronization.
     - Huấn luyện chung kết 3 seed cho `F01` và `T00`, xuất dự đoán ra thư mục `predictions/`.
     - Chạy tự động `eval.py score` và `eval.py grade`.
     - Xuất toàn bộ bảng số liệu chuẩn 7 sheets vào `results.xlsx`.

### Cách 2: Chạy trực tiếp từ dòng lệnh (Local / Terminal)
Từ thư mục gốc của repository:

```bash
# 1. Kích hoạt môi trường ảo
source .venv/bin/activate

# 2. Chạy kiểm tra unit test (38 tests)
python -m unittest discover -s tests -v

# 3. Huấn luyện một backbone cụ thể (ví dụ ResNet-50 hoặc ConvNeXt-Tiny)
python submissions/2A202602818_BuiDinhDe/code/train.py --set exp_id=B01 backbone=resnet50 seed=42

# 4. Chạy chung kết 3 seed cho Champion F01 và Baseline T00
for s in 0 1 2; do
    python submissions/2A202602818_BuiDinhDe/code/train.py --set exp_id=F01 backbone=convnext_tiny aug=color mix=cutmix loss=ls label_smoothing=0.1 ema_decay=0.999 seed=$s save_test_predictions=True
    python submissions/2A202602818_BuiDinhDe/code/train.py --set exp_id=T00 backbone=resnet50 seed=$s save_test_predictions=True
done

# 5. Tự chấm điểm bằng eval.py
python eval.py score --pred "predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out

python eval.py grade \
    --final "predictions/F01_seed*_test.csv" \
    --baseline "predictions/T00_seed*_test.csv" \
    --uncal "predictions/F01_uncal_seed*_test.csv" \
    --final-val "predictions/F01_seed*_val.csv" \
    --val-csv data/labels/val_subset0.csv \
    --latency-p95-ms 6.302 --latency-method proper \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv
```

---

## 3. Cấu Trúc Sản Phẩm Nộp Bài

```text
submissions/2A202602818_BuiDinhDe/
├── README.md          # File này (hướng dẫn chạy lại, phiên bản thư viện, seed)
├── results.xlsx       # Bảng Excel kết quả đủ 7 sheets (Backbones, Training, Inference, Final, PerClass, Latency, Summary)
├── report.md          # Báo cáo khoa học 9 phần chuẩn theo GUIDE.md
├── curves/            # Thư mục chứa biểu đồ Loss và Metric cho từng exp_id (B01-B05, T01-T09, F01, T00)
│   ├── eda_class_distribution.png
│   ├── B01_resnet50.png
│   ├── B02_convnext_tiny.png
│   └── ...
├── predictions/       # File dự đoán test và val của các seed để eval.py tự động chấm điểm
│   ├── F01_seed0_test.csv
│   ├── F01_seed1_test.csv
│   ├── F01_seed2_test.csv
│   ├── F01_uncal_seed0_test.csv
│   ├── F01_seed0_val.csv
│   ├── T00_seed0_test.csv
│   └── ...
└── code/              # Toàn bộ mã nguồn hoàn thiện không còn NotImplementedError
    ├── dataset.py
    ├── model.py
    ├── losses.py
    ├── train.py
    ├── inference.py
    ├── benchmark.py
    ├── export_excel.py
    └── lab_day2.ipynb
```

---

## 4. Tóm Tắt Kết Quả Chính Thức (Official Rubric Score)

| Mã | Tiêu chí đánh giá | Điểm đạt | Điểm tối đa | Kết quả thực nghiệm |
|:---:|---|:---:|:---:|---|
| **I1** | Top-1 accuracy test (mean 3 seeds) | **7** | 7 | **97,96% ± 0,15%** ($\ge 95,7\%$) |
| **I2** | Macro-F1 test cải thiện so với mốc | **5** | 5 | Final: **0,9753**, Mốc: 0,8640, $\mathbf{\Delta = +0,1113}$ ($\gg s = 0,0058$) |
| **I3** | Recall hai lớp cỏ khó nhận diện | **4** | 4 | Chinee apple: **93,4%** (mốc 88,5%), Snake weed: **95,9%** (mốc 88,8%) |
| **I4a**| ECE sau Temperature Scaling < trước | **1** | 1 | Trước: 0,0889 $\rightarrow$ Sau: **0,0069** |
| **I4b**| Chênh lệch Macro-F1 Val/Test $\le 0,02$ | **1** | 1 | Val: 0,9734, Test: 0,9753, Chênh: **0,0018** |
| **I5** | Cấu hình thời gian thực cho robot | **2** | 2 | Latency batch 1 p95 = **6,30 ms** ($\le 100\text{ ms}$, đo đồng bộ CUDA) |
| **TỔNG**| **Điểm tự chấm Mục I** | **20** | **20** | **ĐẠT ĐIỂM TỐI ĐA 20 / 20** |
