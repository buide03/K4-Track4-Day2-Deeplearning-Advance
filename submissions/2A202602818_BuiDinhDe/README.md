# Báo Cáo Kỹ Thuật & Hướng Dẫn Tái Lập Thí Nghiệm — Lab Day 2 (DeepWeeds)

* **Sinh viên**: Bùi Đình Đệ
* **Mã số sinh viên (MSSV)**: `2A202602818`
* **Lớp / Track**: Track 4 · Deep Learning Advance · Day 2
* **Bài lab**: Phân loại cỏ dại DeepWeeds 9 lớp cho robot nông nghiệp tự hành

---

## 1. Môi Trường Thực Nghiệm & Phiên Bản Thư Viện

Thí nghiệm được thực hiện và kiểm chứng trên:
* **Hệ điều hành**: Linux Ubuntu / Google Colab (GPU Environment)
* **GPU**: Tesla T4 (16GB VRAM) trên Colab / NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM) tại Local
* **CUDA**: 12.4
* **Python**: `3.12.3`
* **Các thư viện cốt lõi**:
  * `torch`: `2.6.0+cu124`
  * `torchvision`: `0.21.0+cu124`
  * `timm`: `1.0.30`
  * `pandas`: `3.0.6`
  * `scikit-learn`: `1.9.1`
  * `openpyxl`: `3.1.5`
  * `matplotlib`: `3.11.2`
  * `seaborn`: `0.13.2`

---

## 2. Hướng Dẫn Tái Lập Kết Quả (Reproducibility Guide)

### Cách 1: Chạy lại toàn bộ bằng Notebook trên Google Colab (Khuyên dùng)
* **Notebook tái lập**: [`code/lab_day2.ipynb`](code/lab_day2.ipynb)
* **Quy trình chạy**:
  1. Mở file `code/lab_day2.ipynb` trên Google Colab.
  2. Chọn phần cứng: **Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU**.
  3. Chạy lần lượt các ô từ trên xuống dưới (hoặc bấm **Runtime $\rightarrow$ Run all**).
  4. Notebook sẽ tự động:
     - Tải và kiểm tra MD5 checksum của `images.zip`.
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

# 2. Huấn luyện một backbone cụ thể (ví dụ ResNet-50 hoặc ConvNeXt-Tiny)
python submissions/2A202602818_BuiDinhDe/code/train.py --set exp_id=B01 backbone=resnet50 seed=42

# 3. Chạy chung kết 3 seed
for s in 0 1 2; do
    python submissions/2A202602818_BuiDinhDe/code/train.py --set exp_id=F01 backbone=convnext_tiny seed=$s save_test_predictions=True
    python submissions/2A202602818_BuiDinhDe/code/train.py --set exp_id=T00 backbone=resnet50 seed=$s save_test_predictions=True
done

# 4. Tự chấm điểm bằng eval.py
python eval.py score --pred "submissions/2A202602818_BuiDinhDe/predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01

python eval.py grade \
    --final "submissions/2A202602818_BuiDinhDe/predictions/F01_seed*_test.csv" \
    --baseline "submissions/2A202602818_BuiDinhDe/predictions/T00_seed*_test.csv" \
    --uncal "submissions/2A202602818_BuiDinhDe/predictions/F01_uncal_seed*_test.csv" \
    --final-val "submissions/2A202602818_BuiDinhDe/predictions/F01_seed*_val.csv" \
    --val-csv data/labels/val_subset0.csv \
    --latency-p95-ms 28.5 --latency-method proper \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv
```

---

## 3. Cấu Trúc Sản Phẩm Nộp Bài

```text
submissions/2A202602818_BuiDinhDe/
├── README.md          # File này (hướng dẫn chạy lại, phiên bản thư viện, seed)
├── results.xlsx       # Bảng Excel kết quả đủ 7 sheets (Backbones, Training, Inference, Final, PerClass, Latency, Summary)
├── report.md          # Báo cáo khoa học 9 phần chuẩn theo GUIDE.md
├── curves/            # Thư mục chứa biểu đồ Loss và Metric cho từng exp_id
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
└── code/              # Toàn bộ mã nguồn hoàn thiện
    ├── dataset.py
    ├── model.py
    ├── losses.py
    ├── train.py
    ├── inference.py
    ├── benchmark.py
    ├── export_excel.py
    └── lab_day2.ipynb
```
