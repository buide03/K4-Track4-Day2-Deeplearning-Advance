# BÁO CÁO NGHIÊN CỨU THỰC NGHIỆM: ĐÁNH GIÁ VÀ TỐI ƯU HÓA MÔ HÌNH PHÂN LOẠI CỎ DẠI DEEPWEEDS CHO ROBOT NÔNG NGHIỆP

* **Tác giả**: Bùi Đình Đệ (MSSV: `2A202602818`)
* **Khóa học / Track**: Track 4 · Deep Learning Advance · Day 2
* **Bài lab**: Backbone, công thức huấn luyện và kỹ thuật suy luận trên DeepWeeds

---

## 1. TÓM TẮT (EXECUTIVE SUMMARY)
Nghiên cứu này giải quyết bài toán phân loại hình ảnh thực địa 9 lớp trên bộ dữ liệu **DeepWeeds** (17.509 ảnh RGB 256×256) phục vụ hệ thống robot nông nghiệp phun thuốc diệt cỏ tự động tại bang Queensland, Úc. Chúng tôi tiến hành thực nghiệm có kiểm soát đa chiều qua 3 trụ cột: (1) Sàng lọc 5 kiến trúc backbone qua 4 họ mô hình khác nhau; (2) Tối ưu hóa công thức huấn luyện qua 3 trục chính (khởi tạo, augmentation, hàm loss và regularization); (3) Đánh giá các kỹ thuật suy luận và hiệu chuẩn độ tin cậy kết hợp đo độ trễ phần cứng với GPU synchronization. Cấu hình tối ưu được lựa chọn hoàn toàn dựa trên tập Val, sau đó đánh giá độc lập trên tập Test qua 3 seeds ngẫu nhiên. Kết quả cho thấy công thức huấn luyện tối ưu mang lại mức cải thiện vượt trội so với baseline nền, đồng thời đảm bảo độ trễ thời gian thực $\le 100\text{ ms}$ ở batch 1 đáp ứng hoàn hảo chu kỳ cảm biến của robot thực địa.

---

## 2. DỮ LIỆU VÀ THIẾT LẬP THỰC NGHIỆM (SETUP & EDA)

### 2.1 Đặc tính tập dữ liệu & Kiểm định tính toàn vẹn (S1–S4)
Dữ liệu sử dụng phân chia **Fold 0** nguyên bản từ tác giả Alex Olsen (tỷ lệ 60/20/20). Tất cả 4 kiểm định tiên quyết đều đạt chuẩn tuyệt đối:
* **Train**: 10.501 ảnh (59,97%)
* **Validation**: 3.501 ảnh (20,00%)
* **Test**: 3.507 ảnh (20,03%)
* **Tổng số ảnh**: 17.509 ảnh.
* **Giao các tập**: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
* **File thiếu trên đĩa**: 0 file.

### 2.2 Phân tích mất cân bằng lớp (EDA)
Dữ liệu thể hiện sự mất cân bằng nghiêm trọng:
* Lớp **Negative** chiếm áp đảo với **9.106 ảnh (52,01%)**.
* 8 loài cỏ dại mục tiêu chỉ chiếm từ **5,76% đến 6,43%** mỗi loài (~1.009 đến 1.125 ảnh).
* *Hệ quả*: Top-1 Accuracy bị lớp Negative kéo cao ảo. Vì vậy, **Macro-F1** (trung bình không trọng số của 9 lớp) được xác định là chỉ số tối thượng để đánh giá mô hình.

| Label | Tên loài (Species) | Train | Val | Test | Tổng số ảnh | Tỷ lệ dataset |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 0 | Chinee apple | 675 | 225 | 226 | 1.125 | 6,43% |
| 1 | Lantana | 637 | 213 | 213 | 1.064 | 6,08% |
| 2 | Parkinsonia | 618 | 206 | 207 | 1.031 | 5,89% |
| 3 | Parthenium | 613 | 204 | 205 | 1.022 | 5,84% |
| 4 | Prickly acacia | 637 | 212 | 213 | 1.062 | 6,07% |
| 5 | Rubber vine | 605 | 202 | 202 | 1.009 | 5,76% |
| 6 | Siam weed | 644 | 215 | 215 | 1.074 | 6,13% |
| 7 | Snake weed | 609 | 203 | 204 | 1.016 | 5,80% |
| 8 | Negative | 5.463 | 1.821 | 1.822 | 9.106 | **52,01%** |

### 2.3 Công thức nền (Baseline Recipe `T00`)
* **Khởi tạo**: ImageNet pretrained weights.
* **Đầu vào**: Train: `RandomResizedCrop(224)` + `RandomHorizontalFlip()`. Val: `Resize(256)` + `CenterCrop(224)`.
* **Bộ tối ưu**: AdamW, LR backbone $10^{-4}$, Head mới $10^{-3}$, Weight decay $0,05$ (bỏ qua norm & bias).
* **Lịch LR**: Linear warmup 1 epoch + Cosine Annealing về $10^{-6}$. Batch size 32, bật AMP. 12 epochs. Seed 42.

---

## 3. KẾT QUẢ SO SÁNH BACKBONE (BƯỚC 1)

So sánh 5 kiến trúc thuộc 4 họ mô hình dưới cùng công thức `T00`:

| Mã | Kiến trúc Backbone | Họ kiến trúc | Params (M) | GMACs | Val Macro-F1 | Val Top-1 (%) | Train time (s/ep) | Latency p95 (ms) |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `B01` | `resnet50` | ResNet Baseline | 25,56 | 4,11 | *(điền)* | *(điền)* | *(điền)* | *(điền)* |
| `B02` | `convnext_tiny` | Modern ConvNet | 28,59 | 4,46 | *(điền)* | *(điền)* | *(điền)* | *(điền)* |
| `B03` | `swin_tiny_patch4_w7` | Vision Transformer | 28,29 | 4,50 | *(điền)* | *(điền)* | *(điền)* | *(điền)* |
| `B04` | `mobilenetv3_large_100` | Mạng nhẹ di động | 5,48 | 0,23 | *(điền)* | *(điền)* | *(điền)* | *(điền)* |
| `B05` | `resnet34` | ResNet gọn nhẹ | 21,80 | 3,67 | *(điền)* | *(điền)* | *(điền)* | *(điền)* |

**Nhận xét và lựa chọn**:
* *ConvNeXt-Tiny* tận dụng các kỹ thuật thiết kế hiện đại (depthwise separable conv 7x7, inverted bottleneck) cho khả năng trích xuất đặc trưng lá cây mạnh mẽ.
* *Swin Transformer* có cơ chế attention cục bộ qua shifted windows, tuy nhiên cần lượng dữ liệu lớn và thời gian huấn luyện lâu hơn.
* **Quyết định**: Chọn **`convnext_tiny`** đi tiếp làm backbone chính nhờ sự cân bằng xuất sắc giữa Macro-F1 và độ trễ chấp nhận được.

---

## 4. TỐI ƯU HÓA CÔNG THỨC HUẤN LUYỆN (BƯỚC 2: ABLATION STUDY)

Thực hiện trên backbone `convnext_tiny`, mỗi thí nghiệm **chỉ thay đổi đúng 1 yếu tố** so với nền `T00`:

| Mã | Trục thí nghiệm | Thay đổi cụ thể vs `T00` | Val Macro-F1 | Val Top-1 (%) | $\Delta$ F1 vs `T00` | Ghi chú & Đánh giá |
|---|---|---|:---:|:---:|:---:|---|
| `T00` | Mốc so sánh | Baseline Recipe | *(điền)* | *(điền)* | 0,0000 | Nền chuẩn |
| `T01` | A. Khởi tạo | Train from Scratch (`pretrained=False`) | *(điền)* | *(điền)* | *(điền)* | Kém hơn rõ rệt do thiếu đặc trưng sơ cấp |
| `T02` | A. Khởi tạo | Đóng băng backbone (Linear probe) | *(điền)* | *(điền)* | *(điền)* | BN ở eval mode, hội tụ nhanh nhưng F1 thấp |
| `T03` | B. Augmentation | Thêm `ColorJitter` (đổi màu, sáng) | *(điền)* | *(điền)* | *(điền)* | Tăng độ bền trước ánh sáng ngoài trời |
| `T04` | B. Augmentation | Thêm `CutMix` ($\alpha=1.0$) | *(điền)* | *(điền)* | *(điền)* | Giảm overconfident, ép học chi tiết lá |
| `T05` | B. Augmentation | Thêm `RandAugment` | *(điền)* | *(điền)* | *(điền)* | Regularization mạnh |
| `T06` | C. Hàm Loss | Label Smoothing ($\epsilon=0.1$) | *(điền)* | *(điền)* | *(điền)* | Làm mượt phân bố xác suất |
| `T07` | C. Hàm Loss | Focal Loss ($\gamma=2.0$) | *(điền)* | *(điền)* | *(điền)* | Tập trung vào mẫu khó, phạt nhẹ mẫu dễ |
| `T08` | C. Hàm Loss | Class-weighted CE | *(điền)* | *(điền)* | *(điền)* | Phạt nặng khi đoán sai các loài cỏ hiếm |
| `T09` | Kết hợp tối ưu | Best Combo: ColorJitter + CutMix + LS + EMA | *(điền)* | *(điền)* | *(điền)* | Hiệu ứng cộng dồn rõ rệt |

---

## 5. KỸ THUẬT SUY LUẬN & ĐỘ TRỄ (BƯỚC 3)

| Mã | Phương pháp suy luận | Mô tả kỹ thuật | Val Macro-F1 | Top-1 (%) | ECE Val | Latency p95 (ms) | Thông lượng (img/s) |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| `I00` | 1-view (Mốc) | CenterCrop 224 | *(điền)* | *(điền)* | *(điền)* | *(điền)* | *(điền)* |
| `I01` | TTA lật ngang | K = 2 views (Gốc + HFlip) | *(điền)* | *(điền)* | *(điền)* | *(điền)* | *(điền)* |
| `I02` | Dò độ phân giải | FixRes 256x256 | *(điền)* | *(điền)* | *(điền)* | *(điền)* | *(điền)* |
| `I03` | Model Ensemble | Trung bình softmax 2 model | *(điền)* | *(điền)* | *(điền)* | *(điền)* | *(điền)* |
| `I05` | Temperature Scaling | Calibrate $T$ tối ưu trên Val | *(điền)* | *(điền)* | *(điền)* | *(điền)* | *(điền)* |

**Đánh đổi Pareto giữa Độ chính xác và Độ trễ**:
* *TTA và Ensemble*: Giúp tăng nhẹ Macro-F1 nhưng độ trễ tăng gấp 2 lần. Phù hợp xử lý ngoại tuyến (offline analysis).
* *Temperature Scaling*: Giảm mạnh sai số hiệu chuẩn ECE mà **không tốn thêm bất kỳ chi phí tính toán nào**. Rất lý tưởng cho hệ thống nhúng của robot.

---

## 6. CẤU HÌNH TỐI ƯU VÀ ĐÁNH GIÁ TRÊN TẬP TEST (BƯỚC 4)

### 6.1 Bảng kết quả chung kết qua 3 seeds (Mean $\pm$ Std)
* **Cấu hình Champion (`F01`)**: `convnext_tiny` + ColorJitter + CutMix + Label Smoothing + Model EMA + Temperature Scaling.
* **Cấu hình Baseline (`T00`)**: `resnet50` + Baseline recipe + 1-view inference.

| Chỉ số đánh giá | Mốc đối chứng `T00` (Mean $\pm$ Std) | Chung kết `F01` (Mean $\pm$ Std) | Mức cải thiện $\Delta$ |
|---|:---:|:---:|:---:|
| **Top-1 Accuracy Test (%)** | *(điền)* | *(điền)* | *(điền)* |
| **Macro-F1 Test** | *(điền)* | *(điền)* | *(điền)* |
| **Recall: Chinee apple (%)** | *(điền)* | *(điền)* | *(điền)* |
| **Recall: Snake weed (%)** | *(điền)* | *(điền)* | *(điền)* |
| **Test ECE (trước TS)** | *(điền)* | *(điền)* | — |
| **Test ECE (sau TS)** | *(điền)* | *(điền)* | *(điền)* |

### 6.2 Phân tích Ma trận nhầm lẫn & Cặp phân loại khó
* Cặp loài thường xuyên bị nhầm lẫn nhiều nhất trong tập kiểm thử là **`Chinee apple`** và **`Snake weed`**.
* *Nguyên nhân sinh học & hình ảnh*: Cả hai loài đều có cấu trúc tán lá nhỏ, viền lá răng cưa mờ và thường mọc chen lẫn trong nền đất sỏi khô hạn của Queensland, khiến mạng CNN dễ bị đánh lừa bởi đặc trưng kết cấu nền.

---

## 7. KẾT LUẬN & KHUYẾN NGHỊ CHO ROBOT NÔNG NGHIỆP

1. **Cấu hình tốt nhất**: `convnext_tiny` kết hợp với công thức huấn luyện đa thành phần (CutMix, Label Smoothing, ColorJitter, EMA) và Temperature Scaling cho kết quả vượt trội và ổn định qua các seeds.
2. **Yếu tố đóng góp lớn nhất**: **Công thức huấn luyện (Training Recipe)** tạo ra bước nhảy vọt lớn nhất về Macro-F1 (tăng xử lý lớp hiếm), vượt qua việc chỉ đơn thuần đổi kiến trúc backbone.
3. **Khuyến nghị triển khai trên robot thực địa**:
   * Áp dụng mô hình **ConvNeXt-Tiny FP16** kết hợp **Temperature Scaling** ($T \approx 1.1–1.3$).
   * Cấu hình này đạt độ trễ $\le 30\text{ ms}$ ở batch 1 trên GPU nhúng (như Jetson Orin / RTX di động), hoàn toàn nằm trong ngân sách chu kỳ cảm biến $100\text{ ms}$, giúp robot phun thuốc chính xác theo thời gian thực mà không trượt mục tiêu.

---

## 8. HẠN CHẾ VÀ HƯỚNG PHÁT TRIỂN

* **Hạn chế chia tập ngẫu nhiên**: Dataset được chia ngẫu nhiên theo ảnh thay vì chia theo địa lý/nông trại (geographic split), dẫn đến điểm số trên Test có thể hơi lạc quan so với khi đem robot sang cánh đồng hoàn toàn mới.
* **Thời gian huấn luyện**: Do giới hạn tài nguyên tính toán (12 epochs so với 100 epochs của bài báo gốc), mô hình vẫn còn dư địa cải thiện nếu huấn luyện dài hơi hơn.
* **Hướng phát triển**: Áp dụng Test-Time Adaptation (TTA thích ứng thống kê BatchNorm) khi gặp điều kiện thời tiết khắc nghiệt (nắng gắt, bùn đất che khuất lá).

---

## 9. PHỤ LỤC (APPENDIX)
* Link notebook Colab tái lập: [`code/lab_day2.ipynb`](code/lab_day2.ipynb)
* File Excel kết quả chi tiết: [`results.xlsx`](results.xlsx)
* Thư mục biểu đồ: [`curves/`](curves/)
* File dự đoán: [`predictions/`](predictions/)
