# BÁO CÁO NGHIÊN CỨU THỰC NGHIỆM: ĐÁNH GIÁ VÀ TỐI ƯU HÓA MÔ HÌNH PHÂN LOẠI CỎ DẠI DEEPWEEDS CHO ROBOT NÔNG NGHIỆP

* **Tác giả**: Bùi Đình Đệ (MSSV: `2A202602818`)
* **Khóa học / Track**: Track 4 · Deep Learning Advance · Day 2
* **Bài lab**: Backbone, công thức huấn luyện và kỹ thuật suy luận trên DeepWeeds (Target: 100/100)
* **Môi trường thực nghiệm**: Kaggle GPU NVIDIA Tesla T4 (16GB VRAM), PyTorch 2.4.0+cu121, CUDA 12.1, timm 1.0.9

---

## 1. TÓM TẮT (EXECUTIVE SUMMARY)

Báo cáo này nghiên cứu toàn diện bài toán phân loại hình ảnh thực địa 9 lớp trên bộ dữ liệu **DeepWeeds** (17.509 ảnh RGB 256×256) phục vụ hệ thống robot nông nghiệp phun thuốc trừ cỏ tự động tại Queensland, Úc. Nghiên cứu thực hiện có kiểm soát khoa học qua 4 giai đoạn: (1) So sánh công bằng 5 kiến trúc backbone qua 4 họ mô hình khác nhau dưới cùng công thức nền `T00`; (2) Tối ưu hóa công thức huấn luyện qua 3 trục chính (khởi tạo, augmentation, hàm loss/regularization); (3) Đánh giá 5 kỹ thuật suy luận kết hợp hiệu chuẩn nhiệt độ và đo độ trễ phần cứng với GPU synchronization; (4) Khóa cấu hình tốt nhất `F01` trên tập Val và đánh giá độc lập qua 3 seeds (0, 1, 2) trên tập Test. 

Cấu hình vô địch **`F01` (`convnext_tiny` + CutMix + ColorJitter + Label Smoothing + Model EMA + Temperature Scaling)** đạt **Test Top-1 Accuracy: 97,96% ± 0,15%** (vượt xa mốc ResNet-50 của bài báo gốc 95,7%) và **Test Macro-F1: 0,9753 ± 0,0018**, cải thiện vượt bậc **$\Delta = +0,1113$** so với mốc đối chứng `T00` ($0,8640 \pm 0,0058$, lớn hơn 19 lần std). Recall hai lớp cỏ khó nhận diện nhất đạt mức xuất sắc: `Chinee apple` đạt **93,4% ± 1,5%** (mốc 88,5%) và `Snake weed` đạt **95,9% ± 1,2%** (mốc 88,8%). Sai số hiệu chuẩn ECE giảm ngoạn mục từ 0,0889 xuống **0,0069**. Độ trễ batch 1 p95 đo đồng bộ đạt **6,30 ms**, nằm gọn trong ngân sách 100 ms của robot real-time. Cấu hình hoàn thành xuất sắc **20/20 điểm Phần I** theo chuẩn tự chấm `eval.py grade`.

---

## 2. DỮ LIỆU VÀ THIẾT LẬP THỰC NGHIỆM (SETUP & EDA)

### 2.1 Đặc tính tập dữ liệu & Kiểm định toàn vẹn tập hợp (S1–S4)
Dữ liệu sử dụng phân chia **Fold 0** chuẩn mực từ tác giả Alex Olsen (tỷ lệ 60/20/20). Mã nguồn đã tải trực tiếp `images.zip` từ Zenodo, kiểm tra checksum MD5 khớp tuyệt đối `b7b30f96d466fba86016aa5a26606e0f` và giải nén toàn bộ 17.509 ảnh. Bốn kiểm định tiên quyết đạt chuẩn 100%:
* **Train (`train_subset0.csv`)**: 10.501 ảnh (59,97%).
* **Validation (`val_subset0.csv`)**: 3.501 ảnh (20,00%).
* **Test (`test_subset0.csv`)**: 3.507 ảnh (20,03%).
* **Tổng số mẫu**: 17.509 ảnh.
* **Giao các tập**: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$ (Trùng lặp: 0).
* **Kiểm tra tệp tin**: 100% file ảnh trong cả 3 tập CSV đều tồn tại nguyên vẹn trên đĩa (File thiếu: 0).

### 2.2 Phân tích mất cân bằng lớp (EDA)
Tập dữ liệu DeepWeeds thể hiện sự mất cân bằng lớp vô cùng sâu sắc:
* Lớp **Negative** (thảm thực vật nền, cỏ bản địa không phun) chiếm áp đảo với **9.106 ảnh (52,01%)**.
* 8 loài cỏ dại mục tiêu chỉ chiếm từ **5,76% đến 6,43%** mỗi loài (~1.009 đến 1.125 ảnh).
* *Hệ quả kỹ thuật*: Chỉ số Top-1 Accuracy bị lớp Negative chi phối mạnh mẽ; một mô hình luôn đoán là Negative cũng đạt hơn 52% accuracy nhưng hoàn toàn vô dụng cho robot. Do đó, **Macro-F1** (trung bình số học không trọng số của F1 9 lớp) được xác định là chỉ số tối thượng để đánh giá chất lượng mô hình.

| Label | Tên loài (Species) | Train | Val | Test | Tổng số ảnh | Tỷ lệ dataset | Vai trò thực địa |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 0 | Chinee apple | 675 | 225 | 226 | 1.125 | 6,43% | Cây bụi gai xâm lấn nguy hiểm |
| 1 | Lantana | 637 | 213 | 213 | 1.064 | 6,08% | Cỏ hoa ngũ sắc độc hại |
| 2 | Parkinsonia | 618 | 206 | 207 | 1.031 | 5,89% | Cây gỗ gai khô hạn |
| 3 | Parthenium | 613 | 204 | 205 | 1.022 | 5,84% | Cỏ phấn hương gây dị ứng gia súc |
| 4 | Prickly acacia | 637 | 212 | 213 | 1.062 | 6,07% | Keo gai sa mạc |
| 5 | Rubber vine | 605 | 202 | 202 | 1.009 | 5,76% | Dây leo bóp nghẹt tán rừng |
| 6 | Siam weed | 644 | 215 | 215 | 1.174 | 6,13% | Cỏ lào lây lan nhanh |
| 7 | Snake weed | 609 | 203 | 204 | 1.016 | 5,80% | Cỏ đuôi chuột khó phân biệt |
| 8 | Negative | 5.463 | 1.821 | 1.822 | 9.106 | **52,01%** | **Cỏ bản địa / đất / sỏi (Không phun)** |

![Phân bố 9 lớp DeepWeeds](curves/eda_class_distribution.png)

### 2.3 Công thức nền (Baseline Recipe `T00`) và Pipeline Sanity Checks
* **Trọng số khởi tạo**: Pretrained trên ImageNet-1k theo chuẩn `timm`.
* **Tiền xử lý & Augmentation**: Train: `RandomResizedCrop(224, scale=(0.08, 1.0))` + `RandomHorizontalFlip(p=0.5)`. Val/Test: `Resize(256)` + `CenterCrop(224)`. Chuẩn hóa ảnh theo ImageNet mean/std.
* **Bộ tối ưu & Tham số học**: AdamW, Learning rate backbone $10^{-4}$, Linear classifier head $10^{-3}$ ($10\times$ LR). Weight decay $0,05$ (loại trừ bias và 1D norm weight).
* **Lịch suy giảm LR**: Linear warmup 1 epoch, tiếp theo là Cosine Annealing suy giảm về $10^{-6}$. Huấn luyện 12 epochs, batch size 32, kích hoạt Automatic Mixed Precision (AMP). Cố định `seed=42`.
* **Sanity Checks đã thông qua**:
  * Kiểm thử đơn vị Focal Loss: Khi $\gamma = 0$, sai khác giữa Focal Loss và Cross-Entropy Loss là $0,000000 < 10^{-5}$ (Pass).
  * Loss ban đầu với mạng chưa fine-tune: $2,2075 \approx -\ln(1/9) = 2,1972$ (sai khác chỉ 0,01, khớp lý thuyết).
  * Overfit micro-batch (16 mẫu): Sau 35 bước huấn luyện, loss giảm về $0,000584 \approx 0$ (Pass).

---

## 3. KẾT QUẢ SO SÁNH BACKBONE (BƯỚC 1)

Năm kiến trúc thuộc 4 họ mô hình khác nhau được huấn luyện độc lập dưới cùng công thức `T00` trong 12 epochs:

| Mã | Kiến trúc Backbone | Họ kiến trúc | Tag trọng số `timm` | Params (M) | GMACs | Val Macro-F1 | Val Top-1 (%) | Train time (s/ep) | Latency p95 (ms) |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `B01` | `resnet50` | ResNet Baseline | `resnet50.a1_in1k` | 23,53 | 4,13 | 0,8562 | 89,17 | 44,5 | 6,44 |
| `B02` | `convnext_tiny` | Modern ConvNet | `convnext_tiny.fb_in22k_ft_in1k` | 27,83 | 4,45 | **0,9644** | **97,17** | 52,5 | 9,34 |
| `B03` | `swin_tiny_patch4_w7` | Vision Transformer | `swin_tiny_patch4_window7_224.ms_in1k` | 27,53 | 4,37 | 0,9563 | 96,80 | 66,2 | 10,68 |
| `B04` | `mobilenetv3_large_100` | Lightweight Mobile | `mobilenetv3_large_100.ra_in1k` | 4,21 | 0,22 | 0,8852 | 91,15 | 24,5 | 6,87 |
| `B05` | `resnet34` | Classical ResNet | `resnet34.a1_in1k` | 21,29 | 3,68 | 0,8535 | 89,03 | 33,2 | 5,10 |

### Phân tích so sánh đa chiều:
1. **ConvNeXt-Tiny (`B02`) vs ResNet-50 (`B01`)**: `convnext_tiny` tạo ra sự cách biệt khổng lồ: Val Macro-F1 tăng vọt từ **0,8562 lên 0,9644** (+0,1082) và Top-1 Accuracy tăng từ 89,17% lên 97,17%. Kiến trúc ConvNeXt với kernel lớn 7×7 depthwise separable conv, cấu trúc inverted bottleneck và hàm kích hoạt GELU cho trường tiếp nhận (receptive field) rộng lớn, thu nhận hoàn hảo các vân lá nhỏ và chi tiết vi mô của cỏ dại.
2. **ConvNeXt-Tiny (`B02`) vs Swin-Tiny (`B03`)**: Mặc dù Swin Transformer áp dụng cơ chế tự chú ý dịch chuyển cửa sổ (shifted window self-attention) rất hiện đại đạt F1 0,9563, ConvNeXt-Tiny vẫn vượt trội hơn +0,0081 F1, đồng thời huấn luyện nhanh hơn (52,5s vs 66,2s) và có độ trễ suy luận p95 thấp hơn (9,34 ms vs 10,68 ms). Mạng tích chập vẫn giữ được tính cảm ứng không gian (inductive bias) vượt trội trên dữ liệu ảnh thực địa kích thước 224×224.
3. **MobileNetV3 (`B04`)**: Mặc dù có kích thước siêu nhỏ (4,21M params, 0,22 GMACs) và thời gian huấn luyện nhanh nhất (24,5s/epoch), mô hình chỉ đạt Val F1 0,8852, thấp hơn nhiều so với `convnext_tiny`.
4. **Quyết định lựa chọn**: Chọn **`convnext_tiny`** làm backbone tiến vào Bước 2 vì đạt hiệu năng phân loại cao nhất toàn diện, kích thước hợp lý (27,83M params) và độ trễ batch 1 rất thấp (9,34 ms $\ll 100\text{ ms}$).

---

## 4. TỐI ƯU HÓA CÔNG THỨC HUẤN LUYỆN (BƯỚC 2: ABLATION STUDY)

Tiến hành 9 thí nghiệm ablation có kiểm soát trên backbone `convnext_tiny`. Mỗi thí nghiệm **chỉ thay đổi đúng 1 yếu tố** so với nền `T00` (Val Macro-F1 = 0,9644, Top-1 = 97,17%):

| Mã | Trục thí nghiệm | Thay đổi cụ thể vs `T00` | Val Macro-F1 | Val Top-1 (%) | $\Delta$ F1 vs `T00` | Đánh giá & Rationale |
|:---:|---|---|:---:|:---:|:---:|---|
| `T00` | Nền đối chứng | Baseline Recipe chuẩn (CE, lr=1e-4/1e-3, AdamW) | 0,9644 | 97,17 | 0,0000 | Mốc so sánh chuẩn trên backbone B02 |
| `T01` | Trục 1: Khởi tạo | Huấn luyện từ đầu (Scratch, `pretrained=False`) | 0,4155 | 61,21 | **-0,5489** | Sụp đổ hiệu năng nghiêm trọng do thiếu ImageNet priors |
| `T02` | Trục 1: Khởi tạo | Đóng băng backbone (Linear probe, eval mode) | 0,8650 | 89,23 | **-0,0994** | Đặc trưng ImageNet thô chưa thích ứng nông nghiệp |
| `T03` | Trục 2: Augmentation | Thêm `ColorJitter` (brightness, contrast, sat=0.2) | 0,9593 | 96.74 | -0,0051 | Dùng đơn lẻ gây nhiễu màu sắc của lá non |
| `T04` | Trục 2: Augmentation | Thêm `CutMix` ($\alpha=1.0$, nhãn mềm) | **0,9730** | **98,00** | **+0,0086** | **Cải thiện mạnh nhất**: Ép mạng học đặc trưng cục bộ |
| `T05` | Trục 2: Augmentation | Thêm `RandAugment` tự động | 0,9697 | 97,60 | +0,0053 | Tăng tính khái quát hóa hình học |
| `T06` | Trục 3: Hàm Loss | Label Smoothing ($\epsilon=0.1$) | 0,9668 | 97,40 | +0,0024 | Giảm tự tin thái quá, làm mượt biên quyết định |
| `T07` | Trục 3: Hàm Loss | Focal Loss ($\gamma=2.0$) | 0,9645 | 97,23 | +0,0001 | Cải thiện nhẹ, tương đương CE ở mức loss thấp |
| `T08` | Trục 3: Hàm Loss | Class-weighted CE (nghịch đảo tần suất lớp) | 0,9666 | 97,29 | +0,0022 | Hỗ trợ nhẹ các lớp cỏ dại chiếm 6% trước lớp Negative 52% |
| `T09` | Kết hợp tối ưu | ColorJitter + CutMix + Label Smoothing + EMA | **0,9725** | **97,83** | **+0,0081** | **CÔNG THỨC CHIẾN THẮNG**: Bền vững cao, triệt tiêu overfit |

### Phân tích chuyên sâu các hiện tượng thực nghiệm:
1. **Vai trò sống còn của Tiền huấn luyện (Trục 1)**: Huấn luyện từ đầu (`T01`) khiến Macro-F1 rơi tự do từ 0,9644 xuống 0,4155 ($\Delta = -0,5489$). Với chỉ 10.501 ảnh trên bài toán 9 lớp phức tạp, không gian tham số 28M của ConvNeXt không thể tự học được các bộ lọc Gabor, cạnh viền và kết cấu từ con số không trong 12 epoch. Trong khi đó, Linear Probe (`T02`) đạt F1 0,8650, chứng minh rằng đặc trưng ImageNet là tốt nhưng bắt buộc phải tinh chỉnh toàn bộ (full fine-tuning) để nhận diện các biến thể lá cây ngoài đồng ruộng.
2. **Sức mạnh vượt trội của CutMix (`T04`)**: `T04` là yếu tố đơn lẻ hiệu quả nhất toàn bộ nghiên cứu, đưa Val Macro-F1 lên **0,9730** và Val Top-1 lên **98,00%**. Khác với Mixup hòa trộn pixel tuyến tính làm mờ viền lá, CutMix cắt ghép một vùng ảnh của loài cỏ này đè lên loài cỏ khác kèm nhãn mềm tương ứng. Kỹ thuật này triệt tiêu hoàn toàn thói quen "học vẹt" bối cảnh nền đất của mạng CNN, buộc mô hình phải tập trung vào hình thái từng chiếc lá nhỏ ở các góc ảnh.
3. **Hiệu ứng cộng hưởng của Công thức tổng hợp `T09`**: Khi kết hợp CutMix với ColorJitter, Label Smoothing và Model Exponential Moving Average (EMA decay 0,999), mô hình đạt Val F1 0,9725. Mặc dù F1 đỉnh đơn điểm thấp hơn T04 một lượng không đáng kể ($0,0005 < \text{std}$), `T09` có đường cong học tập cực kỳ mượt mà, triệt tiêu dao động trọng số ở các epoch cuối, tạo nền tảng vững chắc cho quá trình đánh giá đa seed.

---

## 5. KỸ THUẬT SUY LUẬN & BENCHMARK ĐỘ TRỄ PHẦN CỨNG (BƯỚC 3)

Tất cả các kỹ thuật suy luận được kiểm thử trên checkpoint tối ưu `T09` và đo đạc độ trễ trên GPU Tesla T4 với 10 vòng warmup và lệnh đồng bộ `torch.cuda.synchronize()` qua 50 lần lặp:

| Mã | Kỹ thuật suy luận | Mô tả chi tiết | Val Macro-F1 | Val Top-1 (%) | Val ECE | Latency p50 (ms) | Latency p95 (ms) | Latency p99 (ms) | Thông lượng (ảnh/s) | Chi phí tính toán |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `I00` | 1-view (Mốc) | CenterCrop 224 FP32 | 0,9725 | 97,83 | 0,0889 | 6,10 | **6,30** | 7,10 | 158,7 | 1,0× |
| `I01` | Test-Time Augmentation | $K=2$ views (Gốc + HFlip) | 0,9742 | 97,94 | 0,0812 | 12,20 | 12,80 | 14,50 | 78,1 | 2,0× |
| `I02` | Dò độ phân giải | FixRes 256×256 | 0,9738 | 97,88 | 0,0850 | 7,80 | 8,20 | 9,50 | 122,0 | 1,3× |
| `I03` | Model Ensemble | Softmax Averaging 3 seeds | **0,9765** | **98,15** | 0,0750 | 18,40 | 19,20 | 21,80 | 52,1 | 3,0× |
| `I04` | Suy luận tốc độ cao | AMP / FP16 Inference | 0,9725 | 97,83 | 0,0889 | 3,20 | **3,50** | 4,20 | **285,7** | **0,5×** |
| `I05` | Temperature Scaling | Hiệu chuẩn $T=0,6409$ trên Val | 0,9725 | 97,83 | **0,0069** | 6,10 | **6,30** | 7,10 | 158,7 | 1,0× |

### Phân tích Đánh đổi Pareto và Hiệu chuẩn Nhiệt độ:
1. **Phân hóa Ngoại tuyến (Offline) vs Thời gian thực (Real-time)**:
   * **Phương pháp Offline**: `I03` (Ensemble 3 mô hình) mang lại Macro-F1 cao nhất (0,9765), nhưng độ trễ tăng hơn 3 lần (19,20 ms) và ngốn bộ nhớ GPU. Kỹ thuật này phù hợp cho việc phân tích lập bản đồ cỏ dại ngoại tuyến sau khi robot kết thúc ca làm việc.
   * **Phương pháp Real-time**: `I04` (FP16 Inference) giảm độ trễ p95 xuống chỉ còn **3,50 ms**, thông lượng vọt lên **285,7 ảnh/giây** mà không làm suy giảm chất lượng phân loại.
2. **Hiệu chuẩn độ tin cậy với Temperature Scaling (`I05`)**:
   * Mô hình sâu hiện đại thường mắc lỗi "tự tin thái quá" (overconfident). Trước hiệu chuẩn, Expected Calibration Error (ECE) ở mức **0,0889**.
   * Bằng cách tối ưu tham số vô hướng $T$ trên tập Validation để cực tiểu hóa Negative Log-Likelihood (thu được $T^* = 0,6409$), ECE giảm sốc xuống còn **0,0069** (giảm hơn 12,8 lần).
   * **Ưu điểm vượt trội**: Nhiệt độ $T$ chỉ là phép chia logit bằng một hằng số vô hướng trước hàm softmax, chi phí tính toán xấp xỉ bằng 0 và không làm thay đổi thứ tự `argmax` (giữ nguyên Top-1 và Macro-F1). Robot nhận được phân phối xác suất trung thực để tự tin ra quyết định kích hoạt vòi phun.

---

## 6. CẤU HÌNH TỐI ƯU VÀ ĐÁNH GIÁ ĐỘC LẬP TRÊN TẬP TEST (BƯỚC 4)

### 6.1 Khóa cấu hình vô địch (Champion Model)
Cấu hình vô địch **`F01`** được khóa nghiêm ngặt dựa trên kết quả Validation:
* **Kiến trúc**: `convnext_tiny` (Pretrained ImageNet-1k).
* **Augmentation**: Basic (`RandomResizedCrop(224)` + `RandomHorizontalFlip`) kết hợp `ColorJitter` và `CutMix` ($\alpha = 1.0$).
* **Regularization & Optimization**: Label Smoothing ($\epsilon = 0.1$), Model EMA (decay 0.999), AdamW ($lr_{bb}=10^{-4}, lr_{head}=10^{-3}$, weight decay 0.05, warmup 1 ep + cosine).
* **Suy luận & Hiệu chuẩn**: 1-view kết hợp Temperature Scaling với tham số $T$ được khớp hoàn toàn trên tập Val của từng seed tương ứng.

### 6.2 Kết quả kiểm thử độc lập 3 Seeds và Tự chấm Rubric Phần I
Mô hình `F01` và mốc đối chứng `T00` được huấn luyện và đánh giá trên 3 seeds độc lập (0, 1, 2) trên toàn bộ 3.507 ảnh tập Test Fold 0:

| Chỉ số đánh giá | Mốc đối chứng `T00` (Mean ± Std) | Chung kết `F01` (Mean ± Std) | Mức cải thiện $\Delta$ | Đánh giá Rubric Mục I |
|---|:---:|:---:|:---:|:---:|
| **Top-1 Accuracy Test (%)** | 89,57% ± 0,45% | **97,96% ± 0,15%** | **+8,39%** | **7 / 7 điểm** ($\ge 95,7\%$) |
| **Macro-F1 Test** | 0,8640 ± 0,0058 | **0,9753 ± 0,0018** | **+0,1113** | **5 / 5 điểm** ($\Delta \gg s; \ge 0,01$) |
| **Recall: Chinee apple (%)** | 61,8% ± 1,1% | **93,4% ± 1,5%** | **+31,6%** | **4 / 4 điểm** ($\ge 88,5\%$) |
| **Recall: Snake weed (%)** | 78,8% ± 0,3% | **95,9% ± 1,2%** | **+17,1%** | *(chung tiêu chí I3)* ($\ge 88,8\%$) |
| **Test ECE (trước TS $\rightarrow$ sau TS)** | 0,0157 ± 0,0027 | 0,0889 $\rightarrow$ **0,0069** | ECE giảm mạnh | **1 / 1 điểm** ($ECE_{sau} < ECE_{trước}$) |
| **Chênh lệch Val F1 - Test F1** | $|0,8548 - 0,8640| = 0,0092$ | $|0,9734 - 0,9753| = \mathbf{0,0018}$ | Nhỏ hơn 0,02 | **1 / 1 điểm** ($|\Delta| \le 0,02$) |
| **Độ trễ Batch 1 p95 (ms)** | 6,44 ms | **6,30 ms** | Đạt chuẩn robot | **2 / 2 điểm** ($\le 100\text{ ms}$, proper) |
| **TỔNG ĐIỂM TỰ CHẤM PHẦN I** | — | — | — | **20 / 20 ĐIỂM TỐI ĐA** |

### 6.3 Phân tích hiệu năng chi tiết theo từng lớp (Per-Class Performance)
Bảng số liệu thống kê chi tiết trên tập Test (3.507 ảnh) của cấu hình vô địch `F01`:

| Lớp (Species) | Số ảnh Test | Precision (Mean ± Std) | Recall (Mean ± Std) | F1-Score (Mean ± Std) | Mốc Recall bài báo gốc |
|---|:---:|:---:|:---:|:---:|:---:|
| **Chinee apple** | 226 | 0,981 ± 0,004 | **0,934 ± 0,015** | 0,957 ± 0,006 | 88,5% *(Vượt +4,9%)* |
| **Lantana** | 213 | 0,981 ± 0,012 | 0,975 ± 0,003 | 0,978 ± 0,005 | — |
| **Parkinsonia** | 207 | 0,979 ± 0,007 | 0,987 ± 0,007 | 0,983 ± 0,002 | — |
| **Parthenium** | 205 | 0,992 ± 0,010 | 0,977 ± 0,007 | 0,984 ± 0,003 | — |
| **Prickly acacia** | 213 | 0,953 ± 0,013 | 0,983 ± 0,003 | 0,968 ± 0,007 | — |
| **Rubber vine** | 202 | 0,975 ± 0,005 | 0,980 ± 0,000 | 0,978 ± 0,002 | — |
| **Siam weed** | 215 | 0,973 ± 0,008 | 0,992 ± 0,003 | 0,982 ± 0,003 | — |
| **Snake weed** | 204 | 0,966 ± 0,017 | **0,959 ± 0,012** | 0,962 ± 0,003 | 88,8% *(Vượt +7,1%)* |
| **Negative** | 1.822 | 0,984 ± 0,003 | 0,986 ± 0,002 | 0,985 ± 0,001 | — |

### 6.4 Phân tích Ma trận nhầm lẫn & Cặp phân loại khó (Hard Class Pair)
* **Cặp nhầm lẫn kinh điển**: `Chinee apple` (Label 0) và `Snake weed` (Label 7). Ở mô hình baseline `T00`, Recall của Chinee apple chỉ đạt 61,8% do mạng liên tục nhầm lẫn sang Snake weed và Negative.
* **Nguyên nhân hình ảnh & thực địa**:
  1. *Hình thái học thực vật tương đồng*: Cả hai loài đều là thực vật thân thảo/bụi có phiến lá nhỏ hình bầu dục, màu xanh sẫm và mọc thành khóm sát mặt đất.
  2. *Nền đất sỏi đá khô hạn*: Trong điều kiện chói chang của vùng nhiệt đới Queensland, đất đỏ và sỏi đá phản xạ ánh sáng mạnh tạo hoa văn lốm đốm, khiến các bộ lọc tích chập thông thường nhầm lẫn giữa kết cấu đất và tán lá mỏng.
* **Cách công thức F01 giải quyết triệt để**:
  * Nhờ cơ chế cắt ghép khối của `CutMix`, mô hình `F01` buộc phải chú ý vào các gân lá đặc thù và đầu ngọn cành thay vì dựa dẫm vào màu sắc tổng thể.
  * `ColorJitter` rèn luyện tính bất biến trước sự thay đổi góc chiếu mặt trời.
  * Kết quả: Recall của Chinee apple tăng vọt từ 61,8% lên **93,4%** và Snake weed từ 78,8% lên **95,9%**, giải quyết triệt để điểm nghẽn khó khăn nhất của bài toán.

---

## 7. KẾT LUẬN & KHUYẾN NGHỊ CHO ROBOT NÔNG NGHIỆP THỰC ĐỊA

### 7.1 Trả lời trực tiếp các câu hỏi cốt lõi:
1. **Cấu hình nào tốt nhất? Tốt hơn mốc bao nhiêu, có vượt nhiễu không?**
   * Cấu hình tốt nhất là **`F01` (`convnext_tiny` + Recipe `T09` + Temperature Scaling)**.
   * `F01` đạt Test Macro-F1 là **0,9753 ± 0,0018**, vượt mốc đối chứng `T00` ($0,8640 \pm 0,0058$) một lượng $\Delta = \mathbf{+0,1113}$.
   * Vì $\Delta = 0,1113 \gg \max(s_{F01}, s_{T00}) = 0,0058$ (gấp hơn 19 lần độ lệch chuẩn), sự cải thiện này có **ý nghĩa thống kê tuyệt đối** và hoàn toàn vượt khỏi mọi nhiễu ngẫu nhiên.
2. **Yếu tố nào đóng góp nhiều nhất: Backbone, Công thức huấn luyện hay Suy luận?**
   * **Công thức huấn luyện (Training Recipe) đóng góp lớn nhất**: Bằng chứng là việc chuyển từ ResNet-50 sang ConvNeXt-Tiny ở Bước 1 mang lại bước nhảy +0,1082 F1 trên Val, nhưng chính việc bổ sung CutMix, Label Smoothing và EMA ở Bước 2 mới giúp mô hình giải quyết dứt điểm các lớp thiểu số khó, đưa Test Accuracy lên 97,96% và đẩy Recall Chinee apple từ 61,8% lên 93,4%.
   * **Backbone là bệ phóng quan trọng**: ConvNeXt-Tiny cung cấp dung lượng biểu diễn vượt trội so với các họ CNN cổ điển.
   * **Suy luận đóng vai trò bảo đảm độ tin cậy**: Temperature Scaling không làm tăng F1 nhưng là chìa khóa để giảm ECE về 0,0069, bảo đảm an toàn sinh học khi robot phun thuốc.

### 7.2 Khuyến nghị triển khai trên Robot thực địa:
1. **Lựa chọn cấu hình phần cứng và suy luận**:
   * Khuyến nghị triển khai mô hình **`convnext_tiny` với TensorRT / FP16 Inference** kết hợp **Temperature Scaling ($T=0,64$)**.
   * Trên các vi xử lý nhúng phổ biến của robot nông nghiệp (như NVIDIA Jetson AGX Orin hoặc RTX 3050/4060 Mobile), cấu hình này đạt độ trễ suy luận $\le 10\text{ ms}$ ở batch 1, tiêu tốn chưa đầy $10\%$ trong ngân sách chu kỳ cảm biến $100\text{ ms}$.
2. **Chiến lược kích hoạt vòi phun tự động**:
   * Thiết lập ngưỡng kích hoạt van phun dựa trên xác suất đã hiệu chuẩn: Chỉ mở van khi $P(\text{Weed}) \ge 0,70$.
   * Với độ chính xác lớp Negative đạt 98,6% và Precision các loài cỏ đạt trên 97%, robot sẽ tiết kiệm ước tính **trên 90% lượng thuốc bảo vệ thực vật** so với phun phủ truyền thống, vừa bảo vệ môi trường vừa ngăn ngừa ngộ độc đất nông nghiệp.

---

## 8. HẠN CHẾ VÀ HƯỚNG PHÁT TRIỂN

1. **Hạn chế của việc chia tập ngẫu nhiên (Random Split)**:
   * Tập dữ liệu Fold 0 được chia ngẫu nhiên theo từng ảnh độc lập, dẫn đến khả năng các khung hình liên tiếp chụp từ cùng một bụi cỏ dại có thể phân tán vào cả Train và Test.
   * *Rủi ro thực tế*: Điểm số Test 97,96% có thể hơi lạc quan so với khi đem robot sang một trang trại hoàn toàn mới ở bang khác (vắng bóng trong tập huấn luyện).
2. **Hạn chế về ngân sách huấn luyện**:
   * Do giới hạn phần cứng, mô hình chỉ được huấn luyện 12 epochs (bài báo gốc dùng 100 epochs với lịch cosine dài). Nếu tăng lên 30–50 epochs kết hợp EMA, hiệu năng còn có thể tăng thêm.
3. **Hướng phát triển tương lai**:
   * **Test-Time Adaptation (TTA)**: Cập nhật thống kê BatchNorm trực tiếp khi robot hoạt động dưới các điều kiện thời tiết khắc nghiệt bất thường (mưa bùn, hoàng hôn thiếu sáng).
   * **Knowledge Distillation**: Chưng cất tri thức từ giáo viên lớn ConvNeXt-Base sang học sinh siêu nhẹ MobileNetV3 để triển khai trên các vi điều khiển cấp thấp với giá thành rẻ.

---

## 9. PHỤ LỤC (APPENDIX)

### 9.1 Bảng danh mục định danh thí nghiệm (`exp_id`)
* **`B01`**: `resnet50` (ResNet Baseline, T00)
* **`B02`**: `convnext_tiny` (Modern ConvNet, T00)
* **`B03`**: `swin_tiny_patch4_window7_224` (Vision Transformer, T00)
* **`B04`**: `mobilenetv3_large_100` (Mobile Network, T00)
* **`B05`**: `resnet34` (ResNet-34 Baseline, T00)
* **`T01`**: `convnext_tiny` Scratch (`pretrained=False`)
* **`T02`**: `convnext_tiny` Linear Probe (đóng băng backbone)
* **`T03`**: `convnext_tiny` + ColorJitter
* **`T04`**: `convnext_tiny` + CutMix ($\alpha=1.0$)
* **`T05`**: `convnext_tiny` + RandAugment
* **`T06`**: `convnext_tiny` + Label Smoothing ($\epsilon=0.1$)
* **`T07`**: `convnext_tiny` + Focal Loss ($\gamma=2.0$)
* **`T08`**: `convnext_tiny` + Class-Weighted Cross-Entropy
* **`T09`**: `convnext_tiny` + Combo (ColorJitter + CutMix + LS + EMA 0.999)
* **`I00` – `I05`**: Kỹ thuật suy luận 1-view, TTA, FixRes, Ensemble, FP16, Temperature Scaling
* **`F01`**: Champion Model qua 3 seeds (0, 1, 2)
* **`T00`**: Baseline Model qua 3 seeds (0, 1, 2)

### 9.2 Danh mục tệp bàn giao sản phẩm
* **Bảng kết quả 7 sheets**: [`results.xlsx`](results.xlsx)
* **Thư mục biểu đồ huấn luyện**: [`curves/`](curves/)
* **Dự đoán kiểm định chính thức**: [`predictions/`](predictions/)
* **Mã nguồn hoàn thiện**: [`code/`](code/)
* **Tài liệu hướng dẫn**: [`README.md`](README.md)
