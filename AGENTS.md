# AGENTS.md — DeepWeeds Operational SOP & Execution Guide

Concise, high-density protocol for AI Agents and Engineers executing the Track 4 DeepWeeds project under [`RUBRIC.md`](RUBRIC.md) (Target: 100/100 + 10 Bonus).

---

## 1. Problem Overview & Core Metrics

* **Task**: 9-class field image classification on **DeepWeeds** (17,509 RGB 256×256 images) for autonomous agricultural weed-spraying robots in Queensland, Australia.
* **Classes**: 8 target weed species (`Chinee apple`, `Lantana`, `Parkinsonia`, `Parthenium`, `Prickly acacia`, `Rubber vine`, `Siam weed`, `Snake weed`) + 1 `Negative` class (non-target vegetation).
* **Imbalance**: `Negative` = 9,106 images (~52%); each weed species = 1,009–1,125 (~6%). Top-1 Accuracy is inflated by Negative.
* **Primary Metric**: **Macro-F1** (unweighted mean across 9 classes).
* **Hardware Constraints**: Batch 1 latency p95 $\le 100\text{ ms}$ (robot real-time sensor budget). Local GPU: RTX 3050 (4GB VRAM) $\rightarrow$ Use AMP, Batch Size 16/32.

---

## 2. Inviolable Rules & Preconditions (P1–P4)

1. **[P1] Required Deliverables**: `results.xlsx` (7 sheets), `report.md` (9 sections), `curves/` (all exp_ids), `code/` (completed starter), `README.md`, `predictions/` (CSV files).
2. **[P2] 100% Genuine Empirical Data**: Every metric must trace back to an `exp_id`, training log, and curve. Zero synthetic data.
3. **[P3] Strict Fold 0 Usage**: Use `train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`. Do NOT alter CSVs, merge Val into Train, or train on Test.
4. **[P4] Standard Prediction Format**: Save `predictions/<exp_id>_seed<k>_test.csv` with columns: `Filename, y_true, y_pred, p0..p8` for validation via `eval.py`.
5. **No Test Leakage**: All architecture, hyperparameter, and temperature decisions MUST be finalized strictly on **Val**. Test is evaluated **ONCE per seed** at the final stage.
6. **Scientific Rigor**: One change per ablation. When comparing models, if $\Delta < \text{std}$, declare "indistinguishable". Never commit images or model weights to Git.

---

## 3. Checkpoint Execution Workflow (CP0 → CP5)

### CP 0: Environment, Data, EDA & Sanity Checks (Rubric A: 12 pts)
1. **Setup**: Python venv, PyTorch with CUDA, `timm`, `pandas`, `scikit-learn`, `openpyxl`, `matplotlib`, `seaborn`.
2. **Checksum & Extract**: Verify MD5 of `images.zip` (`b7b30f96d466fba86016aa5a26606e0f`). Extract to `data/images/`. Download Fold 0 CSVs to `data/labels/`.
3. **Split Checks (S1–S4)**:
   * Counts: Train ~10,505, Val ~3,502, Test ~3,502 (Sum = 17,509).
   * Disjoint: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
   * File existence: All CSV entries exist in `data/images/`.
4. **EDA**: Plot class distribution bar chart (highlight Negative 52% vs weeds 6%). Inspect $\ge 3$ samples per class. Note visual confusion pairs (`Chinee apple` vs `Snake weed`).
5. **Pipeline Sanity Checks**:
   * Seed fix (`random`, `numpy`, `torch`, workers).
   * Initial CE Loss $\approx -\ln(1/9) \approx 2.197$.
   * Overfit single micro-batch (8–16 images) to near 0 loss in 20–30 steps.
   * Verify input pipeline: Visualize de-normalized post-augmentation images with labels.

### CP 1: Fair Backbone Comparison (Rubric B: 12 pts)
* **Select $\ge 5$ backbones** across 4 required architectural families:
  1. *ResNet baseline*: `resnet50` (or `resnet34`).
  2. *Modern ConvNet*: `convnext_tiny` (or `resnext50_32x4d`).
  3. *Vision Transformer*: `swin_tiny_patch4_window7_224` (or `deit_small_patch16_224`).
  4. *Lightweight*: `efficientnet_b0` (or `mobilenetv3_large_100`).
  5. *Additional*: `regnetx_002` or `resnet34`.
* **Fix Baseline Recipe `T00`**:
  * Weights: ImageNet pretrained (record exact `timm` weight tag).
  * Inputs: Train `RandomResizedCrop(224)` + `RandomHorizontalFlip()`. Val: `Resize(256)` + `CenterCrop(224)`.
  * Optimization: `AdamW`, Backbone LR $10^{-4}$, Head LR $10^{-3}$ ($10\times$). Weight decay $0.05$ (exclude bias & norm). Warmup 1 epoch + Cosine Annealing. Batch size 32, AMP enabled, 10–12 epochs. Seed 42.
* **Metrics**: Params (M), GMACs, Val Macro-F1, Val Top-1, train time/epoch, batch 1 latency.
* **Selection**: Choose 1–2 best backbones with multi-dimensional rationale (Quality vs Size vs Latency). Save curves `curves/B0x_<name>.png`.

### CP 2: Training Recipe Optimization (Rubric C: 16 pts)
Perform controlled ablations (1 factor changed at a time vs `T00`):
* **Axis 1 (Initialization)**: `T00` (Fine-tune all) vs `T01` (Scratch) vs `T02` (Linear probe / freeze backbone with BN in eval mode).
* **Axis 2 (Augmentation)**: `T00` (Basic) vs `T03` (+ColorJitter) vs `T04` (+CutMix/Mixup with soft labels) vs `T05` (+RandAugment).
* **Axis 3 (Loss Function)**: `T00` (CE) vs `T06` (Label Smoothing $\epsilon=0.1$) vs `T07` (Focal Loss $\gamma=2$; verify $\gamma=0 \equiv \text{CE}$) vs `T08` (Class-weighted CE).
* **Optional Axes**: `T09` (Model EMA decay 0.999), `T10` (Head LR = Backbone LR).
* **Combination**: Compose top winning factors (e.g., Best Backbone + CutMix + Focal + EMA). Evaluate if gains are additive or conflicting. Save `curves/T0x_<desc>.png`.

### CP 3: Inference Strategy & Latency Benchmarking (Rubric D: 12 pts)
Evaluate $\ge 4$ post-training inference techniques (no retraining):
* `I00`: Baseline 1-view (`Resize 256` $\rightarrow$ `CenterCrop 224`).
* `I01`: Horizontal Flip TTA ($K=2$).
* `I02`: Resolution Tuning / FixRes (test at 224, 256, 288).
* `I03`: Model Ensemble (Softmax probability averaging across models/seeds).
* `I04`: Conv-BN Fusion + FP16/AMP inference.
* `I05`: **Temperature Scaling & Calibration**: Fit $T > 0$ on **Val** to minimize NLL. Compute 15-bin ECE before & after.
* **Rigorous Latency Measurement**:
  * Warmup $\ge 10$ runs; synchronize with `torch.cuda.synchronize()`; measure $\ge 50$ iterations.
  * Report **p50, p95, p99 (ms)** for **Batch = 1** (real-time robot) and **Batch = 32** (throughput img/s).
* **Pareto Curve**: Plot Val Macro-F1 vs p95 Latency (ms). Categorize offline vs real-time solutions.

### CP 4: Finals & Independent Test Evaluation (Rubric I: 20 pts)
1. **Lock Champion Configuration** strictly based on Val Macro-F1.
2. **Train across $\ge 3$ seeds**: Run final configuration (`F01`) and baseline (`T00` + `I00`) on seeds 0, 1, 2.
3. **Execute Single Test Pass**: Run test inference on full Fold 0 test split (3,502 images).
4. **Save Predictions**:
   * `predictions/F01_seed{0,1,2}_test.csv`
   * `predictions/T00_seed{0,1,2}_test.csv`
   * Val & uncalibrated files for calibration grading: `predictions/F01_seed*_val.csv`, `predictions/F01_uncal_seed*_test.csv`.
5. **Auto-Grade via `eval.py`**:
   ```bash
   python eval.py score --pred "predictions/F01_seed*_test.csv" \
       --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out

   python eval.py grade \
       --final "predictions/F01_seed*_test.csv" \
       --baseline "predictions/T00_seed*_test.csv" \
       --uncal "predictions/F01_uncal_seed*_test.csv" \
       --final-val "predictions/F01_seed*_val.csv" \
       --latency-p95-ms <MEASURED_P95> --latency-method proper \
       --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv
   ```
* **Score Targets for Part I (20 pts)**:
  * **I1 (7 pts)**: Test Top-1 Acc $\ge 95.7\%$ (or $\ge 95.1\%$ for 6 pts).
  * **I2 (5 pts)**: $\Delta \text{Macro-F1} > \text{std}$ and $\ge 0.01$ over baseline.
  * **I3 (4 pts)**: Recall on both `Chinee apple` and `Snake weed` $\ge 88.5\%$.
  * **I4 (2 pts)**: Post-scaling test ECE decreased and $|\text{Val F1} - \text{Test F1}| \le 0.02$.
  * **I5 (2 pts)**: Real-time profile with batch 1 p95 $\le 100\text{ ms}$ properly measured.

### CP 5: Deliverables, Artifacts & Documentation (Rubric E, F, G, H: 24 pts)
1. **`results.xlsx` (8 pts)**: Complete all 7 sheets: `Backbones`, `Training`, `Inference`, `Final`, `PerClass`, `Latency`, `Summary`. Uniform 4-decimal formatting, freeze header row, include units and $\text{mean} \pm \text{std}$.
2. **`curves/` (4 pts)**: Separate PNG per `exp_id` showing train/val loss & val Macro-F1 over epochs.
3. **`report.md` (12 pts)**: 9 structured sections: Executive Summary, Setup/EDA, Backbone Comparison, Training Ablation, Inference & Latency, Champion Analysis & Error Breakdown (Confusion matrix & hard-pair analysis), Robot Recommendations, Limitations, Appendix.
4. **Code Quality (4 pts)**: Finish all stubs in `starter/` (migrated to `code/`). Zero `NotImplementedError`. Pass unit tests: `python -m unittest discover -s tests -v`.

---

## 4. Bonus Points Roadmap (+10 pts Max)

| Item | Description | Pts |
|---|---|---|
| **Bonus 1** | Linear probe on frozen DINOv2 (`dinov2_vits14`) vs fine-tuned CNN | +2 |
| **Bonus 2** | Robustness under Distribution Shift (Gaussian blur / brightness shift test sets; compare ECE & F1) | +2 |
| **Bonus 3** | Grad-CAM error visualization on `Chinee apple` vs `Snake weed` confusion | +1 |
| **Bonus 4** | Export champion model to ONNX & benchmark latency vs PyTorch FP32 | +1 |
| **Bonus 5** | Knowledge distillation (large teacher $\rightarrow$ light student) | +2 |

---

## 5. Submission Pre-Flight Checklist

- [ ] Dataset integrity confirmed (MD5 verified, S1–S4 passed, zero train/val/test overlap).
- [ ] $\ge 5$ backbones tested under uniform baseline `T00` with recorded weight tags.
- [ ] $\ge 3$ training ablation axes executed with single-variable control.
- [ ] $\ge 4$ inference techniques benchmarked with synchronized latency (p50/p95/p99).
- [ ] Champion model selected on Val, trained on $\ge 3$ seeds, tested ONCE per seed.
- [ ] `eval.py score` and `eval.py grade` run clean on generated `predictions/*.csv`.
- [ ] `results.xlsx` includes all 7 sheets matching curves in `curves/`.
- [ ] `report.md` contains comprehensive failure analysis and robot deployment trade-offs.
- [ ] No heavy files (images, zip, checkpoints) staged in git repository.
