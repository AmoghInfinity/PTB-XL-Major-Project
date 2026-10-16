# Calibrated, Selective ECG Diagnosis — PTB-XL (Local / VSCode)

Local project setup for an RTX 5050 (8GB, Blackwell / `sm_120`) GPU.

## 0. Why this needs a specific install command

The RTX 5050 is a Blackwell-generation card (CUDA compute capability `sm_120`).
PyTorch builds older than the CUDA 12.8 toolkit do not ship `sm_120` kernels —
if you `pip install torch` normally, PyTorch will still *import* and
`torch.cuda.is_available()` may even return `True`, but you'll either silently
fall back to slow generic kernels or hit:

```
UserWarning: NVIDIA GeForce RTX 5050 with CUDA capability sm_120 is not
compatible with the current PyTorch installation.
```

Fix: install from the `cu128` (or newer) index, not the default PyPI wheel.

## 1. Environment setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install PyTorch for Blackwell (cu128 minimum — check
# https://pytorch.org/get-started/locally/ for the current recommended
# index; cu128 or cu129 both work as of late 2026):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Everything else:
pip install -r requirements.txt
```

**Verify the GPU is actually recognized correctly before doing anything else:**

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
```

You should see `(12, 0)` as the compute capability. If `torch.cuda.is_available()`
is `False`, your driver or CUDA toolkit is the problem, not PyTorch — update
your NVIDIA driver first (Blackwell needs a recent one).

## 2. Download PTB-XL

```bash
python src/data/download_ptbxl.py
```

Downloads the 100Hz release (~1.7GB) directly from PhysioNet into `data/ptbxl/`.
No login or credentialing required.

## 3. Train

```bash
python src/train.py --epochs 30 --batch-size 128 --amp
```

- `--batch-size 128` is a safe starting point for 8GB VRAM on a 1D CNN this
  size; bump it up if `nvidia-smi` shows headroom, or lower it if you hit an
  out-of-memory error.
- `--amp` enables automatic mixed precision (bf16 on Blackwell), which both
  speeds up training and reduces VRAM pressure — worth leaving on by default
  on this card.

## 4. Project layout

```
ptbxl_project/
├── data/ptbxl/              # downloaded dataset (gitignored)
├── checkpoints/             # saved model weights
├── configs/default.yaml     # hyperparameters
├── src/
│   ├── data/
│   │   ├── download_ptbxl.py
│   │   ├── labels.py        # SCP-ECG parsing, frequency-tier assignment
│   │   └── dataset.py        # PyTorch Dataset / DataLoader
│   ├── models/
│   │   └── resnet1d.py      # 1D ResNet multi-label classifier
│   └── train.py              # training loop, AMP, checkpointing
└── requirements.txt
```

## 5. Evaluation and Research Workflow

The project supports multi-label ECG classification on PTB-XL, with evaluation focused on predictive performance, frequency-tier performance, probability calibration, and uncertainty-based selective prediction.

### Models

- **ResNet1D:** Baseline 1D residual neural network for multi-label ECG classification.
- **ECG-Mamba:** Alternative sequence-model architecture evaluated against ResNet1D.
- **Weighted BCE:** Imbalance-aware training experiment for improving performance on less frequent diagnostic labels.

### Training

Baseline ResNet1D:

```bash
python src/train.py --epochs 30 --batch-size 128 --amp
```

Weighted BCE experiment:

```bash
python src/train.py --epochs 30 --batch-size 256 --amp --loss weighted_bce --checkpoint-name best_model_weighted_bce.pt
```

Use the checkpoint and configuration corresponding to each experiment when reproducing its results.

### Evaluation scripts

| Script | Purpose |
|---|---|
| `src/evaluate.py` | Evaluate the baseline model |
| `src/evaluate_weighted_bce.py` | Evaluate the weighted-BCE model |
| `src/evaluate_thresholds.py` | Select per-label thresholds on validation predictions and evaluate on test predictions |
| `src/evaluate_weighted_bce_thresholds.py` | Perform validation-selected threshold evaluation for weighted BCE |
| `src/calibration.py` | Evaluate probability calibration |
| `src/temperature_scaling.py` | Apply temperature scaling and assess calibration |
| `src/compare_models.py` | Compare models using paired statistical analysis |
| `src/statistical_analysis.py` | Statistical analysis of the ResNet1D model |
| `src/statistical_analysis_ecg_mamba.py` | Statistical analysis of ECG-Mamba |
| `src/selective_prediction.py` | Evaluate uncertainty-based selective prediction and risk-coverage behavior |
| `src/selective_prediction_bootstrap.py` | Estimate paired patient-cluster bootstrap uncertainty for selective-prediction comparisons |
| `src/check_pipeline.py` | Check dataset and preprocessing assumptions |

Run each script with the arguments and input paths supported by its implementation.

## 6. Evaluation Methodology

### Dataset splits

The current project uses the PTB-XL metadata folds as follows:

- Training: folds 1–8
- Validation: fold 9
- Test: fold 10

The current local metadata contains 21,799 recordings. The corresponding split sizes are 17,418 training, 2,183 validation, and 2,198 test recordings.

### Multi-label evaluation

The classifier predicts 44 diagnostic labels. Evaluation includes AUROC, AUPRC, F1-score, and calibration metrics. Fixed-threshold results and validation-selected-threshold results are separate evaluation settings and should not be mixed in tables or conclusions.

### Frequency-tier evaluation

Labels are grouped into common, medium, and rare tiers using the shared frequency-tier mapping implemented in `src/data/labels.py` and exposed through `PTBXLDataset.tier_mask()`.

For the current 44-label vocabulary, the authoritative tier sizes are:

| Tier | Number of labels |
|---|---:|
| Common | 14 |
| Medium | 15 |
| Rare | 15 |
| **Total** | **44** |

Evaluation scripts should reuse the dataset's shared mapping rather than independently reconstructing tiers from validation frequencies. The threshold-evaluation scripts have been updated to follow this rule.

### Threshold selection

Per-label thresholds are selected using validation data only. The selected thresholds are then applied to the test predictions.

Threshold-optimized F1 results should be reported separately from F1 results obtained using a fixed threshold of 0.5. Thresholds and their validation scores should be retained for reproducibility.

### Calibration and selective prediction

Calibration is assessed using metrics including Expected Calibration Error (ECE) and Brier score. Temperature scaling is evaluated separately from raw model probabilities.

Selective prediction ranks recordings by predictive uncertainty and evaluates classification risk at different coverage levels. The project also uses paired patient-cluster bootstrap analysis to quantify uncertainty in model differences.

## 7. Verified Threshold-Selected Results

The following results use validation-selected per-label thresholds and the corrected shared frequency-tier mapping.

| Metric | ResNet1D baseline | Weighted BCE |
|---|---:|---:|
| Macro F1 | 0.2853 | 0.2856 |
| Micro F1 | 0.4425 | 0.5424 |
| Common-tier F1 | 0.5632 | 0.5364 |
| Medium-tier F1 | 0.2353 | 0.2400 |
| Rare-tier F1 | 0.0759 | 0.0971 |
| Macro AUROC | 0.9235 | Not stored in this threshold-evaluation result |

These figures describe threshold-selected performance. They should not be interpreted as fixed-threshold results or as proof that one model is universally superior.

The weighted-BCE threshold-evaluation output does not currently store macro AUROC. Use the corresponding fixed-threshold evaluation artifact for that metric, and clearly identify the evaluation setting when reporting it.

### Corrected result artifacts

The following files contain the corrected tier summaries:

- `results/threshold_evaluation_tiers_corrected.json`
- `results/weighted_bce_threshold_evaluation_tiers_corrected.json`

The original files remain available for historical comparison:

- `results/threshold_evaluation.json`
- `results/weighted_bce_threshold_evaluation.json`

The overall macro and micro F1 values are unchanged by the tier correction. The tier-specific values differ because the label grouping has been made consistent.

## 8. Reproducibility and Artifact Management

- Keep trained checkpoints and raw prediction arrays separate from derived evaluation summaries.
- Select thresholds on validation data only; do not use test performance to tune them.
- Reuse the same label vocabulary and frequency-tier mapping across model evaluations.
- Preserve original result files when correcting evaluation logic. Save corrected outputs under distinct filenames until they have been verified.
- Record the model checkpoint, threshold-selection procedure, split, evaluation metrics, and software environment for every reported experiment.
- Do not compare results from different threshold policies as if they were produced under identical settings.

## 9. Project Status

The project has progressed beyond the initial training skeleton. Its current workflow includes multi-label ECG classification, ResNet1D and ECG-Mamba comparisons, weighted-BCE experiments, calibration, temperature scaling, statistical analysis, and uncertainty-based selective prediction.

Remaining research work includes consolidating the final experimental tables, ensuring all reported metrics use consistent evaluation definitions, reviewing related literature, and documenting the limitations of the current dataset and evaluation setup.