# Calibrated, Selective ECG Diagnosis — PTB-XL (Local / VSCode)

Multi-label 12-lead ECG classification on PTB-XL (44 diagnostic statements) with a frequency-stratified
analysis of calibration and selective prediction. Developed locally on an RTX 5050 (8 GB, Blackwell / `sm_120`) GPU.

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

# PyTorch for Blackwell (cu128 minimum):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Everything else:
pip install -r requirements.txt
```

Verified environment for the reported experiments: **Python 3.11.8, PyTorch 2.11.0+cu128 (CUDA 12.8)**, Windows,
NVIDIA GeForce RTX 5050 Laptop GPU.

**Verify the GPU before doing anything else:**

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
```

You should see `(12, 0)` as the compute capability. If `torch.cuda.is_available()`
is `False`, update your NVIDIA driver first (Blackwell needs a recent one).

## 2. Download PTB-XL

```bash
python src/data/download_ptbxl.py
```

Downloads PTB-XL v1.0.3 (~1.7 GB) from PhysioNet into `data/ptbxl/`. No login required. Only the 100 Hz records are used.

## 3. Project layout

```
ptbxl_project/
├── data/ptbxl/                      # PTB-XL v1.0.3 (gitignored)
├── checkpoints/                     # model weights + per-epoch logs (*.log.csv)
├── configs/default.yaml             # reference hyperparameters (train.py uses CLI arguments)
├── results/                         # predictions (.npz) and analysis outputs (.json, figures)
│   ├── seeds/s1..s5/                # five-seed replication
│   └── folds/f1..f10/               # 10-fold rotation
├── src/
│   ├── data/
│   │   ├── download_ptbxl.py
│   │   ├── labels.py                # SCP-ECG parsing, frequency tiers
│   │   └── dataset.py               # Dataset; optional --val-fold/--test-fold
│   ├── models/
│   │   ├── resnet1d.py              # 1-D ResNet (980,460 parameters)
│   │   └── ecg_mamba.py             # Mamba-style selective SSM (165,804 parameters)
│   ├── train.py                     # training (BCE / weighted BCE / focal), seeds, epoch logs
│   ├── predict_split.py             # save val + test predictions of a checkpoint
│   ├── aggregate_seeds.py           # five-seed summary (mean ± SD, paired differences)
│   ├── aggregate_folds.py           # pooled 10-fold rotation analysis + patient bootstrap
│   ├── recalibration_study.py       # global / tier / label temperature, label Platt scaling
│   ├── recalibration_bootstrap.py   # patient-level CIs for the recalibration comparison
│   ├── retention_bootstrap.py       # patient-level CIs for selective-prediction retention
│   ├── platt_sensitivity.py         # Platt scaling hyper-parameter sensitivity
│   ├── paper_analysis.py            # per-label tables, selective curves
│   ├── make_figures.py              # paper figures (single test fold)
│   └── ... original evaluation scripts (see Section 6)
├── tiers.json                       # 44-label vocabulary and frequency tiers
├── run_seeds.ps1                    # five-seed replication driver
├── run_folds.ps1                    # 10-fold rotation driver
└── requirements.txt
```

## 4. Train

Baseline ResNet1D:

```bash
python src/train.py --epochs 30 --batch-size 128 --amp
```

ECG-Mamba:

```bash
python src/train.py --model ecg_mamba --epochs 30 --batch-size 128 --amp --checkpoint-name ecg_mamba_best.pt
```

Weighted BCE:

```bash
python src/train.py --epochs 30 --batch-size 256 --amp --loss weighted_bce --checkpoint-name best_model_weighted_bce.pt
```

Additional arguments:

| Argument | Meaning |
|---|---|
| `--seed N` | Seeds Python, NumPy, PyTorch and data-loader workers. Without it, behaviour matches the original unseeded runs. |
| `--deterministic` | Requests deterministic cuDNN/cuBLAS kernels (slower). |
| `--val-fold K --test-fold M` | Use other PTB-XL folds for validation/test (default 9 / 10). |

Every run writes `checkpoints/<name>.log.csv` (per-epoch loss and validation AUROC per tier) and stores the
epoch, seed, folds and all arguments inside the checkpoint.

## 5. Experiments

### 5.1 Official split (folds 1–8 / 9 / 10)

The archived checkpoints (`best_model.pt`, `ecg_mamba_best.pt`, `best_model_weighted_bce.pt`, `best_model_focal.pt`)
are evaluated on fold 10 with the scripts in Section 6.

### 5.2 Five-seed replication

```powershell
powershell -ExecutionPolicy Bypass -File run_seeds.ps1
```

Trains ResNet1D and ECG-Mamba with seeds 1–5, saves predictions to `results/seeds/s<k>/`, and writes
`results/seeds/aggregate.json`.

### 5.3 10-fold rotation

```powershell
powershell -ExecutionPolicy Bypass -File run_folds.ps1
```

Each PTB-XL fold is the test fold once (validation = next fold). Calibrators are fitted on each rotation's own
validation fold. The 10 test folds are pooled (all 21,799 records, 709 rare-tier positives instead of 70) and
analysed with a patient-cluster bootstrap. Output: `results/folds/fold_rotation.json`.

### 5.4 Post-hoc analyses (CPU only, from saved predictions)

```bash
python src/recalibration_study.py --results results --tiers-json tiers.json --out results/paper_analysis
python src/recalibration_bootstrap.py --recal-dir results/paper_analysis --tiers-json tiers.json --metadata data/ptbxl/ptbxl_database.csv
python src/retention_bootstrap.py results tiers.json data/ptbxl/ptbxl_database.csv results/paper_analysis/retention_bootstrap.json
python src/platt_sensitivity.py --results results --tiers-json tiers.json --out results/paper_analysis
```

## 6. Original evaluation scripts

| Script | Purpose |
|---|---|
| `src/evaluate.py` | Evaluate a checkpoint on the test fold (AUROC, F1 per tier) |
| `src/evaluate_weighted_bce.py` | Evaluate the weighted-BCE model |
| `src/evaluate_thresholds.py` | Select per-label thresholds on validation, evaluate on test |
| `src/evaluate_weighted_bce_thresholds.py` | Same for weighted BCE |
| `src/calibration.py` | Raw calibration (ECE, Brier, reliability diagrams) |
| `src/temperature_scaling.py`, `src/temperature_scaling_ecg_mamba.py` | Global temperature scaling |
| `src/compare_models.py` | Paired patient-cluster bootstrap, ResNet1D vs ECG-Mamba |
| `src/statistical_analysis.py`, `src/statistical_analysis_ecg_mamba.py` | Raw vs temperature-scaled bootstrap |
| `src/selective_prediction.py`, `src/selective_prediction_bootstrap.py` | Risk–coverage and AURC with CIs |
| `src/explainability.py`, `src/run_explainability_batch.py` | Integrated Gradients (qualitative) |
| `src/benchmark_efficiency.py` | Inference latency / throughput |
| `src/check_pipeline.py` | Data and preprocessing audit |

## 7. Evaluation methodology

- **Splits:** official stratified folds; 17,418 / 2,183 / 2,198 recordings (training / validation / test); 1,904 test patients. Folds are patient-disjoint.
- **Labels:** 44 diagnostic SCP statements with ≥ 10 occurrences. The label list and tiers (14 common, 15 medium, 15 rare) are identical whether computed on all folds or on the training folds only.
- **Fitting rules:** checkpoint selection, decision thresholds, temperatures and Platt parameters use the validation fold only; the test fold is never used for fitting.
- **Calibration metrics:** ECE (15 equal-width bins), ECE divided by prevalence, expected/observed positives (E/O), NLL, Brier score, all per label and averaged per tier.
- **Selective prediction:** recordings ranked by mean binary entropy; Hamming risk, AURC, and per-tier retention of positive findings.
- **Uncertainty:** paired patient-cluster bootstrap (1,000 replicates, seed 42); across-seed paired t-tests for training variability.
- Fixed-threshold (0.5) and validation-tuned-threshold results are reported separately.

## 8. Verified results (official test fold, archived checkpoints)

| Metric | ResNet1D (BCE) | Weighted BCE | ECG-Mamba |
|---|---:|---:|---:|
| Macro AUROC | 0.9235 | 0.9089 | 0.9173 |
| Macro F1, threshold 0.5 | 0.2150 | 0.2285 | 0.2459 |
| Macro F1, tuned thresholds | 0.2853 | 0.2856 | 0.3172 |
| Micro F1, tuned thresholds | **0.5753** | 0.5424 | 0.5853 |
| Rare-tier F1, tuned thresholds | 0.0758 | 0.0971 | 0.1351 |

**Correction:** an earlier version of this README reported a ResNet1D tuned micro-F1 of 0.4425. That value came from
`threshold_evaluation(_tiers_corrected).json`, produced before the threshold-fallback fix. In that run label 2AVB
(one validation positive) received threshold 0.00 and was predicted positive for every test record. The correct
value, 0.5753, is in `results/resnet1d_threshold_evaluation_corrected.json`.

### Five-seed replication (mean ± SD, seeds 1–5)

| Metric | ResNet1D | ECG-Mamba | ECG-Mamba better |
|---|---:|---:|---:|
| Macro AUROC | 0.9166 ± 0.0028 | 0.9027 ± 0.0065 | 0/5 (p = 0.006) |
| Macro AUPRC | 0.3136 ± 0.0085 | 0.2846 ± 0.0128 | 0/5 |
| Macro NLL after label Platt | 0.0592 ± 0.0006 | 0.0610 ± 0.0010 | 0/5 |

The archived single-run advantage of ECG-Mamba on proper scoring rules did **not** replicate across seeds.

## 9. Main findings (hold in all 10 seeded runs)

1. **Global temperature scaling over-predicts rare diagnoses.** It improves common-label calibration but raises the
   rare-tier expected/observed ratio from about 0.84–0.89 to about 1.6.
2. **Per-label Platt scaling** (shrinkage, tier-level fallback for labels with < 5 validation positives) gives the
   lowest NLL and common-tier ECE and keeps rare-tier E/O near 1.
3. **Entropy-based abstention defers rare findings.** At 50 % coverage only about 13–20 % of rare-tier positives
   remain in the accepted set, so accepted-set accuracy overstates reliability on rare findings.

## 10. Reproducibility and artifact management

- Keep checkpoints and raw prediction arrays separate from derived summaries.
- Select thresholds and calibrators on validation data only.
- Reuse the shared label vocabulary and tier mapping (`tiers.json` / `PTBXLDataset.tier_mask()`).
- Preserve original result files when correcting evaluation logic; save corrections under new names.
- Record checkpoint, seed, folds, threshold policy and software environment for every reported experiment
  (`train.py` now stores these automatically).

## 11. Project status

Completed: multi-label classification (ResNet1D, ECG-Mamba, weighted BCE, focal loss), frequency-tier evaluation,
calibration and recalibration study, selective prediction, patient-level bootstrap, five-seed replication,
Integrated Gradients, efficiency benchmark, and a manuscript draft (IEEE single-column).

In progress: 10-fold rotation (ResNet1D) to increase rare-tier test positives tenfold.