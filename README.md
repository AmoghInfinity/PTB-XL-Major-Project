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

## 5. What's next (not yet in this skeleton)

This covers data → model → training. The calibration, selective-prediction,
and explainability components from the project blueprint come next, as
separate scripts once baseline training is working:
- `src/calibrate.py` — temperature scaling, ECE/Brier per frequency tier
- `src/selective_predict.py` — risk-coverage curves, reject head
- `src/explain.py` — Integrated Gradients (via `captum`)
