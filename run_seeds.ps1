# Multi-seed experiment driver (Windows PowerShell). Run from the project root with the venv active:
#   powershell -ExecutionPolicy Bypass -File run_seeds.ps1
#
# Priority 1 = ResNet1D-BCE and ECG-Mamba-BCE, seeds 1..5 (needed for every model-comparison claim).
# Priority 2 = focal and weighted-BCE ResNet1D, seeds 1..3 (loss-function comparison) - uncomment below.
# Existing checkpoints/results are NOT overwritten: new runs are named like checkpoints/resnet1d_bce_s3.pt
# and their predictions go to results/seeds/s3/. Re-running the script skips checkpoints that already exist.

$ErrorActionPreference = "Stop"
$Seeds  = 1..5
$Epochs = 30

# Run python and stop the whole script if it fails (PowerShell does not do this by itself).
function Invoke-Py {
    & python @args
    if ($LASTEXITCODE -ne 0) { throw "python $($args -join ' ') failed with exit code $LASTEXITCODE" }
}

New-Item -ItemType Directory -Force -Path "results/seeds" | Out-Null
Invoke-Py -c "import torch, sys; print(torch.__version__, torch.version.cuda, sys.version)" | Out-File -Encoding utf8 "results/seeds/torch_version.txt"
& python -m pip freeze | Out-File -Encoding utf8 "results/seeds/pip_freeze.txt"

foreach ($s in $Seeds) {
    foreach ($cfg in @(@{model="resnet1d"; loss="bce"; bs=128}, @{model="ecg_mamba"; loss="bce"; bs=128})) {
        $tag = "$($cfg.model)_$($cfg.loss)_s$s"
        if (-not (Test-Path "checkpoints/$tag.pt")) {
            Write-Host "=== Training $tag ==="
            Invoke-Py src/train.py --model $cfg.model --loss $cfg.loss --epochs $Epochs --batch-size $cfg.bs --seed $s --amp --checkpoint-name "$tag.pt"
        }
        Invoke-Py src/predict_split.py --checkpoint-name "$tag.pt" --out-dir "results/seeds/s$s" --tag "$($cfg.model)_$($cfg.loss)"
    }
}

# ---- Priority 2 (remove the leading '#' on the lines below when Priority 1 is done) ----
# foreach ($s in 1..3) {
#     foreach ($cfg in @(@{loss="focal"; bs=128}, @{loss="weighted_bce"; bs=256})) {
#         $tag = "resnet1d_$($cfg.loss)_s$s"
#         if (-not (Test-Path "checkpoints/$tag.pt")) {
#             Invoke-Py src/train.py --model resnet1d --loss $cfg.loss --epochs $Epochs --batch-size $cfg.bs --seed $s --amp --checkpoint-name "$tag.pt"
#         }
#         Invoke-Py src/predict_split.py --checkpoint-name "$tag.pt" --out-dir "results/seeds/s$s" --tag "resnet1d_$($cfg.loss)"
#     }
# }

Invoke-Py src/aggregate_seeds.py --seeds-dir results/seeds --tiers-json tiers.json --out results/seeds/aggregate.json
