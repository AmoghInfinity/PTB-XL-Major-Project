# 10-fold rotation: each PTB-XL fold is the test fold once; validation = the next fold (10 -> 1).
# Run from the project root with the venv active:
#   powershell -ExecutionPolicy Bypass -File run_folds.ps1
#
# Trains ResNet1D on all 10 rotations (~21 min each), then pools and analyses them (~20-30 min).
# Re-running skips checkpoints that already exist, so it is safe to stop and restart.
# Existing checkpoints/results are never overwritten (names like checkpoints/resnet1d_bce_fold3.pt,
# predictions in results/folds/f3/).

$ErrorActionPreference = "Stop"
$Epochs = 30
$Seed   = 1

function Invoke-Py {
    & python @args
    if ($LASTEXITCODE -ne 0) { throw "python $($args -join ' ') failed with exit code $LASTEXITCODE" }
}

New-Item -ItemType Directory -Force -Path "results/folds" | Out-Null

# Deadline mode: ResNet1D only (~3.5-4 h). To add ECG-Mamba later (~7 h), use @("resnet1d", "ecg_mamba").
foreach ($model in @("resnet1d")) {
    foreach ($k in 1..10) {
        $val = ($k % 10) + 1
        $tag = "${model}_bce_fold$k"
        if (-not (Test-Path "checkpoints/$tag.pt")) {
            Write-Host "=== Training $tag  (test fold $k, validation fold $val) ==="
            Invoke-Py src/train.py --model $model --loss bce --epochs $Epochs --batch-size 128 --seed $Seed --amp `
                --test-fold $k --val-fold $val --checkpoint-name "$tag.pt"
        }
        if (-not (Test-Path "results/folds/f$k/${model}_bce_test.npz")) {
            Invoke-Py src/predict_split.py --checkpoint-name "$tag.pt" --out-dir "results/folds/f$k" --tag "${model}_bce"
        }
    }
}

Invoke-Py src/aggregate_folds.py --folds-dir results/folds --tiers-json tiers.json --out results/folds/fold_rotation.json
Write-Host "results/folds/pooled_resnet1d_bce.npz and checkpoints/*_fold*.log.csv"