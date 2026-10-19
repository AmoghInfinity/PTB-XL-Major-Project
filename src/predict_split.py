"""Save sigmoid predictions of a checkpoint on the validation and test folds.

    python src/predict_split.py --checkpoint-name resnet1d_bce_s1.pt --out-dir results/seeds/s1 --tag resnet1d_bce

Writes <out-dir>/<tag>_val.npz and <out-dir>/<tag>_test.npz with y_true, y_prob, vocab.
The model type and the fold assignment are read from the checkpoint.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import PTBXLDataset
from models.resnet1d import ResNet1D
from models.ecg_mamba import ECGMamba

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ROOT / "checkpoints" / a.checkpoint_name, map_location=dev, weights_only=False)
    name = ck.get("model_name", "resnet1d")
    vf, tf = ck.get("val_fold", 9), ck.get("test_fold", 10)
    model = (ResNet1D if name == "resnet1d" else ECGMamba)(n_labels=len(ck["vocab"])).to(dev)
    model.load_state_dict(ck["model_state"]); model.eval()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        ds = PTBXLDataset(split=split, val_fold=vf, test_fold=tf)
        assert list(ds.vocab) == list(ck["vocab"]), "vocabulary mismatch"
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)
        ys, ps = [], []
        with torch.no_grad():
            for x, y in dl:
                ps.append(torch.sigmoid(model(x.to(dev))).cpu().numpy()); ys.append(y.numpy())
        np.savez_compressed(out / f"{a.tag}_{split}.npz", y_true=np.concatenate(ys), y_prob=np.concatenate(ps),
                            vocab=np.asarray(ds.vocab), ecg_id=ds.df.index.to_numpy(),
                            patient_id=ds.df["patient_id"].to_numpy(), epoch=np.asarray(ck.get("epoch", -1)),
                            seed=np.asarray(-1 if ck.get("seed") is None else ck["seed"]))
        print(f"{a.tag} {split}: {len(ds)} records -> {out / f'{a.tag}_{split}.npz'}")


if __name__ == "__main__":
    main()
