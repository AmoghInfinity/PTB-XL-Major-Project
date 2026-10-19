"""Patient-cluster bootstrap for the recalibration comparison (Sec. VI-C of the paper).

Run AFTER recalibration_study.py, from the project root:
    python src/recalibration_bootstrap.py --recal-dir results/paper_analysis --tiers-json tiers.json --metadata data/ptbxl/ptbxl_database.csv
Writes results/paper_analysis/recalibration_bootstrap.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recalibration_study

if __name__ == "__main__":
    recalibration_study.bootstrap_main(sys.argv[1:])
