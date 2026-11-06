# train + evaluate every model and write everything into model/
#   python train.py          full run with the CV grid search (~2-3 min)
#   python train.py --quick  skip CV, reuse the best params from model/cv_results.json

import argparse

from beanml.experiment import SEED, run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the CV grid search")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    run(quick=args.quick, seed=args.seed)
