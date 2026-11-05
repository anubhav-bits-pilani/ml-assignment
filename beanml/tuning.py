from __future__ import annotations

import itertools
import time

import numpy as np

from . import metrics as M
from .base import Pipeline
from .neighbors import KNN, votes_from_neighbours
from .preprocessing import StandardScaler, stratified_kfold


# {'a': [1, 2], 'b': [x, y]}  ->  every combination as a dict (cartesian product)
def expand_grid(grid):
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


# one CV fold -> (macro F1, accuracy)
def _fold_scores(y_true, y_pred, k):
    cm = M.confusion_matrix(y_true, y_pred, k)
    (_, _, f1), _ = M.macro_weighted_prf(cm)
    return f1, M.accuracy(y_true, y_pred)


# stratified k-fold CV over every combo in `grid`, on the TRAINING split only.
# ranked by mean macro-F1 - with imbalanced classes plain accuracy would just favour DERMASON
def grid_search(make_model, grid, X, y, scale=True, folds=5, seed=0, verbose=False):
    k = int(y.max()) + 1
    # same folds for every combo, so they're compared fairly
    splits = list(stratified_kfold(y, folds, seed))
    results = []
    for params in expand_grid(grid):
        t0 = time.time()
        f1s, accs = [], []
        for tr, va in splits:
            # new model + new scaler per fold, the scaler only ever sees that fold's training part
            pipe = Pipeline(make_model(**params), scale=scale).fit(X[tr], y[tr])
            f1, acc = _fold_scores(y[va], pipe.predict(X[va]), k)
            f1s.append(f1)
            accs.append(acc)
        results.append({"params": params, "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                        "acc_mean": float(np.mean(accs)), "secs": round(time.time() - t0, 2)})
        if verbose:
            print(f"    {params} -> macro-F1 {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    results.sort(key=lambda r: -r["f1_mean"])
    return results


# KNN special case: refitting KNN for every k would redo the same distance work over and over.
# instead, per fold, find the k_max nearest neighbours ONCE, then score every smaller k from the
# first k columns of that same sorted list (the k nearest are always a prefix of the k_max nearest)
def knn_search(grid, X, y, folds=5, seed=0, verbose=False):
    k_cls = int(y.max()) + 1
    ks = sorted(grid["k"])
    splits = list(stratified_kfold(y, folds, seed))
    scores = {}
    for metric in grid["metric"]:
        for tr, va in splits:
            sc = StandardScaler().fit(X[tr])
            knn = KNN(k=ks[-1], metric=metric).fit(sc.transform(X[tr]), y[tr])
            d, i = knn.kneighbors(sc.transform(X[va]), ks[-1])
            lab = y[tr][i]
            for w in grid["weights"]:
                for kk in ks:
                    pred = np.argmax(votes_from_neighbours(d, lab, kk, k_cls, w), axis=1)
                    scores.setdefault((kk, w, metric), []).append(_fold_scores(y[va], pred, k_cls))
    results = []
    for (kk, w, metric), vals in scores.items():
        f1s = [v[0] for v in vals]
        accs = [v[1] for v in vals]
        results.append({"params": {"k": kk, "weights": w, "metric": metric}, "f1_mean": float(np.mean(f1s)),
                        "f1_std": float(np.std(f1s)), "acc_mean": float(np.mean(accs))})
    results.sort(key=lambda r: -r["f1_mean"])
    if verbose:
        for r in results[:5]:
            print(f"    {r['params']} -> macro-F1 {r['f1_mean']:.4f} ± {r['f1_std']:.4f}")
    return results
