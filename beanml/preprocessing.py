from __future__ import annotations

import numpy as np
import pandas as pd


class StandardScaler:
    # z = (x - mean) / std, column by column
    # puts Area (~50,000) and ShapeFactor2 (~0.002) on the same footing, which matters a lot
    # for anything distance or gradient based (KNN, SVM, LR, MLP)
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # constant column -> leave it alone instead of dividing by zero
        std[std == 0] = 1.0
        self.scale_ = std
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LabelEncoder:
    # bean names <-> ints 0..K-1, sorted alphabetically so the mapping is always the same
    def fit(self, y):
        self.classes_ = np.array(sorted(set(np.asarray(y).astype(str).tolist())))
        return self

    def transform(self, y):
        y = np.asarray(y).astype(str)
        lookup = {c: i for i, c in enumerate(self.classes_)}
        # better to shout about a label we've never seen than silently map it to something
        unknown = sorted(set(y.tolist()) - set(lookup))
        if unknown:
            raise ValueError(f"labels never seen during training: {unknown}")
        return np.array([lookup[v] for v in y], dtype=int)

    def inverse_transform(self, idx):
        return self.classes_[np.asarray(idx, dtype=int)]


def dedupe(df: pd.DataFrame, feature_cols, target_col):
    # step 1: exact duplicate rows go (otherwise the same bean can end up in train AND test)
    n0 = len(df)
    df = df.drop_duplicates()
    n_exact = n0 - len(df)
    # step 2: if the same feature vector shows up with two different labels we can't trust
    # either copy, so all of them get dropped (Dry Bean happens to have none of these)
    n_labels = df.groupby(list(feature_cols))[target_col].transform("nunique")
    conflict = n_labels > 1
    df = df[~conflict].reset_index(drop=True)
    report = {"rows_raw": n0, "exact_duplicates_dropped": int(n_exact),
              "conflicting_rows_dropped": int(conflict.sum()), "rows_clean": len(df)}
    return df, report


def stratified_split(y, test_size=0.2, seed=0):
    # split each class on its own, so every class keeps (roughly) the same share in train and
    # test - important here since BOMBAY is only ~4% of the data
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    # shuffle again so rows aren't grouped by class (mini-batch training would hate that)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def stratified_kfold(y, k=5, seed=0):
    # deal every class out round-robin into k folds (like dealing cards), so per-class fold
    # sizes differ by at most 1. yields (train_idx, val_idx) for each fold
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % k
    for f in range(k):
        yield np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)


def stratified_subsample(y, n, seed=0):
    # keep ~n rows with the same class proportions - only used as a safety cap for the kernel SVM
    y = np.asarray(y)
    if n >= len(y):
        return np.arange(len(y))
    keep, _ = stratified_split(y, test_size=1 - n / len(y), seed=seed)
    return np.sort(keep)
