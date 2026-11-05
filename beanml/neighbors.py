from __future__ import annotations

import numpy as np

from .base import Classifier


def pairwise_dist(A, B, metric="euclidean"):
    if metric == "euclidean":
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b  -> whole distance matrix from one matrix product.
        # round-off can make tiny values slightly negative, hence the clip before sqrt
        sq = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
        return np.sqrt(np.maximum(sq, 0.0))
    if metric == "manhattan":
        # sum_j |a_j - b_j| via broadcasting (memory hungry, so callers use small chunks)
        return np.abs(A[:, None, :] - B[None, :, :]).sum(-1)
    raise ValueError(f"unknown metric {metric}")


def votes_from_neighbours(dist, labels, k, n_classes, weights="uniform"):
    # dist / labels are (n, >=k), already sorted nearest-first. returns vote shares (n, K)
    d, lab = dist[:, :k], labels[:, :k]
    if weights == "distance":
        # closer neighbours get more say: weight = 1 / distance
        exact = d <= 1e-12
        with np.errstate(divide="ignore"):
            w = 1.0 / d
        # 1/0 is infinite, so if the query sits exactly on a training point,
        # only the exact matches get to vote
        has_exact = exact.any(axis=1)
        w[has_exact] = exact[has_exact].astype(float)
    else:
        w = np.ones_like(d)
    V = np.zeros((len(d), n_classes))
    # scatter-add each neighbour's weight into its class column
    np.add.at(V, (np.repeat(np.arange(len(d)), k), lab.ravel()), w.ravel())
    # tie-break: the single nearest neighbour gets a microscopic bonus
    V[np.arange(len(d)), lab[:, 0]] += 1e-9
    return V / V.sum(axis=1, keepdims=True)


# KNN doesn't really "train" - it just remembers the (scaled) training set
# and looks up the k closest beans at prediction time
class KNN(Classifier):
    def __init__(self, k=5, weights="uniform", metric="euclidean", chunk=1024):
        self.k = k
        self.weights = weights
        self.metric = metric
        self.chunk = chunk

    def fit(self, X, y):
        self.y_ = self._setup_classes(y)
        self.X_ = np.asarray(X, dtype=float)
        return self

    def kneighbors(self, X, k):
        X = np.asarray(X, dtype=float)
        k = min(k, len(self.X_))
        # do the queries in chunks so we never hold a 2700 x 10800 matrix (or worse) at once
        chunk = self.chunk if self.metric == "euclidean" else 64
        all_d, all_i = [], []
        for s in range(0, len(X), chunk):
            D = pairwise_dist(X[s:s + chunk], self.X_, self.metric)
            # argpartition finds the k smallest in O(n) without fully sorting the row,
            # then we only sort those k
            part = np.argpartition(D, k - 1, axis=1)[:, :k]
            pd_ = np.take_along_axis(D, part, axis=1)
            o = np.argsort(pd_, axis=1, kind="mergesort")
            all_i.append(np.take_along_axis(part, o, axis=1))
            all_d.append(np.take_along_axis(pd_, o, axis=1))
        return np.vstack(all_d), np.vstack(all_i)

    def predict_proba(self, X):
        # "probability" here = share of the neighbour vote each class got
        d, i = self.kneighbors(X, self.k)
        return votes_from_neighbours(d, self.y_[i], self.k, self.n_classes_, self.weights)
