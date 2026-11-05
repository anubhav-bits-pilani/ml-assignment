from __future__ import annotations

import numpy as np

from .base import Classifier, softmax


def _class_stats(X, y, k):
    # per-class mean vectors and class priors P(k) = n_k / n
    means = np.vstack([X[y == c].mean(axis=0) for c in range(k)])
    priors = np.bincount(y, minlength=k) / len(y)
    return means, priors


# LDA: every class is a Gaussian with its OWN mean but one SHARED covariance S.
# the quadratic x' S^-1 x term is then the same for every class and cancels, leaving a
# linear score:
#   delta_k(x) = x' S^-1 mu_k - 0.5 * mu_k' S^-1 mu_k + log P(k)
class LDA(Classifier):
    def __init__(self, shrinkage=1e-3):
        self.shrinkage = shrinkage

    def fit(self, X, y):
        y = self._setup_classes(y)
        X = np.asarray(X, dtype=float)
        k, (n, d) = self.n_classes_, X.shape
        self.means_, self.priors_ = _class_stats(X, y, k)
        # pooled within-class covariance: centre each row on its OWN class mean,
        # then divide by n - k (unbiased, one mean estimated per class)
        centred = X - self.means_[y]
        S = centred.T @ centred / (n - k)
        # shrink a bit towards a scaled identity, S <- (1-a)*S + a*(trace(S)/d)*I,
        # because Area / ConvexArea / EquivDiameter are near copies and S is nearly singular
        S = (1 - self.shrinkage) * S + self.shrinkage * np.trace(S) / d * np.eye(d)
        self.covariance_ = S
        # solve S A = M' rather than forming S^-1 - same answer, numerically much safer
        A = np.linalg.solve(S, self.means_.T)
        self.coef_ = A
        self.intercept_ = -0.5 * np.sum(self.means_.T * A, axis=0) + np.log(self.priors_)
        return self

    def predict_proba(self, X):
        return softmax(np.asarray(X, dtype=float) @ self.coef_ + self.intercept_)


# QDA: every class gets its own covariance S_k too, so the boundaries become quadratic:
#   delta_k(x) = -0.5 * log|S_k| - 0.5 * (x - mu_k)' S_k^-1 (x - mu_k) + log P(k)
class QDA(Classifier):
    def __init__(self, reg_param=1e-2):
        self.reg_param = reg_param

    def fit(self, X, y):
        y = self._setup_classes(y)
        X = np.asarray(X, dtype=float)
        k, d = self.n_classes_, X.shape[1]
        self.means_, self.priors_ = _class_stats(X, y, k)
        self.chol_, self.logdet_ = [], np.zeros(k)
        r = self.reg_param
        for c in range(k):
            Xc = X[y == c] - self.means_[c]
            S = Xc.T @ Xc / max(len(Xc) - 1, 1)
            # blend with the identity (data is standardised, so I is a sensible target).
            # without this, BOMBAY's ~420-row covariance of 16 collinear features is basically singular
            S = (1 - r) * S + r * np.eye(d)
            # Cholesky: S = L L'. Gives us both pieces cheaply:
            #   log|S| = 2 * sum(log(diag(L)))
            #   (x-mu)' S^-1 (x-mu) = ||L^-1 (x-mu)||^2
            L = np.linalg.cholesky(S)
            self.chol_.append(L)
            self.logdet_[c] = 2.0 * np.sum(np.log(np.diag(L)))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty((len(X), self.n_classes_))
        for c in range(self.n_classes_):
            # z = L^-1 (x - mu), so sum(z^2) is the squared Mahalanobis distance to class c
            z = np.linalg.solve(self.chol_[c], (X - self.means_[c]).T)
            out[:, c] = -0.5 * self.logdet_[c] - 0.5 * np.sum(z * z, axis=0) + np.log(self.priors_[c])
        return softmax(out)
