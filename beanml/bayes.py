from __future__ import annotations

import numpy as np

from .base import Classifier, softmax


# Gaussian naive Bayes:
#   P(class | x)  is proportional to  P(class) * prod_j N(x_j ; mu_kj, var_kj)
# the "naive" bit is that product - every feature is assumed independent given the class.
# everything is done in log space, so the product of 16 tiny numbers becomes a sum
class GaussianNB(Classifier):
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        y = self._setup_classes(y)
        X = np.asarray(X, dtype=float)
        k, d = self.n_classes_, X.shape[1]
        self.theta_ = np.zeros((k, d))   # per-class feature means
        self.var_ = np.zeros((k, d))     # per-class feature variances
        counts = np.bincount(y, minlength=k)
        for c in range(k):
            Xc = X[y == c]
            self.theta_[c] = Xc.mean(axis=0)
            self.var_[c] = Xc.var(axis=0)
        # add a slice of the largest overall variance to every variance, so a near-constant
        # feature inside one class doesn't produce a crazy-sharp spike (or a divide by ~0)
        self.epsilon_ = self.var_smoothing * X.var(axis=0).max()
        self.var_ += self.epsilon_
        self.class_log_prior_ = np.log(counts / counts.sum())   # log P(class) from class frequencies
        return self

    def joint_log_likelihood(self, X):
        # log P(class) + sum_j log N(x_j; mu, var), where
        #   log N(x; mu, var) = -0.5 * log(2*pi*var) - (x - mu)^2 / (2*var)
        X = np.asarray(X, dtype=float)
        out = np.empty((len(X), self.n_classes_))
        for c in range(self.n_classes_):
            ll = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[c]))
            ll = ll - 0.5 * np.sum((X - self.theta_[c]) ** 2 / self.var_[c], axis=1)
            out[:, c] = self.class_log_prior_[c] + ll
        return out

    def predict_proba(self, X):
        # softmax of the joint log-likelihoods = normalising P(class, x) into P(class | x)
        return softmax(self.joint_log_likelihood(X))
