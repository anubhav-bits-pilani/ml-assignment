from __future__ import annotations

import numpy as np

from .base import Classifier, softmax


def _onehot(y, k):
    # y = [2, 0] with k = 3  ->  [[0,0,1], [1,0,0]]
    Y = np.zeros((len(y), k))
    Y[np.arange(len(y)), y] = 1.0
    return Y


# multinomial logistic regression a.k.a. softmax regression
#   scores  z = XW + b            (one column per class)
#   probs   P = softmax(z)
#   loss    L = -(1/n) * sum_i log P[i, y_i]  +  (l2/2) * ||W||^2
# the bias isn't penalised - shrinking it would just shift every class for no reason
class SoftmaxRegression(Classifier):
    def __init__(self, lr=0.1, l2=1e-4, epochs=300, batch_size=256, tol=1e-6, patience=5,
                 momentum=0.9, seed=0):
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.tol = tol
        self.patience = patience
        self.momentum = momentum
        self.seed = seed

    def loss_and_grad(self, X, Y, W, b):
        P = softmax(X @ W + b)
        n = len(X)
        # cross-entropy: only the log-prob of the true class survives the Y * log(P) product.
        # clip so log never sees an exact 0
        loss = -np.sum(Y * np.log(np.clip(P, 1e-15, 1.0))) / n + 0.5 * self.l2 * np.sum(W * W)
        # the nice thing about softmax + cross-entropy: dL/dz = (P - Y) / n
        # chain rule through z = XW + b then gives
        #   dL/dW = X^T (P - Y) / n + l2 * W      and      dL/db = column sums of (P - Y) / n
        G = (P - Y) / n
        return loss, X.T @ G + self.l2 * W, G.sum(axis=0)

    def fit(self, X, y):
        y = self._setup_classes(y)
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        k = self.n_classes_
        Y = _onehot(y, k)
        rng = np.random.default_rng(self.seed)
        W = np.zeros((d, k))   # starting at zero is fine here, the loss is convex
        b = np.zeros(k)
        vW, vb = np.zeros_like(W), np.zeros_like(b)
        prev, stall = np.inf, 0
        self.loss_history_ = []
        for epoch in range(self.epochs):
            order = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]
                _, gW, gb = self.loss_and_grad(X[idx], Y[idx], W, b)
                # momentum: v = mu*v - lr*grad ; w = w + v
                # keeps a running "velocity" so we don't zig-zag across narrow valleys
                vW = self.momentum * vW - self.lr * gW
                vb = self.momentum * vb - self.lr * gb
                W += vW
                b += vb
            loss = self.loss_and_grad(X, Y, W, b)[0]
            self.loss_history_.append(loss)
            # stop once the full-data loss hasn't improved by more than tol for `patience` epochs
            stall = stall + 1 if prev - loss < self.tol else 0
            prev = min(prev, loss)
            if stall >= self.patience:
                break
        self.coef_, self.intercept_ = W, b
        self.n_iter_ = epoch + 1
        return self

    def predict_proba(self, X):
        return softmax(np.asarray(X, dtype=float) @ self.coef_ + self.intercept_)


# one-vs-rest linear SVM trained directly in the primal:
#   minimise  (lam/2)*||w_k||^2 + (1/n) * sum_i max(0, 1 - t_ik * (w_k.x_i + b_k))
# with t_ik = +1 if bean i is class k, else -1. lam = 1/(C*n) so C means roughly the same as
# in the usual dual form (bigger C = less regularisation)
class LinearSVM(Classifier):
    has_proba = False

    def __init__(self, C=1.0, epochs=60, batch_size=128, lr=0.05, seed=0):
        self.C = C
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

    def fit(self, X, y):
        y = self._setup_classes(y)
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        k = self.n_classes_
        T = np.where(_onehot(y, k) > 0, 1.0, -1.0)   # +1 for own class, -1 for the rest
        lam = 1.0 / (self.C * n)
        rng = np.random.default_rng(self.seed)
        W = np.zeros((d, k))   # all K binary SVMs trained side by side, one column each
        b = np.zeros(k)
        t = 0
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]
                t += 1
                # step size decays like 1/t (standard for SGD on hinge loss), but starts at lr
                eta = self.lr / (1.0 + self.lr * lam * t)
                Xb, Tb = X[idx], T[idx]
                # hinge is only "active" for points inside the margin: t * f(x) < 1
                viol = (Tb * (Xb @ W + b)) < 1.0
                # subgradient of max(0, 1 - t*f) wrt f is -t when violated, 0 otherwise
                G = -(Tb * viol)
                gW = Xb.T @ G / len(idx) + lam * W
                gb = G.mean(axis=0)
                W -= eta * gW
                b -= eta * gb
        self.coef_, self.intercept_ = W, b
        return self

    def decision_function(self, X):
        # raw margins w_k.x + b_k - biggest one wins
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


# multiclass perceptron: predict argmax_k w_k.x ; when wrong, push the true class's weights
# towards x and the wrongly-guessed class's weights away from x.
# the plain version bounces around forever on non-separable data, so we return the AVERAGE of
# the weights over every step instead (much more stable)
class AveragedPerceptron(Classifier):
    has_proba = False

    def __init__(self, epochs=15, seed=0):
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        y = self._setup_classes(y)
        X = np.hstack([np.asarray(X, dtype=float), np.ones((len(X), 1))])   # bias as an extra "1" column
        n, d = X.shape
        k = self.n_classes_
        W = np.zeros((k, d))
        # averaging trick: instead of summing W after every step (slow), keep U = sum of c * update.
        # then average = W - U / c  (works out the same, see Daume's "A Course in ML", ch. 4)
        U = np.zeros((k, d))
        c = 1
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            for i in rng.permutation(n):
                xi, yi = X[i], y[i]
                guess = int(np.argmax(W @ xi))
                if guess != yi:
                    W[yi] += xi
                    W[guess] -= xi
                    U[yi] += c * xi
                    U[guess] -= c * xi
                c += 1
        Wavg = W - U / c
        self.coef_, self.intercept_ = Wavg[:, :-1].T, Wavg[:, -1]   # split the bias column back off
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
