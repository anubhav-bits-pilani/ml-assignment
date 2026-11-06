from __future__ import annotations

import numpy as np

from .base import Classifier, softmax
from .preprocessing import stratified_split


# plain feed-forward net:  x -> [Linear -> ReLU] x len(hidden) -> Linear -> softmax
# trained on cross-entropy + L2 with Adam, keeping the weights from the epoch with the best
# validation loss (early stopping)
class MLP(Classifier):
    def __init__(self, hidden=(64, 32), lr=1e-3, l2=1e-4, epochs=300, batch_size=128,
                 val_fraction=0.1, patience=20, seed=0):
        self.hidden = tuple(hidden)
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.patience = patience
        self.seed = seed

    def _forward(self, X, params):
        # keep every layer's output - backprop needs them
        acts = [X]
        h = X
        for i, (W, b) in enumerate(params):
            z = h @ W + b
            # ReLU(z) = max(0, z) on hidden layers; the last layer stays linear (logits)
            h = np.maximum(z, 0.0) if i < len(params) - 1 else z
            acts.append(h)
        return acts  # last one = logits

    def _loss(self, X, y, params):
        # mean cross-entropy = -mean log P(true class)
        P = softmax(self._forward(X, params)[-1])
        return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1))))

    def fit(self, X, y):
        y = self._setup_classes(y)
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        # inner validation slice from the TRAINING data, used only for early stopping
        tr, va = stratified_split(y, self.val_fraction, seed=self.seed)
        Xt, yt, Xv, yv = X[tr], y[tr], X[va], y[va]
        sizes = [X.shape[1], *self.hidden, self.n_classes_]
        # He init: weights ~ N(0, 2 / fan_in), keeps the activations' scale steady through ReLU layers
        params = [(rng.normal(0, np.sqrt(2.0 / a), (a, b)), np.zeros(b)) for a, b in zip(sizes, sizes[1:])]
        # Adam's running first (m) and second (v) moments of the gradient, one pair per layer
        m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
        v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
        b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
        best, best_params, stall = np.inf, None, 0
        self.val_loss_ = []
        for epoch in range(self.epochs):
            order = rng.permutation(len(yt))
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                acts = self._forward(Xt[idx], params)
                # softmax + cross-entropy gradient wrt the logits is just (P - Y) / batch size
                delta = softmax(acts[-1])
                delta[np.arange(len(idx)), yt[idx]] -= 1.0
                delta /= len(idx)
                step += 1
                # backprop, last layer first
                for li in range(len(params) - 1, -1, -1):
                    W, b = params[li]
                    # dL/dW = (layer input)' * delta (+ L2 term),  dL/db = sum of delta over the batch
                    gW = acts[li].T @ delta + self.l2 * W
                    gb = delta.sum(axis=0)
                    if li > 0:
                        # push delta back one layer: through W, then through ReLU (gradient 1 if active, else 0).
                        # uses the OLD W on purpose - the update happens just below
                        delta = (delta @ W.T) * (acts[li] > 0)
                    # Adam:
                    #   m = b1*m + (1-b1)*g        (momentum-ish average of gradients)
                    #   v = b2*v + (1-b2)*g^2      (average of squared gradients)
                    #   w -= lr * m_hat / (sqrt(v_hat) + eps), with hats = bias-corrected by 1 - b^t
                    mW, mb = m[li]
                    vW, vb = v[li]
                    mW = b1 * mW + (1 - b1) * gW
                    mb = b1 * mb + (1 - b1) * gb
                    vW = b2 * vW + (1 - b2) * gW * gW
                    vb = b2 * vb + (1 - b2) * gb * gb
                    m[li], v[li] = (mW, mb), (vW, vb)
                    c1, c2 = 1 - b1 ** step, 1 - b2 ** step
                    W = W - self.lr * (mW / c1) / (np.sqrt(vW / c2) + eps)
                    b = b - self.lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
                    params[li] = (W, b)
            # early stopping: remember the best weights, quit after `patience` epochs with no gain
            loss = self._loss(Xv, yv, params)
            self.val_loss_.append(loss)
            if loss < best - 1e-5:
                best, stall = loss, 0
                best_params = [(W.copy(), b.copy()) for W, b in params]
            else:
                stall += 1
                if stall >= self.patience:
                    break
        self.params_ = best_params if best_params is not None else params
        self.n_iter_ = epoch + 1
        return self

    def predict_proba(self, X):
        return softmax(self._forward(np.asarray(X, dtype=float), self.params_)[-1])
