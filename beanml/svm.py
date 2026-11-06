from __future__ import annotations

import warnings

import numpy as np

from .base import Classifier
from .preprocessing import stratified_subsample

# kernel SVM. each binary problem is solved with SMO, picking the working pair with the
# second-order rule from Fan, Chen & Lin (2005) - same idea libsvm uses. multiclass = one-vs-one

TAU = 1e-12   # stand-in for a zero/negative curvature so we never divide by 0


def rbf_kernel(A, B, gamma):
    # K(a, b) = exp(-gamma * ||a - b||^2)  -> 1 for identical points, ~0 for far-apart ones
    sq = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def smo_binary(K, y, C, tol=1e-3, max_iter=100_000):
    # the SVM dual problem:
    #   minimise  f(a) = 0.5 * a' Q a - sum(a)
    #   subject to  0 <= a_i <= C   and   sum_i y_i a_i = 0,      with Q_ij = y_i y_j K_ij
    # SMO's trick: the equality constraint means we can't move one alpha alone, but we CAN move
    # two at a time and solve that tiny 2-variable problem in closed form. repeat until KKT holds.
    # returns (alpha, rho, iterations); decision value is f(x) = sum_t a_t y_t K(x_t, x) - rho
    n = len(y)
    y = y.astype(float)
    alpha = np.zeros(n)
    G = -np.ones(n)            # gradient of f at a = 0 is Q*0 - 1 = -1
    Kd = np.diag(K).astype(float)
    it = 0
    while it < max_iter:
        v = -y * G
        # I_up: alphas that can still move "up" in the y-direction; I_low: ones that can move "down"
        up = ((y > 0) & (alpha < C)) | ((y < 0) & (alpha > 0))
        low = ((y > 0) & (alpha > 0)) | ((y < 0) & (alpha < C))
        if not up.any() or not low.any():
            break
        # i = the most violating index on the "up" side
        v_up = np.where(up, v, -np.inf)
        i = int(np.argmax(v_up))
        g_max = v_up[i]
        v_low = np.where(low, v, np.inf)
        # optimality: max over I_up of -yG  <=  min over I_low of -yG  (up to tol). gap small -> done
        if g_max - v_low.min() < tol:
            break
        # pick j with the second-order rule: the j whose joint update with i would drop the
        # objective the most, which is roughly  -b^2 / a  with
        #   b = gradient gap between i and j,  a = K_ii + K_jj - 2 K_ij (curvature along that pair)
        Ki = K[i].astype(float)
        b = g_max - v
        a = Kd[i] + Kd - 2.0 * Ki
        a = np.where(a > 0, a, TAU)
        gain = np.where(low & (b > 0), -(b * b) / a, np.inf)
        j = int(np.argmin(gain))
        Kj = K[j].astype(float)

        ai_old, aj_old = alpha[i], alpha[j]
        yi, yj = y[i], y[j]
        quad = max(Kd[i] + Kd[j] - 2.0 * Ki[j], TAU)
        # two-variable update, then clip back into the [0, C] box while keeping
        # y_i a_i + y_j a_j constant (this is libsvm's update, case by case)
        if yi != yj:
            # opposite labels: a_i - a_j stays fixed, both move by the same delta
            delta = (-G[i] - G[j]) / quad
            diff = alpha[i] - alpha[j]
            alpha[i] += delta
            alpha[j] += delta
            if diff > 0:
                if alpha[j] < 0:
                    alpha[j], alpha[i] = 0.0, diff
            elif alpha[i] < 0:
                alpha[i], alpha[j] = 0.0, -diff
            if diff > 0:
                if alpha[i] > C:
                    alpha[i], alpha[j] = C, C - diff
            elif alpha[j] > C:
                alpha[j], alpha[i] = C, C + diff
        else:
            # same labels: a_i + a_j stays fixed, one goes up as the other goes down
            delta = (G[i] - G[j]) / quad
            s = alpha[i] + alpha[j]
            alpha[i] -= delta
            alpha[j] += delta
            if s > C:
                if alpha[i] > C:
                    alpha[i], alpha[j] = C, s - C
            elif alpha[j] < 0:
                alpha[j], alpha[i] = 0.0, s
            if s > C:
                if alpha[j] > C:
                    alpha[j], alpha[i] = C, s - C
            elif alpha[i] < 0:
                alpha[i], alpha[j] = 0.0, s
        dai, daj = alpha[i] - ai_old, alpha[j] - aj_old
        # only two alphas changed, so the gradient update is cheap:
        #   G += Q[:, i] * d_ai + Q[:, j] * d_aj,   and  Q[:, i] = y * y_i * K[:, i]
        G += y * (yi * dai * Ki + yj * daj * Kj)
        it += 1
    if it >= max_iter:
        warnings.warn(f"SMO hit max_iter={max_iter} before converging")

    # bias: for "free" support vectors (0 < a < C) the margin is exactly 1, which pins
    # rho = y_i * G_i. average over all of them to smooth out round-off
    yG = y * G
    at_ub, at_lb = alpha >= C, alpha <= 0
    free = ~at_ub & ~at_lb
    if free.any():
        rho = float(yG[free].mean())
    else:
        # no free SVs at all: rho can sit anywhere in a feasible interval, take its midpoint
        ub_mask = (at_ub & (y < 0)) | (at_lb & (y > 0))
        lb_mask = (at_ub & (y > 0)) | (at_lb & (y < 0))
        ub = yG[ub_mask].min() if ub_mask.any() else np.inf
        lb = yG[lb_mask].max() if lb_mask.any() else -np.inf
        rho = float((ub + lb) / 2) if np.isfinite(ub + lb) else 0.0
    return alpha, rho, it


class KernelSVM(Classifier):
    has_proba = False

    def __init__(self, C=1.0, gamma="scale", tol=1e-3, max_iter=100_000, max_pair_size=5000, seed=0):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.max_pair_size = max_pair_size
        self.seed = seed

    def fit(self, X, y):
        y = self._setup_classes(y)
        X = np.asarray(X, dtype=float)
        # gamma="scale" = 1 / (n_features * overall variance); on standardised data that's ~1/16
        self.gamma_ = 1.0 / (X.shape[1] * X.var()) if self.gamma == "scale" else float(self.gamma)
        self.pairs_, self.sv_, self.coef_, self.rho_, self.n_iter_ = [], [], [], [], []
        k = self.n_classes_
        # one-vs-one: a separate binary SVM for every pair of classes, K*(K-1)/2 = 21 of them here
        for a in range(k):
            for b in range(a + 1, k):
                idx = np.flatnonzero((y == a) | (y == b))
                # safety cap so the n x n kernel matrix can't blow up memory
                # (the biggest Dry Bean pair is ~4,900 rows, under the 5,000 cap, so nothing is dropped)
                if len(idx) > self.max_pair_size:
                    idx = idx[stratified_subsample(y[idx], self.max_pair_size, seed=self.seed + a * k + b)]
                Xp = X[idx]
                yp = np.where(y[idx] == a, 1.0, -1.0)   # class a = +1, class b = -1
                Kp = rbf_kernel(Xp, Xp, self.gamma_).astype(np.float32)   # float32 halves the memory
                alpha, rho, it = smo_binary(Kp, yp, self.C, self.tol, self.max_iter)
                # only rows with alpha > 0 (the support vectors) matter for predictions
                sv = alpha > 1e-8
                self.pairs_.append((a, b))
                self.sv_.append(Xp[sv])
                self.coef_.append(alpha[sv] * yp[sv])   # store a_i * y_i, that's all f(x) needs
                self.rho_.append(rho)
                self.n_iter_.append(it)
        return self

    @property
    def n_support_(self):
        return int(sum(len(s) for s in self.sv_))

    def pair_decisions(self, X, chunk=2048):
        # f_pair(x) = sum_i (a_i y_i) K(sv_i, x) - rho for every pair; > 0 means "first class"
        X = np.asarray(X, dtype=float)
        out = np.empty((len(X), len(self.pairs_)))
        for p, (sv, coef, rho) in enumerate(zip(self.sv_, self.coef_, self.rho_)):
            for s in range(0, len(X), chunk):
                out[s:s + chunk, p] = rbf_kernel(X[s:s + chunk], sv, self.gamma_) @ coef - rho
        return out

    def decision_function(self, X):
        # each pairwise SVM casts one vote; the class with the most votes wins.
        # we also add up the raw margins as "confidence", squashed by c / (3*(|c|+1)) so it's
        # always under 1/3 - it can break ties between votes but never overturn a vote lead.
        # these scores are what the AUC is computed on (AUC only needs a sensible ranking)
        F = self.pair_decisions(X)
        votes = np.zeros((len(F), self.n_classes_))
        conf = np.zeros_like(votes)
        for p, (a, b) in enumerate(self.pairs_):
            f = F[:, p]
            votes[:, a] += f > 0
            votes[:, b] += f <= 0
            conf[:, a] += f
            conf[:, b] -= f
        return votes + conf / (3.0 * (np.abs(conf) + 1.0))
