from __future__ import annotations

import numpy as np

from .base import Classifier, softmax
from .preprocessing import stratified_split
from .tree import DecisionTree, RegressionTree


# random forest = bagging + feature randomness.
# each tree sees a bootstrap sample (n rows drawn WITH replacement) and each split only looks at
# a random sqrt(d) features. trees end up different, so averaging them cancels a lot of variance
class RandomForest(Classifier):
    def __init__(self, n_estimators=100, max_features="sqrt", max_depth=None, min_samples_leaf=1,
                 criterion="gini", seed=0):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.seed = seed

    def fit(self, X, y):
        y = self._setup_classes(y)
        X = np.asarray(X, dtype=float)
        n, k = len(y), self.n_classes_
        rng = np.random.default_rng(self.seed)
        self.trees_ = []
        oob_sum = np.zeros((n, k))
        oob_cnt = np.zeros(n)
        for _ in range(self.n_estimators):
            boot = rng.integers(0, n, n)
            t = DecisionTree(criterion=self.criterion, max_depth=self.max_depth,
                             min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                             random_state=int(rng.integers(1 << 31)), compact=True)
            t.fit(X[boot], y[boot], n_classes=k)
            self.trees_.append(t)
            # a bootstrap leaves out ~1/e = 37% of rows; those "out-of-bag" rows are a free
            # validation set for this tree
            out = np.ones(n, dtype=bool)
            out[boot] = False
            oob_sum[out] += t.predict_proba(X[out])
            oob_cnt[out] += 1
        seen = oob_cnt > 0
        self.oob_score_ = float(np.mean(np.argmax(oob_sum[seen], 1) == y[seen]))
        self.feature_importances_ = np.mean([t.feature_importances_ for t in self.trees_], axis=0)
        return self

    def predict_proba(self, X):
        # forest probability = plain average of the trees' leaf distributions
        X = np.asarray(X, dtype=float)
        return np.mean([t.predict_proba(X) for t in self.trees_], axis=0)


# multiclass AdaBoost, SAMME version (Zhu et al. 2009). each round:
#   1. fit a small tree on the current sample weights
#   2. err = weighted share of rows it gets wrong
#   3. alpha = lr * (log((1-err)/err) + log(K-1))   <- the log(K-1) is what makes it multiclass;
#      a learner only has to beat random guessing (err < 1 - 1/K), not 50%
#   4. misclassified rows get their weight multiplied by exp(alpha), then renormalise
class AdaBoostSAMME(Classifier):
    def __init__(self, n_estimators=150, max_depth=2, learning_rate=0.5, seed=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed

    def fit(self, X, y):
        y = self._setup_classes(y)
        X = np.asarray(X, dtype=float)
        n, k = len(y), self.n_classes_
        w = np.full(n, 1.0 / n)   # start with every bean equally important
        self.estimators_, self.alphas_ = [], []
        for m in range(self.n_estimators):
            # depth-2 trees rather than stumps - a single split can't beat chance on 7 classes
            t = DecisionTree(max_depth=self.max_depth, random_state=self.seed + m)
            t.fit(X, y, sample_weight=w, n_classes=k)
            miss = t.predict(X) != y
            err = float(w[miss].sum() / w.sum())
            if err >= 1.0 - 1.0 / k:
                break   # no better than guessing -> this learner is useless, stop here
            err = max(err, 1e-10)   # avoid log(1/0) if a tree happens to be perfect
            alpha = self.learning_rate * (np.log((1 - err) / err) + np.log(k - 1))
            self.estimators_.append(t)
            self.alphas_.append(alpha)
            if err <= 1e-10:
                break   # perfect fit, nothing left to reweight
            w = w * np.exp(alpha * miss)
            w /= w.sum()
        if not self.estimators_:
            raise RuntimeError("AdaBoost couldn't find a single weak learner better than chance")
        return self

    def decision_function(self, X):
        # weighted vote: every tree adds its alpha to the class it predicts
        X = np.asarray(X, dtype=float)
        D = np.zeros((len(X), self.n_classes_))
        for t, a in zip(self.estimators_, self.alphas_):
            D[np.arange(len(X)), t.predict(X)] += a
        return D / np.sum(self.alphas_)

    def predict_proba(self, X):
        # each tree hands out alpha worth of vote, so the normalised votes already sum to 1
        return self.decision_function(X)


# multiclass gradient boosting with the softmax / cross-entropy loss (Friedman 2001, alg. 6)
#   F_k(x) = raw score for class k, P = softmax(F)
#   each round, for every class k:
#     residual  r = y_k - p_k            (negative gradient of the loss wrt F_k)
#     fit a regression tree to r, then set each leaf to a Newton step:
#       leaf = (K-1)/K * sum(r) / sum(|r| * (1-|r|))
#     F_k += learning_rate * tree_k(x)
class GradientBoosting(Classifier):
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=3, min_samples_leaf=5,
                 subsample=1.0, val_fraction=0.1, n_iter_no_change=15, seed=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.val_fraction = val_fraction
        self.n_iter_no_change = n_iter_no_change
        self.seed = seed

    def _raw(self, X, n_rounds=None):
        # start from the prior scores and add up every round's (shrunk) tree outputs
        X = np.asarray(X, dtype=float)
        F = np.tile(self.init_, (len(X), 1))
        for round_trees in self.trees_[:n_rounds]:
            for c, t in enumerate(round_trees):
                F[:, c] += self.learning_rate * t.predict(X)
        return F

    def fit(self, X, y):
        y = self._setup_classes(y)
        X = np.asarray(X, dtype=float)
        k = self.n_classes_
        rng = np.random.default_rng(self.seed)
        # hold out a slice of the TRAINING data to decide when to stop (never the test set)
        if self.val_fraction and self.val_fraction > 0:
            tr, va = stratified_split(y, self.val_fraction, seed=self.seed)
        else:
            tr, va = np.arange(len(y)), np.array([], dtype=int)
        Xt, yt = X[tr], y[tr]
        Y = np.zeros((len(yt), k))
        Y[np.arange(len(yt)), yt] = 1.0
        # initial scores = log class priors, so round 0 predicts the class frequencies
        prior = np.bincount(yt, minlength=k) / len(yt)
        self.init_ = np.log(np.clip(prior, 1e-12, None))
        F = np.tile(self.init_, (len(yt), 1))
        Fv = np.tile(self.init_, (len(va), 1))
        self.trees_, self.val_loss_ = [], []
        best, best_round, stall = np.inf, 0, 0
        scale = (k - 1) / k
        for m in range(self.n_estimators):
            P = softmax(F)
            rows = np.arange(len(yt))
            if self.subsample < 1.0:
                rows = rng.choice(len(yt), int(self.subsample * len(yt)), replace=False)
            round_trees = []
            for c in range(k):
                r = Y[:, c] - P[:, c]              # negative gradient
                h = np.abs(r) * (1 - np.abs(r))    # diagonal hessian approx, = p(1-p)
                t = RegressionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                   random_state=self.seed + m, compact=True)
                t.fit(Xt[rows], r[rows], hess=h[rows])
                t.value_ = (t.value_ * scale).astype(np.float32)   # the (K-1)/K factor from Friedman
                F[:, c] += self.learning_rate * t.predict(Xt)
                if len(va):
                    Fv[:, c] += self.learning_rate * t.predict(X[va])
                round_trees.append(t)
            self.trees_.append(round_trees)
            if len(va):
                # validation cross-entropy; stop after n_iter_no_change rounds without improvement
                Pv = softmax(Fv)
                loss = float(-np.mean(np.log(np.clip(Pv[np.arange(len(va)), y[va]], 1e-15, 1))))
                self.val_loss_.append(loss)
                if loss < best - 1e-6:
                    best, best_round, stall = loss, m + 1, 0
                else:
                    stall += 1
                    if stall >= self.n_iter_no_change:
                        break
        if len(va):
            self.trees_ = self.trees_[:best_round]   # roll back to the best round
        self.n_rounds_ = len(self.trees_)
        return self

    def predict_proba(self, X):
        return softmax(self._raw(X))


# averages predict_proba of already-trained pipelines. only the member NAMES get pickled -
# the app hands the loaded members back in via attach(), so nothing is stored twice
class SoftVoting:
    has_proba = True

    def __init__(self, member_names, class_names, feature_names, name="Soft Voting Ensemble"):
        self.member_names = list(member_names)
        self.class_names = list(class_names)
        self.feature_names = list(feature_names)
        self.name = name
        self._members = None

    def attach(self, pipelines):
        self._members = [pipelines[m] for m in self.member_names]
        return self

    def __getstate__(self):
        # drop the attached members when pickling, otherwise we'd save 5 models twice
        state = self.__dict__.copy()
        state["_members"] = None
        return state

    def scores(self, X):
        if self._members is None:
            raise RuntimeError("call attach() with the member models first")
        # P_ensemble = (P_1 + ... + P_m) / m
        return np.mean([m.scores(X) for m in self._members], axis=0)

    def predict(self, X):
        return np.argmax(self.scores(X), axis=1)

    def predict_labels(self, X):
        return np.asarray(self.class_names)[self.predict(X)]
