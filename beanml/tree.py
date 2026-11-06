from __future__ import annotations

import numpy as np

from .base import Classifier

# CART trees. Built with an explicit stack (no recursion limits to worry about) and stored as
# flat arrays: feature_[i], threshold_[i], left_[i], right_[i], value_[i] for node i.
# a leaf is any node with feature_ == -1


def _n_candidate_features(max_features, d):
    # how many features each split is allowed to look at (random forests use sqrt(d))
    if max_features is None:
        return d
    if max_features == "sqrt":
        return max(1, int(np.sqrt(d)))
    if max_features == "log2":
        return max(1, int(np.log2(d)))
    if isinstance(max_features, float):
        return max(1, int(max_features * d))
    return max(1, min(d, int(max_features)))


class _TreeBase:
    def _init_storage(self):
        self._feat, self._thr, self._left, self._right, self._vals = [], [], [], [], []

    def _new_node(self, value):
        # every node starts life as a leaf; it gets turned into a split later if worth it
        self._feat.append(-1)
        self._thr.append(0.0)
        self._left.append(-1)
        self._right.append(-1)
        self._vals.append(value)
        return len(self._feat) - 1

    def _grow(self, X, n_features_total):
        rng = np.random.default_rng(self.random_state)
        n_try = _n_candidate_features(self.max_features, n_features_total)
        self._init_storage()
        self.importances_raw_ = np.zeros(n_features_total)
        root_idx = np.arange(len(X))
        root = self._new_node(self._leaf_value(root_idx))
        stack = [(root, root_idx, 0)]   # (node id, rows that reach it, depth)
        while stack:
            node, idx, depth = stack.pop()
            # stopping rules: too deep, too few rows, or already pure -> stays a leaf
            if (self.max_depth is not None and depth >= self.max_depth) \
                    or len(idx) < self.min_samples_split or self._is_pure(idx):
                continue
            feats = rng.permutation(n_features_total)[:n_try] if n_try < n_features_total \
                else np.arange(n_features_total)
            best = None   # (impurity of the two children, feature, threshold, go-left mask)
            for f in feats:
                cand = self._best_split_on_feature(X[idx, f], idx)
                if cand is not None and (best is None or cand[0] < best[0]):
                    best = (cand[0], f, cand[1], cand[2])
            if best is None:
                continue
            # gain = impurity before split - impurity after (both already weighted by node size)
            gain = self._node_impurity(idx) - best[0]
            if gain <= self.min_impurity_decrease:
                continue
            _, f, thr, go_left = best
            # feature importance = total impurity drop this feature was responsible for
            self.importances_raw_[f] += gain
            li, ri = idx[go_left], idx[~go_left]
            l = self._new_node(self._leaf_value(li))
            r = self._new_node(self._leaf_value(ri))
            self._feat[node], self._thr[node] = int(f), float(thr)
            self._left[node], self._right[node] = l, r
            stack.append((r, ri, depth + 1))
            stack.append((l, li, depth + 1))
        self.feature_ = np.array(self._feat, dtype=np.int32)
        self.threshold_ = np.array(self._thr, dtype=np.float64)   # keep thresholds exact
        self.left_ = np.array(self._left, dtype=np.int32)
        self.right_ = np.array(self._right, dtype=np.int32)
        # forests have hundreds of trees, float32 leaf values halve the pickle size
        self.value_ = np.array(self._vals, dtype=np.float32 if self.compact else np.float64)
        del self._feat, self._thr, self._left, self._right, self._vals
        s = self.importances_raw_.sum()
        self.feature_importances_ = self.importances_raw_ / s if s > 0 else self.importances_raw_

    def _valid_cut_mask(self, xs_sorted):
        # "cut after position i" means left = first i+1 sorted rows. A cut is only allowed
        # between two DIFFERENT values (otherwise equal values would land on both sides, and
        # the threshold couldn't reproduce the split), and both sides need min_samples_leaf rows
        n = len(xs_sorted)
        pos = np.arange(n - 1)
        ok = xs_sorted[:-1] != xs_sorted[1:]
        ok &= (pos + 1 >= self.min_samples_leaf) & (n - pos - 1 >= self.min_samples_leaf)
        return ok

    def _leaf_ids(self, X):
        # walk ALL rows down the tree together, one level per loop, instead of row by row
        X = np.asarray(X, dtype=float)
        node = np.zeros(len(X), dtype=np.int64)
        while True:
            f = self.feature_[node]
            inner = f >= 0
            if not inner.any():
                return node   # everyone has reached a leaf
            rows = np.flatnonzero(inner)
            n_in = node[rows]
            go_left = X[rows, f[rows]] <= self.threshold_[n_in]
            node[rows] = np.where(go_left, self.left_[n_in], self.right_[n_in])

    @property
    def n_nodes_(self):
        return len(self.feature_)

    @property
    def depth_(self):
        depth = np.zeros(self.n_nodes_, dtype=int)
        for i in range(self.n_nodes_):  # children are always created after their parent
            if self.feature_[i] >= 0:
                depth[self.left_[i]] = depth[self.right_[i]] = depth[i] + 1
        return int(depth.max())


class DecisionTree(_TreeBase, Classifier):
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, min_impurity_decrease=0.0, random_state=0, compact=False):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.compact = compact

    def _impurity(self, counts):
        # counts: (..., K) weighted class totals for a node -> impurity * node weight
        #   gini    = 1 - sum_k p_k^2         (chance two random picks disagree)
        #   entropy = -sum_k p_k * log2(p_k)  (bits of uncertainty)
        # multiplying by the node total makes child impurities directly addable
        tot = counts.sum(axis=-1)
        p = counts / np.maximum(tot, 1e-12)[..., None]
        if self.criterion == "gini":
            return tot * (1.0 - np.sum(p * p, axis=-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.sum(np.where(p > 0, p * np.log2(p), 0.0), axis=-1)   # 0*log(0) = 0
        return tot * ent

    def _node_impurity(self, idx):
        return float(self._impurity(self._W[idx].sum(axis=0)))

    def _is_pure(self, idx):
        return np.count_nonzero(self._W[idx].sum(axis=0) > 0) <= 1

    def _leaf_value(self, idx):
        # leaf stores the (weighted) class distribution -> that's predict_proba
        c = self._W[idx].sum(axis=0)
        return c / c.sum() if c.sum() > 0 else np.full(self.n_classes_, 1.0 / self.n_classes_)

    def _best_split_on_feature(self, x, idx):
        # sort once, then cumulative sums give the class counts on the left for EVERY possible
        # cut at the same time; right = total - left. O(n log n) per feature instead of O(n^2)
        order = np.argsort(x, kind="mergesort")
        xs = x[order]
        ok = self._valid_cut_mask(xs)
        if not ok.any():
            return None
        left = np.cumsum(self._W[idx[order]], axis=0)[:-1]
        right = left[-1] + self._W[idx[order[-1]]] - left
        child = self._impurity(left) + self._impurity(right)
        child = np.where(ok, child, np.inf)
        i = int(np.argmin(child))
        thr = 0.5 * (xs[i] + xs[i + 1])   # threshold halfway between the two neighbouring values
        return float(child[i]), thr, x <= thr

    def fit(self, X, y, sample_weight=None, n_classes=None):
        y = self._setup_classes(y, n_classes)
        X = np.asarray(X, dtype=float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        # weighted one-hot matrix: row i has w_i in its class column. every split statistic
        # is a sum of rows of this, which is how AdaBoost's sample weights come for free
        self._W = np.zeros((len(y), self.n_classes_))
        self._W[np.arange(len(y)), y] = w
        self._grow(X, X.shape[1])
        del self._W
        return self

    def predict_proba(self, X):
        return self.value_[self._leaf_ids(X)].astype(float)


# regression tree with squared-error splits, used inside gradient boosting.
# if `hess` is passed, the leaf value is a Newton step  sum(w*g) / sum(w*h)  instead of a mean
class RegressionTree(_TreeBase):
    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features=None,
                 min_impurity_decrease=0.0, random_state=0, compact=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.compact = compact

    def _node_impurity(self, idx):
        # weighted SSE around the mean, without computing the mean:
        #   SSE = sum w*t^2 - (sum w*t)^2 / sum w
        w, t = self._w[idx], self._t[idx]
        sw = w.sum()
        return float((w * t * t).sum() - (w * t).sum() ** 2 / sw) if sw > 0 else 0.0

    def _is_pure(self, idx):
        return np.ptp(self._t[idx]) == 0   # all targets identical -> nothing to split on

    def _leaf_value(self, idx):
        w = self._w[idx]
        if self._h is not None:
            return float((w * self._t[idx]).sum() / ((w * self._h[idx]).sum() + 1e-12))
        return float((w * self._t[idx]).sum() / max(w.sum(), 1e-12))

    def _best_split_on_feature(self, x, idx):
        # same cumsum trick as the classifier, using the SSE formula above on both sides
        order = np.argsort(x, kind="mergesort")
        xs = x[order]
        ok = self._valid_cut_mask(xs)
        if not ok.any():
            return None
        w, t = self._w[idx[order]], self._t[idx[order]]
        cw, cs, cq = np.cumsum(w), np.cumsum(w * t), np.cumsum(w * t * t)
        lw, ls, lq = cw[:-1], cs[:-1], cq[:-1]
        rw, rs, rq = cw[-1] - lw, cs[-1] - ls, cq[-1] - lq
        with np.errstate(divide="ignore", invalid="ignore"):
            sse = (lq - ls * ls / lw) + (rq - rs * rs / rw)
        sse = np.where(ok, sse, np.inf)
        i = int(np.argmin(sse))
        thr = 0.5 * (xs[i] + xs[i + 1])
        return float(sse[i]), thr, x <= thr

    def fit(self, X, target, sample_weight=None, hess=None):
        X = np.asarray(X, dtype=float)
        self._t = np.asarray(target, dtype=float)
        self._w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, float)
        self._h = None if hess is None else np.asarray(hess, dtype=float)
        self._grow(X, X.shape[1])
        del self._t, self._w, self._h   # training-only scratch, don't pickle it
        return self

    def predict(self, X):
        return self.value_[self._leaf_ids(X)].astype(float)
