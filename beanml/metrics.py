from __future__ import annotations

import numpy as np

# all the evaluation maths, by hand. labels are always ints 0..K-1 in here


def confusion_matrix(y_true, y_pred, n_classes):
    # cm[i, j] = how many beans of true class i got predicted as class j
    # rows = actual, columns = predicted, so the diagonal is everything we got right
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    # add.at does the "+1 at (true, pred)" for every row in one go, duplicates included
    np.add.at(cm, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return cm


def accuracy(y_true, y_pred):
    # accuracy = correct / total
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _safe_div(a, b):
    # a / b, but 0/0 becomes 0 instead of nan (same idea as zero_division=0)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.zeros_like(a)
    np.divide(a, b, out=out, where=b != 0)
    return out


def per_class_prf(cm):
    # for each class k, treating it as "positive" vs everything else:
    #   TP_k = cm[k, k]
    #   precision_k = TP_k / (column sum)  -> of everything we called k, how much really was k
    #   recall_k    = TP_k / (row sum)     -> of all the real k's, how many we caught
    #   F1_k = 2PR / (P + R)               -> harmonic mean, punishes a big gap between P and R
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0)
    actual_pos = cm.sum(axis=1)
    precision = _safe_div(tp, pred_pos)
    recall = _safe_div(tp, actual_pos)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1, actual_pos


def macro_weighted_prf(cm):
    p, r, f, support = per_class_prf(cm)
    # macro = plain mean over classes (BOMBAY counts as much as DERMASON)
    # weighted = mean weighted by how many true rows each class has
    # only classes that actually show up (in truth or predictions) count towards macro,
    # otherwise a test file missing a class would get dragged down by fake zeros
    present = (cm.sum(axis=0) + cm.sum(axis=1)) > 0
    w = support / support.sum() if support.sum() else np.zeros_like(support, dtype=float)
    macro = tuple(float(m[present].mean()) if present.any() else 0.0 for m in (p, r, f))
    weighted = tuple(float((m * w).sum()) for m in (p, r, f))
    return macro, weighted


def mcc(cm):
    # multiclass Matthews correlation (Gorodkin 2004):
    #   c = correctly classified (trace), s = total samples
    #   t_k = true count of class k (row sums), p_k = predicted count of class k (column sums)
    #   MCC = (c*s - sum_k t_k*p_k) / sqrt((s^2 - sum_k p_k^2) * (s^2 - sum_k t_k^2))
    # for K = 2 this works out to exactly the usual (TP*TN - FP*FN) / sqrt(...) formula
    cm = np.asarray(cm, dtype=float)
    s = cm.sum()
    c = np.trace(cm)
    t = cm.sum(axis=1)
    p = cm.sum(axis=0)
    num = c * s - t @ p
    den = np.sqrt(s * s - p @ p) * np.sqrt(s * s - t @ t)
    # den = 0 happens when everything is predicted (or is) one class -> no correlation, call it 0
    return float(num / den) if den > 0 else 0.0


def _average_ranks(x):
    # 1-based ranks of x, where tied values all get the average of the ranks they span
    # e.g. [10, 20, 20, 30] -> [1, 2.5, 2.5, 4]
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts       # how many values are strictly smaller
    avg = starts + (counts + 1) / 2.0         # mean of ranks starts+1 .. starts+count
    return avg[inv.ravel()]


def binary_auc(y01, score):
    # AUC = probability that a random positive gets a higher score than a random negative
    # (ties count as half). That's the Mann-Whitney U statistic:
    #   AUC = (sum of positive ranks - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    # the n_pos*(n_pos+1)/2 bit is the smallest the positives' rank-sum could possibly be
    y01 = np.asarray(y01).astype(bool)
    n_pos = int(y01.sum())
    n_neg = len(y01) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")   # AUC isn't defined without both classes
    ranks = _average_ranks(np.asarray(score, dtype=float))
    return float((ranks[y01].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(y01, score):
    # sweep the threshold from high to low; at each distinct score we record
    #   TPR = TP / all positives,  FPR = FP / all negatives
    y01 = np.asarray(y01).astype(bool)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    s, y = score[order], y01[order]
    # only cut at the last position of each run of equal scores - tied rows move together
    cut = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tps = np.cumsum(y)[cut]
    fps = (cut + 1) - tps                     # everything above the cut that isn't a TP
    tpr = np.r_[0.0, tps / max(y.sum(), 1)]
    fpr = np.r_[0.0, fps / max((~y).sum(), 1)]
    return fpr, tpr


def trapezoid(y, x):
    # area under a curve with the trapezoid rule: sum of (width * average height)
    # written by hand because numpy renamed trapz -> trapezoid between versions
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def macro_ovr_auc(y_true, scores, n_classes):
    # one-vs-rest: class k vs all the others, scored on column k, then a plain average.
    # classes missing from y_true (or the only class present) get skipped as nan
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    per_class = np.full(n_classes, np.nan)
    for k in range(n_classes):
        per_class[k] = binary_auc(y_true == k, scores[:, k])
    valid = ~np.isnan(per_class)
    macro = float(per_class[valid].mean()) if valid.any() else float("nan")
    return macro, per_class


def evaluate(y_true, y_pred, scores, n_classes):
    # everything the README / app reports, in one dict
    cm = confusion_matrix(y_true, y_pred, n_classes)
    (p, r, f), (pw, rw, fw) = macro_weighted_prf(cm)
    auc, _ = macro_ovr_auc(y_true, scores, n_classes) if scores is not None else (float("nan"), None)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "auc": auc,
        "precision": p,
        "recall": r,
        "f1": f,
        "mcc": mcc(cm),
        "precision_weighted": pw,
        "recall_weighted": rw,
        "f1_weighted": fw,
        "n_samples": int(len(y_true)),
    }


def classification_report(y_true, y_pred, class_names):
    # per-class rows + macro + weighted rows, as dicts so the app can turn it into a table
    k = len(class_names)
    cm = confusion_matrix(y_true, y_pred, k)
    p, r, f, sup = per_class_prf(cm)
    rows = [{"class": class_names[i], "precision": p[i], "recall": r[i], "f1": f[i],
             "support": int(sup[i])} for i in range(k)]
    (mp, mr, mf), (wp, wr, wf) = macro_weighted_prf(cm)
    total = int(sup.sum())
    rows.append({"class": "macro avg", "precision": mp, "recall": mr, "f1": mf, "support": total})
    rows.append({"class": "weighted avg", "precision": wp, "recall": wr, "f1": wf, "support": total})
    return rows
