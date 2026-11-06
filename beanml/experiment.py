from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .base import Pipeline
from .bayes import GaussianNB
from .discriminant import LDA, QDA
from .ensemble import AdaBoostSAMME, GradientBoosting, RandomForest, SoftVoting
from .io import env_info, load_json, load_pickle, save_json, save_pickle
from .linear import AveragedPerceptron, LinearSVM, SoftmaxRegression
from .neighbors import KNN
from .neural import MLP
from .preprocessing import LabelEncoder, dedupe, stratified_split
from .svm import KernelSVM
from .tree import DecisionTree
from .tuning import grid_search, knn_search

# the whole training run lives here: load -> dedupe -> split -> tune -> fit -> save -> score.
# train.py and the notebook both just call run(), so they can never drift apart

ROOT = Path(__file__).resolve().parents[1]
TARGET = "Class"
# last 4 digits of my student id (2026AJ05327) - a plain 42 would match half the class
SEED = 5327

METRIC_COLS = [("accuracy", "Accuracy"), ("auc", "AUC"), ("precision", "Precision"),
               ("recall", "Recall"), ("f1", "F1"), ("mcc", "MCC")]


# every model in one list: slug, display name, required?, factory, scale?, CV grid, folds.
# grid=None means fixed hyper-parameters (the slow ensembles), given in `fixed`
def model_specs(seed=SEED):
    return [
        dict(slug="logistic_regression", name="Logistic Regression", required=True,
             make=lambda **p: SoftmaxRegression(seed=seed, **p), scale=True,
             grid={"lr": [0.01, 0.1], "l2": [0.0, 1e-4, 1e-3, 1e-2]}, folds=5),
        dict(slug="decision_tree", name="Decision Tree", required=True,
             make=lambda **p: DecisionTree(random_state=seed, **p), scale=False,
             grid={"criterion": ["gini", "entropy"], "max_depth": [6, 8, 10, 12, None],
                   "min_samples_leaf": [1, 5]}, folds=5),
        dict(slug="knn", name="KNN", required=True,
             make=lambda **p: KNN(**p), scale=True,
             grid={"k": list(range(1, 32, 2)), "weights": ["uniform", "distance"],
                   "metric": ["euclidean", "manhattan"]}, folds=5),
        dict(slug="naive_bayes", name="Naive Bayes", required=True,
             make=lambda **p: GaussianNB(**p), scale=True,
             grid={"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1]}, folds=5),
        # C x gamma grid spans a few orders of magnitude; 3 folds since each fit solves 21 SMO problems
        dict(slug="svm", name="Support Vector Machine", required=True,
             make=lambda **p: KernelSVM(seed=seed, **p), scale=True,
             grid={"C": [1.0, 10.0, 100.0, 1000.0], "gamma": [0.003, 0.01, 0.03, 0.1]}, folds=3),
        # ---- extras, beyond the 5 the assignment asks for ----
        dict(slug="linear_svm", name="Linear SVM", required=False,
             make=lambda **p: LinearSVM(seed=seed, **p), scale=True,
             grid={"C": [0.1, 1.0, 10.0]}, folds=5),
        dict(slug="perceptron", name="Averaged Perceptron", required=False,
             make=lambda **p: AveragedPerceptron(seed=seed, **p), scale=True, grid=None, folds=0),
        dict(slug="lda", name="LDA", required=False,
             make=lambda **p: LDA(**p), scale=True,
             grid={"shrinkage": [0.0, 1e-3, 1e-2, 0.1]}, folds=5),
        dict(slug="qda", name="QDA", required=False,
             make=lambda **p: QDA(**p), scale=True,
             grid={"reg_param": [1e-3, 1e-2, 0.1]}, folds=5),
        dict(slug="random_forest", name="Random Forest", required=False,
             make=lambda **p: RandomForest(seed=seed, **p), scale=False, grid=None, folds=0,
             fixed={"n_estimators": 100, "max_features": "sqrt", "min_samples_leaf": 1}),
        dict(slug="adaboost", name="AdaBoost (SAMME)", required=False,
             make=lambda **p: AdaBoostSAMME(seed=seed, **p), scale=False, grid=None, folds=0,
             fixed={"n_estimators": 150, "max_depth": 2, "learning_rate": 0.5}),
        dict(slug="gradient_boosting", name="Gradient Boosting", required=False,
             make=lambda **p: GradientBoosting(seed=seed, **p), scale=False, grid=None, folds=0,
             fixed={"n_estimators": 300, "learning_rate": 0.1, "max_depth": 3}),
        dict(slug="mlp", name="MLP Neural Net", required=False,
             make=lambda **p: MLP(seed=seed, **p), scale=True,
             grid={"hidden": [(64, 32), (128, 64)], "l2": [1e-4, 1e-3]}, folds=5),
    ]


# a mix of model families that all give real probabilities
VOTING_MEMBERS = ["logistic_regression", "knn", "random_forest", "gradient_boosting", "mlp"]


# read the csv, tidy the labels, drop duplicates (before any splitting!)
def load_dataset(path=None):
    df = pd.read_csv(path or ROOT / "data" / "dry_bean.csv")
    features = [c for c in df.columns if c != TARGET]
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df, report = dedupe(df, features, TARGET)
    return df, features, report


def _fmt(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


# plain text version of the results table for the console
def summary_table(test_metrics):
    head = f"{'ML Model Name':26s}" + "".join(f"{c:>11s}" for _, c in METRIC_COLS)
    out = []
    for title, req in [("Required models", True), ("Extra models", False)]:
        out += [title, head, "-" * len(head)]
        for r in test_metrics["models"].values():
            if r["required"] == req:
                out.append(f"{r['name']:26s}" + "".join(f"{_fmt(r['metrics'][k]):>11s}" for k, _ in METRIC_COLS))
        out.append("")
    return "\n".join(out)


# load every pickle fresh from disk and re-read test_data.csv - exactly what the app does,
# so the README numbers and the app numbers come from the same code path
def score_saved_models(model_dir, test_csv, meta):
    test = pd.read_csv(test_csv)
    le = LabelEncoder()
    le.classes_ = np.array(meta["class_names"])
    y = le.transform(test[TARGET].astype(str).str.strip())
    k = len(le.classes_)
    pipes = {m["slug"]: load_pickle(Path(model_dir) / m["file"]) for m in meta["models"]}
    # the voter only knows its members' names, hand it the actual loaded models
    if "soft_voting" in pipes:
        pipes["soft_voting"].attach(pipes)
    out = {}
    for m in meta["models"]:
        p = pipes[m["slug"]]
        s = p.scores(test[meta["feature_names"]])
        pred = np.argmax(s, axis=1)
        out[m["slug"]] = {"name": m["name"], "required": m["required"],
                          "metrics": M.evaluate(y, pred, s, k),
                          "confusion_matrix": M.confusion_matrix(y, pred, k)}
    return out


def run(quick=False, run_cv=True, seed=SEED, out_dir=None, test_csv=None, data_path=None, verbose=True):
    out_dir = Path(out_dir or ROOT / "model")
    test_csv = Path(test_csv or ROOT / "test_data.csv")
    say = print if verbose else (lambda *a, **k: None)
    t_all = time.time()

    df, features, report = load_dataset(data_path)
    say(f"data: {report}")
    le = LabelEncoder().fit(df[TARGET])
    y_all = le.transform(df[TARGET])
    # stratified 80/20 split - the 20% is written out as test_data.csv and never touched until the end
    tr, te = stratified_split(y_all, test_size=0.2, seed=seed)
    train_df, test_df = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    test_df.to_csv(test_csv, index=False)
    say(f"train {len(train_df)} rows / test {len(test_df)} rows -> {test_csv.name}")
    X = train_df[features].to_numpy(dtype=float)
    y = le.transform(train_df[TARGET])

    # quick mode skips the grid search and reuses the best params saved by an earlier full run
    do_cv = run_cv and not quick
    old_cv = {}
    cv_path = out_dir / "cv_results.json"
    if not do_cv and cv_path.exists():
        old_cv = load_json(cv_path)
    cv_results, fitted, meta_models = {}, {}, []

    for spec in model_specs(seed):
        slug = spec["slug"]
        say(f"\n[{spec['name']}]")
        t0 = time.time()
        if spec["grid"] is None:
            params = spec.get("fixed", {})
        elif do_cv:
            if slug == "knn":
                res = knn_search(spec["grid"], X, y, folds=spec["folds"], seed=seed, verbose=verbose)
            else:
                res = grid_search(spec["make"], spec["grid"], X, y, scale=spec["scale"],
                                  folds=spec["folds"], seed=seed, verbose=verbose)
            params = res[0]["params"]
            cv_results[slug] = {"folds": spec["folds"], "best": res[0], "all": res}
            say(f"  best {params}  cv macro-F1 {res[0]['f1_mean']:.4f}  ({time.time() - t0:.1f}s)")
        else:
            prev = old_cv.get(slug)
            params = prev["best"]["params"] if prev else {}
            if prev:
                cv_results[slug] = prev
            say(f"  reusing params {params}")
        # json turns tuples into lists, put the MLP layer sizes back
        if "hidden" in params:
            params["hidden"] = tuple(params["hidden"])

        # refit the winning config on the whole training split
        t1 = time.time()
        pipe = Pipeline(spec["make"](**params), scale=spec["scale"], feature_names=features,
                        class_names=le.classes_, name=spec["name"]).fit(X, y)
        fit_secs = time.time() - t1
        say(f"  final fit {fit_secs:.1f}s")
        fname = f"{slug}.pkl"
        save_pickle(pipe, out_dir / fname)
        fitted[slug] = pipe
        meta_models.append({"slug": slug, "name": spec["name"], "required": spec["required"],
                            "file": fname, "params": params, "tuned": spec["grid"] is not None,
                            "fit_seconds": round(fit_secs, 2), "has_proba": pipe.has_proba})

    # nothing to train for the voter, it just averages the members above
    voter = SoftVoting(VOTING_MEMBERS, le.classes_, features)
    save_pickle(voter, out_dir / "soft_voting.pkl")
    meta_models.append({"slug": "soft_voting", "name": voter.name, "required": False,
                        "file": "soft_voting.pkl", "params": {"members": VOTING_MEMBERS},
                        "tuned": False, "fit_seconds": 0.0, "has_proba": True})

    # everything the app needs to know about the models, plus the env for the lab proof
    meta = {"feature_names": features, "class_names": le.classes_.tolist(), "target": TARGET,
            "seed": seed, "test_size": 0.2, "dataset": report,
            "class_counts_clean": df[TARGET].value_counts().to_dict(),
            "n_train": len(train_df), "n_test": len(test_df), "env": env_info(), "models": meta_models}
    save_json(meta, out_dir / "meta.json")
    if cv_results:
        save_json(cv_results, cv_path)

    # test set is used exactly once, right here, for reporting
    scored = score_saved_models(out_dir, test_csv, meta)
    test_metrics = {"class_names": meta["class_names"], "n_test": len(test_df), "models": scored}
    save_json(test_metrics, out_dir / "test_metrics.json")
    say("\n" + summary_table(test_metrics))
    say(f"total time {time.time() - t_all:.1f}s")
    return {"meta": meta, "cv": cv_results, "test": test_metrics, "pipelines": fitted,
            "train_df": train_df, "test_df": test_df}
