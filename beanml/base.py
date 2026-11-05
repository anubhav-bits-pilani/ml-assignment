from __future__ import annotations

import inspect

import numpy as np

from .preprocessing import StandardScaler


def softmax(z):
    # softmax(z)_k = exp(z_k) / sum_j exp(z_j)
    # subtracting the row max first doesn't change the answer (it cancels top and bottom)
    # but it stops exp() from overflowing when the scores get big
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# tiny common base so every model looks the same from the outside:
# fit(X, y) with y as ints 0..K-1, then predict / predict_proba or decision_function
class Classifier:
    has_proba = True

    def get_params(self):
        # read the constructor arguments back off the object - handy for cloning in CV
        names = [p for p in inspect.signature(type(self).__init__).parameters if p != "self"]
        return {n: getattr(self, n) for n in names}

    def scores(self, X):
        # whatever we rank on for AUC: probabilities if the model has them, raw margins if not
        return self.predict_proba(X) if self.has_proba else self.decision_function(X)

    def predict(self, X):
        # predicted class = the column with the highest score
        return np.argmax(self.scores(X), axis=1)

    def _setup_classes(self, y, n_classes=None):
        y = np.asarray(y, dtype=int)
        # ensembles pass n_classes explicitly because a bootstrap sample might miss a class,
        # and then y.max()+1 would come out too small
        self.n_classes_ = int(n_classes) if n_classes is not None else int(y.max()) + 1
        return y


def clone(model):
    # fresh, unfitted copy with the same hyper-parameters
    return type(model)(**model.get_params())


# scaler (optional) + model + the label / feature bookkeeping the app needs.
# this is the object that actually gets pickled into model/
class Pipeline:
    def __init__(self, model, scale=True, feature_names=None, class_names=None, name=""):
        self.model = model
        self.scale = scale
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.class_names = list(class_names) if class_names is not None else None
        self.name = name
        self.scaler = None

    def _matrix(self, X):
        # if we get a DataFrame, pick the columns by name so the csv column order doesn't matter
        if hasattr(X, "loc") and self.feature_names is not None:
            X = X[self.feature_names]
        X = np.asarray(X, dtype=float)
        # reuse the mean/std learned on the training data - never refit on new data
        return self.scaler.transform(X) if self.scaler is not None else X

    def fit(self, X, y):
        if hasattr(X, "loc") and self.feature_names is not None:
            X = X[self.feature_names]
        X = np.asarray(X, dtype=float)
        # scaler is fit on whatever we train on (a CV fold or the full train split), nothing else,
        # so no test information leaks into the scaling
        self.scaler = StandardScaler().fit(X) if self.scale else None
        self.model.fit(self.scaler.transform(X) if self.scaler else X, y)
        return self

    def scores(self, X):
        return self.model.scores(self._matrix(X))

    def predict(self, X):
        return self.model.predict(self._matrix(X))

    def predict_labels(self, X):
        # turn the class index back into the bean name, e.g. 3 -> "DERMASON"
        return np.asarray(self.class_names)[self.predict(X)]

    @property
    def has_proba(self):
        return self.model.has_proba
