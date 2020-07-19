import os

from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import loguniform, uniform
from time import time
import pandas as pd
from hashlib import sha256
import pickle
from io import StringIO
from pathlib import Path
import itertools
import dask.array as da
from sklearn.datasets import make_circles
from sklearn.utils import check_random_state
import msgpack
from sklearn.model_selection import ShuffleSplit, ParameterSampler
from dask.distributed import get_client

from dask.distributed import Client
from sklearn.base import clone

from dask_ml.model_selection import HyperbandSearchCV, IncrementalSearchCV
from tune_sklearn import TuneGridSearchCV


def _hash(o):
    if isinstance(o, dict):
        o = list(o.items())
    if isinstance(o, list):
        o = tuple(sorted(o))
    ir = pickle.dumps(o)

    m = sha256()
    m.update(ir)
    return m.hexdigest()


class Timer(MLPClassifier):
    def __init__(
        self,
        hidden_layer_sizes=None,
        alpha=1e-4,
        batch_size=32,
        momentum=0.9,
        n_iter_no_change=20,
        solver="sgd",
        activation="relu",
        random_state=None,
        max_iter=200,
        prefix="",
        tol=1e-4,
        learning_rate_init=1e-3,
        early_stopping=False,
    ):
        self.prefix = prefix
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            batch_size=batch_size,
            momentum=momentum,
            n_iter_no_change=n_iter_no_change,
            solver=solver,
            activation=activation,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
            learning_rate_init=learning_rate_init,
            early_stopping=early_stopping,
        )

    def _init(self):
        if not hasattr(self, "initialized_"):
            self.history_ = []
            self._pf_calls = 0
            self._score_calls = 0
            self._num_eg = 0
            self.initialized_ = True
            params = list(self.get_params().items())
            self.ident_ = self.prefix + "-" + str(_hash(params))

    def _write(self):
        abs_path = "/Users/scott/Developer/stsievert/dask-hyperband-comparison/ray"
        with open(f"{abs_path}/model-histories/{self.ident_}.msgpack", "wb") as f:
            msgpack.dump(self.history_, f)

    def partial_fit(self, X, y, classes, n_partial_fits=1, **kwargs):
        self._init()
        for _ in range(n_partial_fits):
            self._num_eg += len(X)
            self._pf_calls += 1
            super().partial_fit(X, y, classes, **kwargs)
        return self

    def fit(self, X_train, y_train, X_test, y_test):
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=420
        )
        for _ in range(self.max_iter):
            self.partial_fit(X_train, y_train, classes=np.unique(y_train))
            self.score(X_val, y_val, prefix="val_")
        self.score(X_test, y_test, prefix="test_")
        return self

    def score(self, X, y, prefix=""):
        self._init()
        score = super().score(X, y)
        self._score_calls += 1

        static = self.get_params()
        datum = {
            f"{prefix}score": score,
            "time": time(),
            "pf_calls": self._pf_calls,
            "score_calls": self._score_calls,
            "ident": self.ident_,
            "num_eg": self._num_eg,
            "n_iter_": getattr(self, "n_iter_", -1),
            **static,
        }
        self.history_.append(datum)
        self._write()
        return score


def dataset():
    X1, y1 = make_circles(n_samples=30_000, random_state=0, noise=0.04)
    X2, y2 = make_circles(n_samples=30_000, random_state=1, noise=0.04)
    X2[:, 0] += 0.6
    X_info = np.concatenate((X1, X2))
    y = np.concatenate((y1, y2 + 2))

    rng = check_random_state(42)
    random_feats = rng.uniform(-1, 1, size=(X_info.shape[0], 4))
    X = np.hstack((X_info, random_feats))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=int(10e3), random_state=42
    )
    return (X_train, y_train), (X_test, y_test)


def tune_ray(clf, params, X_train, y_train, X_test, y_test, n_params=-1, max_epochs=-1):
    common = dict(random_state=42)
    split = ShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
    clf = clone(clf).set_params(prefix="ray")
    params = list(range(50))
    search = TuneGridSearchCV(
        clf,
        params,
        cv=split,
        early_stopping=True,
        n_iter=n_params,
        max_iters=max_epochs,
        **common,
    )

    start = time()
    search.fit(X_train, y_train, classes=np.unique(y_train))
    fit_time = time() - start

    acc = search.best_estimator_.score(X_test, y_test)

    data = {
        "score": acc,
        "library": "ray",
        "accuracy": acc,
        "best_score": search.best_score_,
        "best_params": search.best_params_,
        "fit_time": fit_time,
        "start_time": start,
    }
    return search, data


def tune_sklearn(
    clf,
    params,
    X_train,
    y_train,
    X_test,
    y_test,
    max_epochs=-1,
    n_params=-1,
    fits_per_score=1,
):
    clf = clone(clf).set_params(prefix="sklearn")

    search = IncrementalSearchCV(
        clf,
        params,
        n_initial_parameters=n_params,
        max_iter=max_epochs,
        fits_per_score=fits_per_score,
        decay_rate=None,
        random_state=42,
    )

    start = time()
    search.fit(X_train, y_train, classes=np.unique(y_train))
    fit_time = time() - start

    acc = search.best_estimator_.score(X_test, y_test)

    data = {
        "score": acc,
        "library": "sklearn",
        "accuracy": acc,
        "best_score": search.best_score_,
        "best_params": search.best_params_,
        "fit_time": fit_time,
        "start_time": start,
    }
    return search, data


def tune_dask(
    clf, params, X_train, y_train, X_test, y_test, n_params=-1, max_epochs=-1
):
    common = dict(random_state=42)
    clf = clone(clf).set_params(prefix="dask")
    classes = np.unique(y_train)

    n_chunks = max(1, n_params / max_epochs)
    max_iter = n_params

    #  chunk_size = len(X_train) / n_chunks = len(X_train) * max_epochs / n_params

    print(f"max_iter, n_chunks = {max_iter}, {n_chunks}")
    print(f"n_params = {n_params}")
    X_train = da.from_array(X_train).rechunk(n_chunks=(n_chunks, -1))
    y_train = da.from_array(y_train).rechunk(n_chunks=n_chunks)
    print(y_train.chunks)

    search = HyperbandSearchCV(clf, params, max_iter=max_iter, aggressiveness=4, random_state=42)
    meta = search.metadata
    print({k: meta[k] for k in ["n_models", "partial_fit_calls"]})

    start = time()
    search.fit(X_train, y_train, classes=classes)
    fit_time = time() - start

    acc = search.best_estimator_.score(X_test, y_test)
    data = {
        "score": acc,
        "library": "dask",
        "accuracy": acc,
        "best_score": search.best_score_,
        "best_params": search.best_params_,
        "fit_time": fit_time,
        "start_time": start,
    }
    abs_path = "/Users/scott/Developer/stsievert/dask-hyperband-comparison/ray"
    with open(f"{abs_path}/dask-final.json", "w") as f:
        import json
        json.dump(data, f)
    return search, data


def run_search_and_write(
    clf, params, X_train, y_train, X_test, y_test, max_epochs=-1, n_params=-1
):
    clf = clone(clf).set_params(prefix="")
    client = get_client()

    futures = []
    params = ParameterSampler(params, n_params, random_state=42)

    args = client.scatter((X_train, y_train, X_test, y_test))
    for param in params:
        model = clone(clf).set_params(**param)
        future = client.submit(model.fit, *args)
        futures.append(future)

    start = time()
    out = client.gather(futures)
    fit_time = time() - start
    # total time: 1199.644 seconds
    # time training: 1200 * 4 = 4800 (4 Dask workers)
    # average time per model = time_training / 100 = 48
    # average time per fit + score = time_per_model / 100 = 0.48 seconds
    # latency = time_per_fit / (1 + 1.5) = 0.192

    return True
