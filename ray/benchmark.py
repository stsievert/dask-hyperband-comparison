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
from sklearn.datasets import make_circles, make_moons, make_s_curve, make_checkerboard
from sklearn.utils import check_random_state
import msgpack
from sklearn.model_selection import ShuffleSplit, ParameterSampler
from dask.distributed import get_client, LocalCluster

from dask.distributed import Client
from sklearn.base import clone

from dask_ml.model_selection import HyperbandSearchCV, IncrementalSearchCV

WDIR = "/mnt/ws/home/ssievert/ray"

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
        with open(f"{WDIR}/model-histories/{self.ident_}.msgpack", "wb") as f:
            msgpack.dump(self.history_, f)

    def partial_fit(self, X, y, classes=None, n_partial_fits=1, **kwargs):
        self._init()
        for _ in range(n_partial_fits):
            self._num_eg += len(X)
            self._pf_calls += 1
            super().partial_fit(X, y, classes=classes, **kwargs)
        return self

    def fit(self, X, y):
        assert X.shape[0] == 50_000, "X is entire training set"
        return super().fit(X, y)

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
    #  X1, y1 = make_circles(n_samples=30_000, random_state=0, noise=0.08)
    #  X2, y2 = make_circles(n_samples=30_000, random_state=1, noise=0.08)
    #  X2[:, 0] += 0.6

    X1, y1 = make_circles(n_samples=30_000, random_state=0, factor=0.8, noise=0.10)
    X2, y2 = make_circles(n_samples=30_000, random_state=1, noise=0.08)

    X2[:, 0] += 0.6
    X2[:, 1] += 0.6

    X_info = np.concatenate((X1, X2))
    y = np.concatenate((y1, y2 + 2))

    rng = check_random_state(42)
    random_feats = rng.uniform(X_info.min(), X_info.max(), size=(X_info.shape[0], 6))
    X = np.hstack((X_info, random_feats))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=int(10e3), random_state=42
    )
    return (X_train, y_train), (X_test, y_test)


def tune_ray(clf, params, X_train, y_train, X_test, y_test, n_params=-1, max_epochs=-1):
    common = dict(random_state=42)
    split = ShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
    clf = clone(clf).set_params(prefix="ray")
    from tune_sklearn import TuneSearchCV
    search = TuneSearchCV(
        clf,
        params,
        cv=split,
        early_stopping=True,
        max_iters=max_epochs,
        n_iter=n_params,
        random_state=42,
    )

    start = time()
    search.fit(X_train, y_train)#, classes=np.unique(y_train))
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


def tune_scikitlearn(
    clf,
    params,
    X_train,
    y_train,
    X_test,
    y_test,
    max_epochs=-1,
    n_params=-1,
    fits_per_score=5,
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
        test_size=0.2,
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
    clf = clone(clf).set_params(prefix="dask")
    classes = np.unique(y_train)

    chunk_size = len(X_train) * max_epochs // n_params
    max_iter = n_params

    #  chunk_size = len(X_train) / n_chunks = len(X_train) * max_epochs / n_params

    print(f"max_iter, chunk_size = {max_iter}, {chunk_size}")
    print(f"n_params = {n_params}")
    X_train = da.from_array(X_train).rechunk(chunks=(chunk_size, -1))
    y_train = da.from_array(y_train).rechunk(chunks=chunk_size)
    print(y_train.chunks)

    search = HyperbandSearchCV(clf, params, max_iter=max_iter, aggressiveness=4, random_state=42, test_size=0.2)
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
    with open(f"{WDIR}/dask-final.json", "w") as f:
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

def get_meta():
    (X_train, y_train), (X_test, y_test) = dataset()
    assert len(X_train) == 50_000
    assert len(X_test) == 10_000

    #  n_params = 4
    #  max_epochs = 10

    n_params = 100
    max_epochs = 100

    clf = Timer(max_iter=max_epochs, tol=-1, n_iter_no_change=max_epochs * 4)

    # Grid search requires: 5 x 3 x (3 * 3) x 5 x (2 * 3) = 4050
    # Let's search over 100 parameters

    params = {
        "hidden_layer_sizes": [(24,), (12,) * 2, (6,) * 4, (4,) * 6, (12, 6, 3, 3)],  # 5
        "activation": ["relu", "logistic", "tanh"],  # 3
        "alpha": loguniform(1e-6, 1e-3),  # 3 orders
        "batch_size": [32, 64, 128, 256, 512],  # 5
        "learning_rate_init": loguniform(1e-4, 1e-2),  # 2 orders
        "solver": ["adam"],
        "random_state": list(range(10_000)),
    }

    args = (params, X_train, y_train, X_test, y_test)
    return clf, args, {"n_params": n_params, "max_epochs": max_epochs}


def main():
    clf, args, common = get_meta()

    from dask.distributed import LocalCluster, Client
    # dask-worker --nprocs 8 localhost:8786
    client = Client("localhost:8786")

    import ray
    # ray start --num-cpus 8 --head
    ray.init(address='auto', redis_password='5241590000000000')

    from pprint import pprint
    sklearn_search, sklearn_data = tune_scikitlearn(clf, *args, **common)
    pprint(sklearn_data)
    dask_search, dask_data = tune_dask(clf, *args, **common)
    pprint(dask_data)
    ray_search, ray_data = tune_ray(clf, *args, **common)
    pprint(ray_data)

    df = pd.DataFrame([dask_data, ray_data, sklearn_data])
    df.to_csv("out/final.csv")

    for name, est in [("dask", dask_search), ("sklearn", sklearn_search), ("ray", ray_search)]:
        with open(f"out/{name}.pkl", "wb") as f:
            pickle.dump(est, f)

if __name__ == "__main__":
    main()
