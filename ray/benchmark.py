import os
#  os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
#  os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
#  os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
#  os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
#  os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

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

from dask.distributed import Client
from sklearn.base import clone

import msgpack
from dask_ml.model_selection import HyperbandSearchCV, IncrementalSearchCV
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from sklearn.datasets import make_circles
from sklearn.utils import check_random_state

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
        with open(f"{abs_path}/cv-results/{self.ident_}.msgpack", "wb") as f:
            msgpack.dump(self.history_, f)

    def partial_fit(self, X, y, classes, n_partial_fits=1, **kwargs):
        self._init()
        for _ in range(n_partial_fits):
            self._num_eg += len(X)
            self._pf_calls += 1
            super().partial_fit(X, y, classes, **kwargs)
        return self

    def fit(self, X, y, **kwargs):
        for _ in range(self.max_iter):
            self.partial_fit(X, y, **kwargs)
        return self

    def score(self, X, y):
        self._init()
        score = super().score(X, y)

        self._score_calls += 1
        static = self.get_params()
        datum = {
            "score": score,
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


def _dataset():
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
    search = TuneSearchCV(
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

    y_hat = search.predict(X_test)
    acc = (y_hat == y_test).sum() / len(y_hat)

    return (
        search,
        {
            "score": acc,
            "library": "ray",
            "accuracy": acc,
            "best_score": search.best_score_,
            "best_params": search.best_params_,
            "fit_time": fit_time,
            "start_time": start,
        },
    )


def tune_sklearn(
    clf, params, X_train, y_train, X_test, y_test, max_epochs=-1, n_params=-1
):
    common = dict(random_state=42)
    clf = clone(clf).set_params(prefix="sklearn", max_iter=max_epochs)

    # Need IncrementalSearchCV to test on the validation set
    search = IncrementalSearchCV(
        clf,
        params,
        n_initial_parameters=n_params,
        max_iter=max_epochs,
        fits_per_score=5,
        decay_rate=None,
        **common,
    )

    start = time()
    search.fit(X_train, y_train, classes=np.unique(y_train))
    fit_time = time() - start

    y_hat = search.predict(X_test)
    acc = (y_hat == y_test).sum() / len(y_hat)
    acc = acc.compute()

    return (
        search,
        {
            "score": acc,
            "library": "sklearn",
            "accuracy": acc,
            "best_score": search.best_score_,
            "best_params": search.best_params_,
            "fit_time": fit_time,
            "start_time": start,
        },
    )


def tune_dask(
    clf, params, X_train, y_train, X_test, y_test, n_params=-1, max_epochs=-1
):
    common = dict(random_state=42)
    clf = clone(clf).set_params(prefix="dask", max_iter=1_000_000)
    classes = np.unique(y_train)

    n_chunks = max(1, n_params / max_epochs)
    max_iter = n_params

    #  chunk_size = len(X_train) / n_chunks = len(X_train) * max_epochs / n_params

    print(f"max_iter, n_chunks = {max_iter}, {n_chunks}")
    print(f"n_params = {n_params}")
    X_train = da.from_array(X_train).rechunk(n_chunks=(n_chunks, -1))
    y_train = da.from_array(y_train).rechunk(n_chunks=n_chunks)
    print(y_train.chunks)

    search = HyperbandSearchCV(clf, params, max_iter=max_iter, **common)
    meta = search.metadata
    print({k: meta[k] for k in ["n_models"]})

    start = time()
    search.fit(X_train, y_train, classes=classes, n_partial_fits=2)
    fit_time = time() - start

    y_hat = search.predict(X_test)
    acc = (y_hat == y_test).sum() / len(y_hat)
    acc = acc.compute()

    return (
        search,
        {
            "score": acc,
            "library": "dask",
            "accuracy": acc,
            "best_score": search.best_score_,
            "best_params": search.best_params_,
            "fit_time": fit_time,
            "start_time": start,
        },
    )


if __name__ == "__main__":
    # $ dask-worker --nprocs 4 localhost:7786
    client = Client("localhost:7786")
    #  $ OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 ray start --num-cpus 4 --address localhost:6379
    import ray
    ray.init(num_cpus=4)
    print("done with ray init")
    from tune_sklearn import TuneSearchCV

    (X_train, y_train), (X_test, y_test) = _dataset()
    assert len(X_train) == 50_000
    assert len(X_test) == 10_000

    n_params = 50
    max_epochs = 100
    #  n_params = 10
    #  max_epochs = 20

    clf = Timer(max_iter=max_epochs, tol=-1, n_iter_no_change=max_epochs * 4)

    params = {
        "hidden_layer_sizes": [(24,), (12,) * 2, (6,) * 4, (4,) * 6, (12, 6, 3, 3)],
        "activation": ["relu", "logistic", "tanh"],
        "alpha": loguniform(1e-6, 1e-3),
        "batch_size": [32, 64, 128, 256, 512],
        "solver": ["adam"],
        "random_state": list(range(10_000)),
    }

    args = (clf, params, X_train, y_train, X_test, y_test)

    print("\n" * 3, "dask" + "\n" * 3)
    __start = time()
    dask_search, dask_data = tune_dask(*args, n_params=n_params, max_epochs=max_epochs)
    print("Time for dask:", time() - __start)

    print("\n" * 3, "ray" + "\n" * 3)
    __start = time()
    ray_search, ray_data = tune_ray(*args, n_params=n_params, max_epochs=max_epochs)
    print("Time for ray:", time() - __start)

    print("\n" * 3, "sklearn" + "\n" * 3)
    __start = time()
    sklearn_search, sklearn_data = tune_sklearn(
        *args, n_params=n_params, max_epochs=max_epochs
    )
    print("Time for sklearn:", time() - __start)

    data = [ray_data] + [dask_data] + [sklearn_data]
    df = pd.DataFrame(data)
    df.to_csv("out/final.csv", index=False)

    for library, search in [
        ("sklearn", sklearn_search),
        ("ray", ray_search),
        ("dask", dask_search),
    ]:
        cv_res = pd.DataFrame(search.cv_results_)
        cv_res.to_csv(f"out/{library}-cv-results.csv", index=False)
