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
from tune_sklearn import TuneSearchCV
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from sklearn.datasets import make_circles
from sklearn.utils import check_random_state

import ray

#  ray.init()


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
            params = list(self.get_params().items())
            self.history_ = []
            self._pf_calls = 0
            self._score_calls = 0
            self._num_eg = 0
            self.initialized_ = True
            self.ident_ = self.prefix + "-" + str(_hash(params))

    def _write(self):
        abs_path = "/Users/scott/Developer/stsievert/dask-hyperband-comparison/ray"
        with open(f"{abs_path}/cv-results/{self.ident_}.msgpack", "wb") as f:
            msgpack.dump(self.history_, f)

    def partial_fit(self, X, y, classes, **kwargs):
        self._init()
        self._pf_calls += 1
        self._num_eg += len(X)
        return super().partial_fit(X, y, classes, **kwargs)

    def score(self, X, y):
        self._init()
        self._write()

        self._score_calls += 1
        score = super().score(X, y)
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
        return score


def _hash(o):
    if isinstance(o, dict):
        o = list(o.items())
    if isinstance(o, list):
        o = tuple(sorted(o))
    ir = pickle.dumps(o)

    m = sha256()
    m.update(ir)
    return m.hexdigest()


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


def _get_chunks(n, chunksize):
    leftover = n % chunksize
    n_chunks = n // chunksize

    chunks = [chunksize] * n_chunks
    if leftover:
        chunks.append(leftover)
    return tuple(chunks)


def test_get_chunksize():
    N = [201, 49, 531, 5030]
    Chunks = [48, 4, 10, 49]
    for n, chunks in itertools.product(N, Chunks):
        x = np.arange(n)
        c = da.from_array(x, chunks=chunks).chunks[0]
        c_hat = _get_chunks(n, chunks)
        assert sum(c_hat) == n
        assert c == c_hat


def _get_even_chunks(arr: da.Array, n_chunks: int, eps: float = 0.1) -> da.Array:
    chunk_size = len(arr) / n_chunks
    min_chunks = int(chunk_size)
    max_chunks = int(chunk_size) + 1
    possible_chunksizes = range(min_chunks, max_chunks + 1)

    possible_chunks = [
        _get_chunks(len(arr), chunksize) for chunksize in possible_chunksizes
    ]
    diffs = [max(chunks) - min(chunks) for chunks in possible_chunks]
    idx = np.argmin(diffs)
    return possible_chunksizes[idx]


def test_get_even_chunks():
    x = np.random.uniform(size=1321)
    for n_chunks in range(3, 12):
        chunksize = _get_even_chunks(x, n_chunks)
        y = da.from_array(x, chunksize)
        assert len(y.chunks[0]) == n_chunks


def tune_ray(clf, params, X_train, y_train, X_test, y_test, hparams=None):
    common = dict(random_state=42)
    split = ShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
    clf = clone(clf).set_params(prefix="ray")
    search = TuneSearchCV(
        clf, params, cv=split, early_stopping=True, **hparams or {}, **common,
    )

    start = time()
    search.fit(X_train, y_train)
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


def tune_sklearn(clf, params, X_train, y_train, X_test, y_test, hparams=None):
    common = dict(random_state=42)
    split = ShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
    clf = clone(clf).set_params(prefix="sklearn")
    search = IncrementalSearchCV(
        clf, params, decay_rate=None, **common, **hparams or {},
    )

    start = time()
    search.fit(X_train, y_train, classes=list(range(4)))
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


def tune_dask(clf, params, X_train, y_train, X_test, y_test, hparams=None):
    common = dict(random_state=42)
    n_iter_no_change = clf.get_params()["n_iter_no_change"]
    clf = clone(clf).set_params(prefix="dask")

    n_params = hparams["n_params"]
    max_epochs = hparams["max_epochs"]

    max_iter = n_params
    n_chunks = max_epochs / n_params
    chunk_size = _get_even_chunks(y_train, n_chunks)

    X_train = da.from_array(X_train, chunks=(chunk_size, -1))
    y_train = da.from_array(y_train, chunks=chunk_size)
    clf = clf.set_params(
        n_iter_no_change=int(n_iter_no_change * n_chunks) + 1,
        max_iter=int(max_iter * n_chunks) + 1
    )

    search = HyperbandSearchCV(
        clf,
        params,
        aggressiveness=4,
        max_iter=max_iter,
        **common,
    )

    start = time()
    search.fit(X_train, y_train, classes=list(range(4)))
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
    test_get_chunksize()
    test_get_even_chunks()

    #  client = Client("localhost:7786")
    client = Client()
    ray.init()

    (X_train, y_train), (X_test, y_test) = _dataset()
    assert len(X_train) == 50_000
    assert len(X_test) == 10_000

    n_params = 75
    max_epochs = 200
    #  n_params = 10
    #  max_epochs = 20

    tol = 0.001
    patience = max_epochs // 3

    clf = Timer(max_iter=max_epochs, tol=tol, n_iter_no_change=patience)

    params = {
        "hidden_layer_sizes": [(24,), (12,) * 2, (6,) * 4, (4,) * 6, (12, 6, 3, 3)],
        "alpha": loguniform(1e-6, 1e-3),
        "batch_size": [32, 64, 128, 256, 512],
        "solver": ["sgd", "adam"],
        "activation": ["relu"],
        "random_state": list(range(10_000)),
    }

    args = (clf, params, X_train, y_train, X_test, y_test)
    print("\n" * 3, "dask" + "\n" * 3)
    dask_search, dask_data = tune_dask(
        *args, hparams={"n_params": n_params, "max_epochs": max_epochs}
    )

    print("\n" * 3, "ray" + "\n" * 3)
    ray_search, ray_data = tune_ray(
        *args, hparams={"n_iter": n_params, "max_iters": max_epochs}
    )

    print("\n" * 3, "sklearn" + "\n" * 3)
    sklearn_search, sklearn_data = tune_sklearn(*args, hparams={"n_initial_parameters": n_params, "max_iter": max_epochs})

    data = [ray_data] + [dask_data] + [sklearn_data]
    df = pd.DataFrame(data)
    df.to_csv("out/final.csv", index=False)

    for library, search in [
        ("ray", ray_search),
        ("dask", dask_search),
        ("sklearn", sklearn_search),
    ]:
        cv_res = pd.DataFrame(search.cv_results_)
        cv_res.to_csv(f"out/{library}-cv-results.csv", index=False)
