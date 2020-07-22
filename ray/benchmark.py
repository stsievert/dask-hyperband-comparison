import os
from time import sleep

from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
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
from sklearn.datasets import fetch_covtype
from sklearn.utils import check_random_state
import msgpack
from sklearn.model_selection import ShuffleSplit, ParameterSampler
from dask.distributed import get_client, LocalCluster

from dask.distributed import Client
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import RandomizedSearchCV

from dask_ml.model_selection import HyperbandSearchCV, IncrementalSearchCV

WDIR = "/Users/scott/Developer/stsievert/dask-hyperband-comparison/ray"


def _hash(o):
    if isinstance(o, dict):
        o = list(o.items())
    if isinstance(o, list):
        o = tuple(sorted(o))
    ir = pickle.dumps(o)

    m = sha256()
    m.update(ir)
    return m.hexdigest()


class ConstantFunction(BaseEstimator):
    def __init__(self, value=0, max_iter=100, latency=0.1, n_jobs=1, prefix=""):
        self.value = value
        self.max_iter = max_iter
        self.latency = latency
        self.n_jobs = n_jobs
        self.prefix = prefix
        super().__init__()

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
        return
        # with open(f"{WDIR}/model-histories/{self.ident_}.msgpack", "wb") as f:
        #     msgpack.dump(self.history_, f)

    def partial_fit(self, X, y, _=None):
        self._init()
        sleep(self.latency * 1.5 * len(X))
        return self

    def fit(self, X, y, _=None):
        for _ in range(self.max_iter):
            self.partial_fit(X, y)
        return self

    def score(self, X, y, prefix=""):
        self._init()
        sleep(self.latency * 1.0 * len(X))
        self._score_calls += 1
        score = self.value

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


def tune_ray(
    clf, params, X_train, y_train, X_test, y_test, n_params=-1, max_epochs=-1, n_jobs=4
):
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
        refit=False,
        n_jobs=n_jobs,
    )

    start = time()
    search.fit(X_train, y_train)
    fit_time = time() - start

    data = {
        "library": "ray",
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
    fits_per_score=1,
    n_jobs=4,
):
    clf = clone(clf).set_params(prefix="sklearn")

    split = ShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
    search = RandomizedSearchCV(
        clf,
        params,
        cv=split,
        n_iter=n_params,
        random_state=42,
        n_jobs=n_jobs,
        refit=False,
    )

    start = time()
    search.fit(X_train, y_train)
    fit_time = time() - start

    data = {
        "library": "sklearn",
        "best_score": search.best_score_,
        "best_params": search.best_params_,
        "fit_time": fit_time,
        "start_time": start,
    }
    return search, data


def tune_dask(
    clf, params, X_train, y_train, X_test, y_test, n_params=-1, max_epochs=-1, n_jobs=4
):
    clf = clone(clf).set_params(prefix="dask")

    chunk_size = len(X_train) * max_epochs // n_params
    max_iter = n_params

    print(f"max_iter, chunk_size = {max_iter}, {chunk_size}")
    print(f"n_params = {n_params}")
    X_train = da.from_array(X_train).rechunk(chunks=(chunk_size, -1))
    y_train = da.from_array(y_train).rechunk(chunks=chunk_size)
    print(y_train.chunks)

    search = HyperbandSearchCV(
        clf, params, max_iter=max_iter, random_state=42, test_size=0.2,
    )
    meta = search.metadata
    print({k: meta[k] for k in ["n_models", "partial_fit_calls"]})

    start = time()
    search.fit(X_train, y_train)
    fit_time = time() - start

    data = {
        "library": "dask",
        "best_score": search.best_score_,
        "best_params": search.best_params_,
        "fit_time": fit_time,
        "start_time": start,
    }
    return search, data


def get_meta():
    # total time: 1199.644 seconds
    # time training: 1200 * 4 = 4800 (4 Dask workers)
    # average time per model = time_training / 100 = 48
    # average time per fit + score = time_per_model / 100 = 0.48 seconds
    # latency = time_per_fit / (1 + 1.5) = 0.192

    N = 50_000
    X = np.random.uniform(size=(N, 784))
    y = np.random.choice(2, size=N)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #  n_params = 4
    #  max_epochs = 10
    n_params = 100
    max_epochs = 100

    n_jobs = 8

    clf = ConstantFunction(latency=0.1 / 50e3, n_jobs=n_jobs)

    params = {"value": uniform(0, 1)}

    args = (params, X_train, y_train, X_test, y_test)
    return clf, args, {"n_params": n_params, "max_epochs": max_epochs, "n_jobs": n_jobs}


def main():
    clf, args, meta = get_meta()

    from dask.distributed import LocalCluster, Client

    #  cluster = LocalCluster(n_workers=meta["n_jobs"])
    #  client = Client(cluster)
    client = Client("localhost:8786")
    print("Dask initialized")

    import ray

    #  ray.init(num_cpus=meta["n_jobs"])
    ray.init(address='auto', redis_password='5241590000000000')
    print("Ray initialized")

    from pprint import pprint

    sklearn_search, sklearn_data = tune_scikitlearn(clf, *args, **meta)
    pprint(sklearn_data)
    dask_search, dask_data = tune_dask(clf, *args, **meta)
    pprint(dask_data)
    ray_search, ray_data = tune_ray(clf, *args, **meta)
    pprint(ray_data)

    df = pd.DataFrame([dask_data, ray_data, sklearn_data])
    df.to_csv(f"out/final-{meta['n_jobs']}.csv")

    for name, est in [
        ("dask", dask_search),
        ("sklearn", sklearn_search),
        ("ray", ray_search),
    ]:
        with open(f"out/{name}-{meta['n_jobs']}.pkl", "wb") as f:
            pickle.dump(est, f)


if __name__ == "__main__":
    main()
