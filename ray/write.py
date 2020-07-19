from time import time

from dask.distributed import Client
from scipy.stats import loguniform

from benchmark import run_search_and_write, Timer, dataset

(X_train, y_train), (X_test, y_test) = dataset()
assert len(X_train) == 50_000
assert len(X_test) == 10_000

#  n_params = 100
#  max_epochs = 100

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

if __name__ == "__main__":
    client = Client("localhost:7786")
    client.upload_file("benchmark.py")

    print("Starting search...")
    __start = time()
    run_search_and_write(clf, *args, n_params=n_params, max_epochs=max_epochs)
    print("Time for search:", time() - __start)
