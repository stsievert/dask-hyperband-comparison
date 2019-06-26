#from keras.datasets import mnist
import torch
from torchvision import datasets, transforms
import numpy as np
import random

import skimage.filters
import skimage
import scipy.signal
from sklearn.utils import check_random_state
from skimage.util import random_noise


def noise_img(x, random_state=None):
    assert isinstance(random_state, float) or random_state is None
    if random_state:
        y = x * random_state
        # assert y.ndim == 1
        random_state = abs(hash(tuple(y))) % (2 ** 30)
        # random_state = (np.abs(x) * random_state).sum() % (2 ** 30)
    else:
        random_state = None
    random_state = check_random_state(random_state)
    noises = [
        {"mode": "s&p", "amount": random_state.uniform(0.1, 0.1)},
        {"mode": "gaussian", "var": random_state.uniform(0.02, 0.12)},
    ]
    # noise = random.choice(noises)
    noise = noises[1]
    seed = random_state.randint(2 ** 30)
    img = random_noise(x, seed=seed, **noise)
    return np.clip(img, 0, 1)


def train_formatting(img):
    img = img.reshape(28, 28).astype("float32")
    return img.flat[:]


def blur_img(img):
    assert img.ndim == 1
    n = int(np.sqrt(img.shape[0]))
    img = img.reshape(n, n)
    h = np.zeros((n, n))
    angle = np.random.uniform(-5, 5)
    w = random.choice(range(1, 3))
    h[n // 2, n // 2 - w : n // 2 + w] = 1
    h = skimage.transform.rotate(h, angle)
    h /= h.sum()
    y = scipy.signal.convolve(img, h, mode="same")
    return y.flat[:]


def _get_dataset(library="pytorch"):
    if library == "mnist":
        (x_train, _), (x_test, _) = mnist.load_data()
        x = np.concatenate((x_train, x_test))
        if n:
            x = x[:n]
        else:
            n = int(70e3)
        x = x.astype("float32") / 255.
        x = np.reshape(x, (len(x), 28 * 28))
        return x
    elif library == "pytorch":
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transform)
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True,
                           transform=transform)
        )
        train_imgs = [images.numpy() for images, labels in train_loader]
        test_imgs = [images.numpy() for images, labels in test_loader]
        imgs = train_imgs + test_imgs
        x = np.concatenate(imgs)
        assert x.shape == (70_000, 1, 28, 28)
        x = x.reshape(len(x), 28*28)
        assert 0 <= x.min() and 0.9 <= x.max() <= 1
        return x
    else:
        raise ValueError("wrong library to get MNIST dataset")

def dataset(n=None, random_state=None):
    random_state = check_random_state(random_state)
    x = _get_dataset()
    y = np.apply_along_axis(train_formatting, 1, x)

    clean = y.copy()

    noisy = y.copy()
    # order = [noise_img, blur_img]
    # order = [blur_img]
    order = [noise_img]

    random.shuffle(order)

    for fn in order:
        noisy = np.apply_along_axis(fn, 1, noisy, random_state=random_state.rand())

    noisy = noisy.astype("float32")
    clean = clean.astype("float32")
#     noisy = noisy.reshape(-1, 1, 28, 28).astype("float32")
#     clean = clean.reshape(-1, 1, 28, 28).astype("float32")
    return noisy, clean

if __name__ == "__main__":
    x = _get_dataset()