from keras.datasets import mnist
import numpy as np
import skimage.util
import random

import skimage.filters
import skimage
import scipy.signal


def noise_img(x):
    noises = [
        {"mode": "s&p", "amount": np.random.uniform(0.1, 0.1)},
        {"mode": "gaussian", "var": np.random.uniform(0.10, 0.15)},
    ]
    # noise = random.choice(noises)
    noise = noises[1]
    return skimage.util.random_noise(x, **noise)


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


def dataset(n=None):
    (x_train, _), (x_test, _) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    if n:
        x = x[:n]
    else:
        n = int(70e3)
    x = x.astype("float32") / 255.
    x = np.reshape(x, (len(x), 28 * 28))

    y = np.apply_along_axis(train_formatting, 1, x)

    clean = y.copy()

    noisy = y.copy()
    # order = [noise_img, blur_img]
    # order = [blur_img]
    order = [noise_img]

    random.shuffle(order)

    for fn in order:
        noisy = np.apply_along_axis(fn, 1, noisy)

    noisy = noisy.astype("float32")
    clean = clean.astype("float32")
#     noisy = noisy.reshape(-1, 1, 28, 28).astype("float32")
#     clean = clean.reshape(-1, 1, 28, 28).astype("float32")
    return noisy, clean
