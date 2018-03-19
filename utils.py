import tensorflow as tf
import os
import glob

from skimage.color import gray2rgb
from skimage.io import imread, imsave
from skimage.transform import resize

import scipy
import matplotlib.pyplot as plt
import pickle

import numpy as np
import math

def sample_code(batch_size, z_size, uniform=False):
    if uniform:
        return np.random.uniform(size=(BATCH_SIZE, z_size))

    x = np.random.multivariate_normal(mean=np.zeros(z_size),cov=np.identity(z_size),size=batch_size)
    row_sums = np.linalg.norm(x, axis=1)
    x = x / row_sums[:, np.newaxis]

    return x

def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        return data

def load_MNIST():
    mnist_data = tf.contrib.learn.datasets.mnist.load_mnist(train_dir='MNIST-data')
    train_data = mnist_data.train.images
    test_data = mnist_data.test.images
    train_data = np.reshape(train_data, (np.shape(train_data)[0], 28, 28, 1))
    test_data = np.reshape(test_data, (np.shape(test_data)[0], 28, 28, 1))
    train_labels = mnist_data.train.labels
    test_labels = mnist_data.test.labels

    return train_data, train_labels, test_data, test_labels
