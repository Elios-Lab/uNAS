import numpy as np
import tensorflow as tf
import random

# from datasets import load_dataset
from uNAS.dataset import Dataset

class WV_Dataset_Serialized(Dataset):
    def __init__(self, cache_dir='/ssd/.data_cache/huggingface/datasets/', input_shape=(50, 50), fix_seeds=False):
        if fix_seeds:
            np.random.seed(42)
            tf.random.set_seed(42)
            random.seed(42)

        self._input_shape = input_shape
        self._num_classes = 2
        self._wake_vision = None # load_dataset("Harvard-Edge/Wake-Vision", cache_dir=cache_dir)
        self._data_preprocessing = tf.keras.Sequential([
            tf.keras.layers.Resizing(self._input_shape[0], self._input_shape[1])
        ])
        self._data_augmentation = tf.keras.Sequential([
            self._data_preprocessing,
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2)
        ])

    def train_dataset(self):
        ds = self._wake_vision['train_quality'].to_tf_dataset(columns='image', label_cols='person')
        ds = ds.shuffle(1000).map(lambda x, y: (self._data_augmentation(x, training=True), y))
        return ds.prefetch(tf.data.AUTOTUNE)

    def validation_dataset(self):
        ds = self._wake_vision['validation'].to_tf_dataset(columns='image', label_cols='person')
        ds = ds.map(lambda x, y: (self._data_preprocessing(x, training=True), y))
        return ds.prefetch(tf.data.AUTOTUNE)

    def test_dataset(self):
        ds = self._wake_vision['test'].to_tf_dataset(columns='image', label_cols='person')
        ds = ds.map(lambda x, y: (self._data_preprocessing(x, training=True), y))
        return ds.prefetch(tf.data.AUTOTUNE)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def input_shape(self):
        return self._input_shape + (3, )