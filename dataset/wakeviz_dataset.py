import numpy as np
import tensorflow as tf
import random

from uNAS.dataset import Dataset

class WV_Dataset(Dataset):
    def __init__(self, data_dir, input_shape=(50, 50), fix_seeds=False):
        if fix_seeds:
            np.random.seed(42)
            tf.random.set_seed(42)
            random.seed(42)

        self._input_shape = input_shape
        self._num_classes = 2       
        
        self._data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2)
        ])
        
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                directory=f'{data_dir}/train_quality/',
                labels='inferred',
                label_mode='categorical',
                batch_size=None,
                image_size=input_shape)
        
        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                directory=f'{data_dir}/validation/',
                labels='inferred',
                label_mode='categorical',
                batch_size=None,
                image_size=input_shape)
        
        self.test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                directory=f'{data_dir}/test/',
                labels='inferred',
                label_mode='categorical',
                batch_size=None,
                image_size=input_shape)

    def train_dataset(self):
        ds = self.train_ds.shuffle(1000).map(lambda x, y: (self._data_augmentation(x, training=True), y))
        return ds

    def validation_dataset(self):
        return self.val_ds

    def test_dataset(self):
        return self.test_ds

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def input_shape(self):
        return self._input_shape + (3, )