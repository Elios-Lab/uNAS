import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from uNAS.dataset import Dataset
from typing import Tuple

#Parameters
n_samples = 5000  
width = 10       
n_test_samples = 1000  

def datasetManagement():
    X = np.random.randn(n_samples, width).astype(np.float32)  # Input casuale
    y = np.random.uniform(low=-10.0, high=10.0, size=(n_samples,)).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test_samples/n_samples, random_state=42)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, width

class REG_Dataset(Dataset):
    def __init__(self, classes = []):
        self._X_train, self._X_val, self._X_test, self._y_train, self._y_val, self._y_test, self._width = datasetManagement()
        self._input_shape = (self._width, 1)
        self._num_classes = 1 # Regression

    def train_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices((self._X_train, self._y_train))

    def validation_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices((self._X_val, self._y_val))

    def test_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices((self._X_test, self._y_test))

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self._input_shape