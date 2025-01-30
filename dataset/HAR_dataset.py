import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from uNAS.dataset import Dataset
from typing import Tuple

dataset_path= rf'/mnt/c/Users/Matteo/Desktop/Dottorato/uNAS_HAR/'

width=561
def datasetManagement():
    def load_data():
        X_train = np.loadtxt(dataset_path+'Train/X_train.txt')
        y_train = np.loadtxt(dataset_path+'Train/y_train.txt')
        X_test = np.loadtxt(dataset_path+'Test/X_test.txt')
        y_test = np.loadtxt(dataset_path+'Test/y_test.txt')
        return X_train, y_train, X_test, y_test

    x_train, y_train, x_test, y_test = load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)
    y_train = y_train.astype(int) - 1
    y_val = y_val.astype(int) - 1
    y_test = y_test.astype(int) - 1
    return x_train, x_val, x_test, y_train, y_val, y_test, width

class HAR_Dataset(Dataset):
    def __init__(self, classes = [0,1,2,3,4,5,6,7,8,9,10,11]):
        self._X_train, self._X_val, self._X_test, self._y_train, self._y_val, self._y_test, self._width = datasetManagement()
        self._input_shape = (self._width, 1)
        self._num_classes = len(classes)



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