import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from uNAS.dataset import Dataset
from typing import Tuple

width=21
def datasetManagement():
    #https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
    from ucimlrepo import fetch_ucirepo
    # Fetch dataset
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

    # Data (as pandas dataframes)
    X = cdc_diabetes_health_indicators.data.features.to_numpy()  # Convert features to NumPy array
    y = cdc_diabetes_health_indicators.data.targets.to_numpy()
    
    # Split the dataset into train (80%) and temp (20%) for validation and test
    x_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Further split temp into validation (50% of temp) and test (50% of temp)
    x_val, x_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    y_train = y_train.flatten()
    y_val = y_val.flatten()
    y_test = y_test.flatten()
    return x_train, x_val, x_test, y_train, y_val, y_test, width

class DIA_Dataset(Dataset):
    def __init__(self, classes = [0,1]):
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