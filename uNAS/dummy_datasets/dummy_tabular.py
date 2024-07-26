from typing import Tuple
from uNAS.dataset import Dataset
import tensorflow as tf
import numpy as np
import random

class DummyTabular(Dataset):
    """
    DummyTabular Dataset
    ====================
    
    A dataset class for generating DummyTabular data. The dataset consists of tabular data with a specified number of features and classes.
    
    Args:
        num_features (int): The number of features in the dataset.
        num_classes (int): The number of classes in the dataset.
        length (int, optional): The number of samples to generate. If None, the dataset is infinite.
    """

    def __init__(self, num_features, num_classes, length=None):
        self._num_features = num_features
        self._num_classes = num_classes
        self._length = length

        self.x_train, self.y_train = generate_dummy_tabular_data(length, num_features, num_classes)
        self.x_val, self.y_val = generate_dummy_tabular_data(length//10, num_features, num_classes)
        self.x_test, self.y_test = generate_dummy_tabular_data(length//10, num_features, num_classes)


    def train_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))

    def validation_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))

    def test_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def input_shape(self) -> Tuple[int]:
        return (self._num_features,)
    

def generate_dummy_tabular_data(num_samples, num_features, num_classes):
    """
    Generates dummy tabular data for testing purposes.
    
    Args:
        num_samples (int): The number of samples to generate.
        num_features (int): The number of features in the dataset.
        num_classes (int): The number of classes in the dataset.
        
    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing the input and output tensors.
    """

    data = np.zeros((num_samples, num_features))

    # Generate random data
    for i in range(num_features):
        data[:, i] = np.random.choice([1, 2], size=num_samples)

        
    # Generate classes
    scores = np.sum(data, axis=1) 
    
    labels = scores % num_classes
    
    labels = labels.astype(int)


    return data, labels