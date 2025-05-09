from typing import Tuple

import tensorflow as tf

from uNAS.dataset import Dataset


class Dummy2D(Dataset):
    """
    Dummy2D Dataset
    ===============
    A dataset class for generating Dummy2D image data. The dataset consists of images with a specified shape and a specified number of classes.

    Args:
        img_shape (Tuple[int, int, int]): The shape of the images (height, width, channels).
        num_classes (int): The number of classes in the dataset.
        length (int, optional): The number of images to generate. If None, the dataset is infinite.

    Attributes:
        _img_shape (Tuple[int, int, int]): The shape of the images.
        _num_classes (int): The number of classes in the dataset.
        _length (int): The number of images to generate.

    Methods:
        _dataset(): Generates the dataset of images.
        train_dataset(): Returns the training dataset.
        validation_dataset(): Returns the validation dataset.
        test_dataset(): Returns the test dataset.
        num_classes(): Returns the number of classes in the dataset.
        input_shape(): Returns the shape of the input data.
    """

    def __init__(self, img_shape, num_classes, length=None):
        self._img_shape = img_shape
        self._num_classes = num_classes
        self._length = length

    def _dataset(self):
        x = tf.zeros(self._img_shape, dtype=tf.float32)
        y = tf.zeros([], dtype=tf.int64)
        return tf.data.Dataset.from_tensors((x, y)).repeat(count=self._length)

    def train_dataset(self) -> tf.data.Dataset:
        return self._dataset()

    def validation_dataset(self) -> tf.data.Dataset:
        return self._dataset()

    def test_dataset(self) -> tf.data.Dataset:
        return self._dataset()

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._img_shape
