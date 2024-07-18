from typing import Tuple

import tensorflow as tf
import numpy as np
import random

from .dataset import Dataset



class DummyWaveform(Dataset):
    """
    A dataset class for generating dummy waveform data. There are two classes, sine wave and square wave (respectively, label 0 and 1), with a random phase.

    Args:
        samples_per_second (int): The number of samples per second.
        duration (int): The duration of each waveform in seconds.
        length (int): The number of waveforms to generate.

    Attributes:
        _samples_per_second (int): The number of samples per second.
        _duration (int): The duration of each waveform in seconds.
        _num_classes (int): The number of classes in the dataset.
        _length (int): The number of waveforms to generate.
        _input_shape (Tuple[int, int]): The shape of the input data.

    Methods:
        _dataset(): Generates the dataset of waveforms.
        train_dataset(): Returns the training dataset.
        validation_dataset(): Returns the validation dataset.
        test_dataset(): Returns the test dataset.
        num_classes(): Returns the number of classes in the dataset.
        input_shape(): Returns the shape of the input data.
    """

    def __init__(self, samples_per_second=50, duration=1, length=100, difficulty=1, num_classes=2):
        self._samples_per_second = samples_per_second
        self._duration = duration
        self._num_classes = num_classes
        self._length = length
        self._input_shape = (self._samples_per_second * self._duration, 1)
        self._difficulty = difficulty

        self._classes_mean = [20, 60, 100, 140, 180, 220]
        self._classes_variance = 10 * np.ones(len(self._classes_mean))

        self._generate_dataset()

    def _generate_dataset(self):

        x_train = []
        y_train = []

        x_val = []
        y_val = []

        x_test = []
        y_test = []

        for i in range(self.num_classes):
            x_train += [tf.convert_to_tensor(generate_sine_wave(freq=generate_random_frequency(self._difficulty, i, self._classes_mean, self._classes_variance), sample_rate=self._samples_per_second, phase=0, duration=self._duration)) for _ in range(self._length)]
            y_train += [tf.convert_to_tensor(tf.zeros([], dtype=tf.int64) + i, dtype=tf.int64) for _ in range(self._length)]

            x_val += [tf.convert_to_tensor(generate_sine_wave(freq=generate_random_frequency(self._difficulty, i, self._classes_mean, self._classes_variance), sample_rate=self._samples_per_second, phase=0, duration=self._duration)) for _ in range(self._length//10)]
            y_val += [tf.convert_to_tensor(tf.zeros([], dtype=tf.int64) + i, dtype=tf.int64) for _ in range(self._length//10)]

            x_test += [tf.convert_to_tensor(generate_sine_wave(freq=generate_random_frequency(self._difficulty, i, self._classes_mean, self._classes_variance), sample_rate=self._samples_per_second, phase=0, duration=self._duration)) for _ in range(self._length//10)]
            y_test += [tf.convert_to_tensor(tf.zeros([], dtype=tf.int64) + i, dtype=tf.int64) for _ in range(self._length//10)]

        self.x_train = tf.stack(x_train)
        self.y_train = tf.stack(y_train)

        self.x_val = tf.stack(x_val)
        self.y_val = tf.stack(y_val)

        self.x_test = tf.stack(x_test)
        self.y_test = tf.stack(y_test)

    def train_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))

    def validation_dataset(self) -> tf.data.Dataset:
        return  tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))

    def test_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self._input_shape



def generate_sine_wave(freq, sample_rate, phase, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    x = x + phase
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = 1 * np.sin(2 * np.pi * frequencies)
    return y

def generate_square_wave(freq, sample_rate, phase, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    x = x + phase
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = 1 * np.sign(np.sin(2 * np.pi * frequencies))
    return y




def generate_random_frequency(difficulty = 1, class_label = 0, classes_mean = [250, 650], classes_variance = [130, 130]):
    # Generate a random frequency according to the difficulty level

    # use the mean and variance of the classes

    if difficulty == 0:
        if class_label > 1:
            raise ValueError("Class label must be 0 or 1 for difficulty 0")
        return random.randint(1, 100) if class_label == 0 else random.randint(1, 100) * 2 + 1
    
    if difficulty == 1:
        if class_label > len(classes_mean) - 1 :
            raise ValueError("Class label is beyond the number of classes for difficulty 1")
        return max(1, int(random.normalvariate(classes_mean[class_label], classes_variance[class_label])))

    