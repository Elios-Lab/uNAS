from typing import Tuple

import tensorflow as tf
import numpy as np
import random

from .dataset import Dataset

difficulty = 1


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

    def __init__(self, samples_per_second=50, duration=1, length=100):
        self._samples_per_second = samples_per_second
        self._duration = duration
        self._num_classes = 2
        self._length = length
        self._input_shape = (self._samples_per_second * self._duration, 1)

    def _dataset(self, is_training=True):
        x = [tf.convert_to_tensor(generate_sine_wave(freq=generate_random_frequency(i % 2 == 0), sample_rate=self._samples_per_second, phase=0, duration=self._duration)) for i in range(self._length if is_training else self._length // 10)]
        y = [tf.convert_to_tensor(tf.zeros([], dtype=tf.int64) if i % 2 == 0 else tf.ones([], dtype=tf.int64), dtype=tf.int64) for i in range(self._length if is_training else self._length // 10)]

        a = tf.data.Dataset.from_tensor_slices((x, y))
        return a

    def train_dataset(self) -> tf.data.Dataset:
        return self._dataset(is_training=True)

    def validation_dataset(self) -> tf.data.Dataset:
        return self._dataset(is_training=False)

    def test_dataset(self) -> tf.data.Dataset:
        return self._dataset(is_training=False)

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




def generate_random_frequency(even = None):
    # Generate a random frequency according to the difficulty level

    # if difficulty is 0, generate a random frequency between 1Hz and 100Hz (even for class 0, odd for class 1)
    if difficulty == 0:
        # if even is None, generate a random frequency
        if even is None:
            return random.randint(1, 100)
        # if even is True, generate an even frequency
        if even:
            return random.randint(1, 100) * 2
        # if even is False, generate an odd frequency
        return random.randint(1, 100) * 2 + 1
    
    # if difficulty is 1, generate a random frequency with normal distribution, centered at 250 Hz for class 0 and 650 Hz for class 1
    if difficulty == 1:
        if even is None:
            return random.randint(1, 1000)
        if even:
            return int(random.normalvariate(250, 200))
        return int(random.normalvariate(650, 200))