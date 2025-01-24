import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np

from uNAS.dataset import Dataset
from typing import Tuple

#ATTENZIONE NE PRENDE SOLO 10 PER TEST (take(10))

width = 16000
def datasetManagement():
    # Carica il dataset con train, validation e test
    dataset_name = "speech_commands"

    # Suddivisione dei dati: train, validation e test
    train_data, val_data, test_data = tfds.load(
        dataset_name,
        split=['train', 'validation', 'test'],  # 80% train, 20% validation
        as_supervised=True,  # Per ottenere (audio, label)
        with_info=False
    )

    # Funzione per fare padding alle sequenze fino a una lunghezza di 16000
    def pad_audio_sequence(audio, target_length=width):
        audio = audio.numpy()
        if len(audio) < target_length:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')  # Padding con zeri
        else:
            audio = audio[:target_length]  # Troncamento se più lungo
        return audio

    # Elaborazione dei dati con padding
    # def process_data(data, target_length=width):
    #     data_list = []
    #     for audio, label in data: 
    #         # Aggiungi padding
    #         padded_audio = pad_audio_sequence(audio, target_length)
    #         label_array = label.numpy()
    #         data_list.append({
    #             'audio': padded_audio.tolist(),
    #             'label': label_array
    #         })
    #     # Converti in DataFrame
    #     df = pd.DataFrame(data_list)
    #     x_data = np.array(df["audio"])
    #     x_data = [np.array(x) for x in x_data]
    #     y_data = np.array(df["label"])
    #     return x_data, y_data
    
    def process_data(data, target_length=width):
        x_data, y_data = [], []
        for audio, label in data:
            # Add padding
            padded_audio = pad_audio_sequence(audio, target_length)
            x_data.append(padded_audio)
            y_data.append(label.numpy())
        return np.array(x_data), np.array(y_data)

    # Prepara i dati di train, validation e test
    x_train, y_train = process_data(train_data, target_length=width)
    x_val, y_val = process_data(val_data, target_length=width)
    x_test, y_test = process_data(test_data, target_length=width)

    return x_train, x_val, x_test, y_train, y_val, y_test, width

class SR_Dataset(Dataset):
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