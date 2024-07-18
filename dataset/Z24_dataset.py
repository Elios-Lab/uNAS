import os
import numpy as np
import scipy.io
import tensorflow as tf
import random


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from .dataset import Dataset
from typing import Tuple




timeserie_len = 65536


def load_data(path, classes): 
    class_paths=[]
    for label in classes:
        class_path = os.path.join(path, label, 'avt')
        class_paths.append(class_path)


    #Load data
    dataset_full = []
    labels_full = []
    for class_path in class_paths:
        for path in os.listdir(class_path):
            if path.endswith(".mat"):
                mat = scipy.io.loadmat(os.path.join(class_path, path))
                dataAggregated = mat['data']
                dataAggregated = dataAggregated.T

                for i in range(len(dataAggregated)):
                    #padding to timeserie_len
                    if(len(dataAggregated[i])<timeserie_len):
                        npData=np.array(dataAggregated[i])
                        last_value = npData[-1]
                        additional_values = np.full(timeserie_len - len(npData), last_value)
                        npData = np.concatenate((npData, additional_values))                        
                        dataset_full.append(np.array(npData))
                    else:
                        #add directly to data
                        dataset_full.append(np.array(dataAggregated[i]))
                    #add label
                    labels_full.append(class_paths.index(class_path))

    return dataset_full, labels_full


def shuffle_dataset(dataset, labels):
    # Combine dataset and labels for shuffling
    combined_data = list(zip(dataset, labels))

    # Shuffle the combined data
    shuffled_data = shuffle(combined_data, random_state=42)
    combined_data=None
    # Split the shuffled data back into dataset and labels
    shuffled_dataset, shuffled_labels = zip(*shuffled_data)
    shuffled_dataset = np.array(shuffled_dataset)
    shuffled_labels = np.array(shuffled_labels)
    return shuffled_dataset, shuffled_labels


def split_dataset(combined_data, classes):
    train_data, test_data, val_data = [], [],[]
    for class_label in range(len(classes)):
        class_data, class_labels = zip(*[(data, label) for data, label in combined_data if label == class_label])
        train, test, ytrain, ytest = train_test_split(class_data, class_labels, test_size=1/5, random_state=42)
        train, val, ytrain, yval = train_test_split(train, ytrain, test_size=1/8, random_state=42)
        train_data.extend(zip(train, ytrain))
        test_data.extend(zip(test, ytest))
        val_data.extend(zip(val, yval))

    # Unzip and convert the data into numpy arrays
    X_train_pre, y_train_pre = map(np.array, zip(*train_data))
    X_test_pre, y_test_pre = map(np.array, zip(*test_data))
    X_val_pre, y_val_pre = map(np.array, zip(*val_data))

    X_train_pre = np.array(X_train_pre)
    X_test_pre = np.array(X_test_pre)
    X_val_pre = np.array(X_val_pre)
    y_train_pre = np.array(y_train_pre)
    y_test_pre = np.array(y_test_pre)
    y_val_pre = np.array(y_val_pre)

    return X_train_pre, X_test_pre, X_val_pre, y_train_pre, y_test_pre, y_val_pre


# for windowing (with full length sequence will not change the data)
def create_windows(data_array, label_array, windows_length): #Generate from 65K to N windows of selected length
    X, y = [], []
    for i in range(len(data_array)):
        for start in range(0, len(data_array[i]) - windows_length + 1, windows_length):
            end = start + windows_length
            X.append(data_array[i][start:end])
            y.append(label_array[i])
    return X, y

def datasetManagement(path, classes,windows_length=65536):
    if classes is None:
        raise ValueError("Classes cannot be None.")        

    dataset_full, labels_full = load_data(path, classes)


    shuffled_dataset, shuffled_labels = shuffle_dataset(dataset_full, labels_full)

    # Scale and reshape dataset
    scaler = StandardScaler()
    scaled_dataset = scaler.fit_transform(shuffled_dataset.reshape(-1, timeserie_len)).reshape(shuffled_dataset.shape)

    combined_data = list(zip(scaled_dataset, shuffled_labels))
    
    X_train_pre, X_test_pre, X_val_pre, y_train_pre, y_test_pre, y_val_pre = split_dataset(combined_data, classes)


        
    X_train, y_train = create_windows(X_train_pre, y_train_pre, windows_length)
    X_val, y_val = create_windows(X_val_pre, y_val_pre, windows_length)
    X_test, y_test = create_windows(X_test_pre, y_test_pre, windows_length)
    
    width = len(X_train[0])
 
    X_train = np.array([np.array(x) for x in X_train])
    X_val = np.array([np.array(x) for x in X_val])
    X_test = np.array([np.array(x) for x in X_test])
    
    # Reshape the input data to have a third dimension
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) 
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)     
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    return X_train, X_val, X_test, y_train, y_val, y_test, width


'''
#Insert here the classes you want to use for the training, the first element will be labeled as 0, the second as 1 and so on
#values from 01 to 17
#classes = ['01', '03', '04', '05', '06','07','09','10','11','12','13','14','15','16','17']
classes = ['01', '03', '04', '05', '06']
'''


class Z24_Dataset(Dataset):
    def __init__(self, classes = ['01', '03', '04', '05', '06'], windows_length=8192, path= 'C:\\Dottorato\\Z24\\DatasetPDT', fix_seeds = False):
        if fix_seeds:
            np.random.seed(42)
            tf.random.set_seed(42)
            random.seed(42)
            
        self._classes = classes
        self._windows_length = windows_length
        self._X_train, self._X_val, self._X_test, self._y_train, self._y_val, self._y_test, self._width = datasetManagement(path, classes, windows_length)
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