import os
import pickle

import numpy as np
import pandas as pd

from .dataset import Dataset


class RealDataset(Dataset):
    def __init__(self, raw_path, **kwargs):
        super().__init__(**kwargs)
        self.raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/raw/", raw_path)

class RealPickledDataset(Dataset):
    """Class for pickled datasets from https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection"""

    def __init__(self, name, training_path):
        self.name = name
        self.training_path = training_path
        self.test_path = self.training_path.replace("train", "test")
        self._data = None

    def data(self):
        if self._data is None:
            with open(self.training_path, 'rb') as f:
                X_train = pd.DataFrame(pickle.load(f))
            #if('space_shuttle' in self.training_path):
            #    X_train = X_train.groupby(X_train.index // 3).mean()
            X_train = X_train.iloc[:, :-1]
            #minimum, maximum = np.min(X_train),np.max(X_train)
            mean, std = X_train.mean(), X_train.std()
            X_train = (X_train - mean) / std
            #X_train = (X_train - minimum) / (maximum - minimum)
            with open(self.test_path, 'rb') as f:
                X_test = pd.DataFrame(pickle.load(f))
            #if('space_shuttle' in self.training_path):
            #    X_test = X_test.groupby(X_test.index // 3).mean()
            y_test = X_test.iloc[:, -1]
            y_test = y_test.apply(np.ceil)
            X_test = X_test.iloc[:, :-1]
            X_test = (X_test - mean) / std
            #X_test = (X_test - minimum)/(maximum - minimum)
            self._data = X_train, np.zeros(len(X_train)), X_test, y_test
        return self._data

class Combine_space_shuttle_data:
    def space_shuttle_data_loader(sequence_length, step):
        print("Combining the datasets TEK14,TEK16,TEK17")
        X_train_c = pd.DataFrame()
        for data_set_path in glob.glob('./*'):
            with open(data_set_path,'rb') as f:
                X_train = pd.DataFrame(pickle.load(f))
                ind = [i for i in range(0, X_train.shape[0] - sequence_length +1, step)]
                print(X_train.shape)
                print(len(ind))
                ind = ind[-1]
                ind = ind + sequence_length
                X_train = X_train.iloc[0:ind]
                X_train_c = X_train_c.append(X_train)
                print(X_train_c.shape)
                X_train_c.to_pickle(f'TEK_14_16_17_seq_{sequence_length}_step_{step}.pkl')

class RealCSVDataset(Dataset):
    """Class for pickled datasets from https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection"""

    def __init__(self, name, training_path):
        self.name = name
        self.training_path = training_path
        self.test_path = self.training_path.replace("train", "test")
        self._data = None

    def data(self):
        if self._data is None:
            with open(self.training_path, 'rb') as f:
                X_train = pd.read_csv(f,names=['Feature'],index_col=False)
            X_train = X_train.iloc[:, :]
            minimum, maximum = np.min(X_train),np.max(X_train)
            #mean, std = X_train.mean(), X_train.std()
            #X_train = (X_train - mean) / std
            X_train = (X_train - minimum) / (maximum - minimum)


            with open(self.test_path, 'rb') as f:
                X_test = pd.read_csv(f,names=['Feature','outlier'],index_col=False)
            y_test = X_test.iloc[:, -1]
            X_test = X_test.iloc[:, :-1]
            #X_test = (X_test - mean) / std
            X_test = (X_test - minimum)/(maximum - minimum)
            self._data = X_train, np.zeros(len(X_train)), X_test, y_test
        return self._data
