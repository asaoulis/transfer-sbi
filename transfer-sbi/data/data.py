# data/data.py
import numpy as np
from pathlib import Path

data_dir = Path('/share/gpu0/asaoulis/cmd/')

class DataScaler:
    def __init__(self):
        self.min = None
        self.max = None
        self.mean = None
        self.std = None

    def fit_minmax(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
    
    def transform_minmax(self, X):
        return (X - self.min) / (self.max - self.min)

    def fit_standard(self, X):
        self.mean = X.mean()
        self.std = X.std()
    
    def transform_standard(self, X):
        return (X - self.mean) / self.std

import numpy as np

def build_train_test_split(params, data, factor=15, n_test=100):
    """
    Splits the dataset while ensuring the last `n_test` unique parameters are always in the test set.
    
    Args:
        params (np.ndarray): Array of parameter values (before augmentation).
        data (np.ndarray): Corresponding dataset.
        factor (int): Number of times each parameter is repeated.
        n_test (int): Number of unique parameter sets reserved for testing.

    Returns:
        tuple: Train parameters, Train data, Test parameters, Test data
    """
    # Get unique parameter count
    n_unique = params.shape[0]
    
    if n_test > n_unique:
        raise ValueError(f"Cannot allocate {n_test} test samples from only {n_unique} unique parameters.")

    # Identify train/test indices (last `n_test` unique samples are test)
    test_ids = np.arange(n_unique - n_test, n_unique)
    train_ids = np.arange(0, n_unique - n_test)

    # Repeat parameters accordingly
    train_params = np.repeat(params[train_ids], factor, axis=0)
    test_params = np.repeat(params[test_ids], factor, axis=0)
    
    train_data_ids = np.vstack([train_ids * i for i in range(factor)]).flatten()
    test_data_ids = np.vstack([test_ids * i for i in range(factor)]).flatten()

    train_data = data[train_data_ids]
    test_data = data[test_data_ids]

    return train_params, train_data, test_params, test_data
