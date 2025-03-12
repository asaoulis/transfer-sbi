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

def build_train_test_split(x, y, n_test):
    np.random.seed(0)
    test_ids = np.random.choice(x.shape[0], n_test, replace=False)
    train_ids = np.array([i for i in range(x.shape[0]) if i not in test_ids])
    np.random.seed()
    return x[train_ids], y[train_ids], x[test_ids], y[test_ids]
