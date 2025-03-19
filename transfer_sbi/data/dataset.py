import numpy as np
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import transforms
import torch
import random

from .data import data_dir, DataScaler, build_train_test_split

class DatasetLoader:
    sampling_set_cosmo_indices = {'LH': (0,1), 'SB28': (0,1,6,7,8)}
    def __init__(self, dataset_name, sampling_set='SB28'):
        self.dataset_name = dataset_name
        self.sampling_set = sampling_set
        self.indices = self.sampling_set_cosmo_indices[sampling_set]
        self.params, self.data = self.load_data()
    
    def load_data(self):
        if self.dataset_name == "illustris":
            params_file = data_dir / f'params_{self.sampling_set}_IllustrisTNG.txt'
            data_file = data_dir / f'Maps_Mcdm_IllustrisTNG_{self.sampling_set}_z=0.00.npy'
        elif self.dataset_name == "astrid":
            params_file = data_dir / 'params_LH_Astrid.txt'
            data_file = data_dir / 'Maps_Mcdm_Astrid_LH_z=0.00.npy'
        elif self.dataset_name == "nbody":
            params_file = data_dir / f'params_{self.sampling_set}_Nbody_IllustrisTNG.txt'
            data_file = data_dir / f'Maps_Mtot_Nbody_IllustrisTNG_{self.sampling_set}_z=0.00.npy'
        else:
            raise ValueError("Invalid dataset name. Choose 'illustris', 'astrid', or 'nbody'.")
        
        print('Loading data from', params_file, data_file)
        params = np.loadtxt(params_file)[:, self.indices]
        data = np.load(data_file)
        return params, data
    
    def get_repeated_params(self, factor=15):
        return np.repeat(self.params, factor, axis=0)

def get_scalers(dataset_name, dataset_suite):
    dataset = DatasetLoader(dataset_name, dataset_suite)
    x_repeated = dataset.params
    
    param_scaler = DataScaler()
    param_scaler.fit_minmax(x_repeated)
    
    data_scaler = DataScaler()
    data_scaler.fit_standard(np.log(dataset.data))
    
    return param_scaler, data_scaler

def get_dataset(dataset_name, dataset_suite, dataset_size, scaling_dataset=None, n_test=2000):
    param_scaler, data_scaler = get_scalers(scaling_dataset or dataset_name, dataset_suite)
    scalers = (param_scaler, data_scaler)
    dataset = DatasetLoader(dataset_name, dataset_suite)
    x_scaled = param_scaler.transform_minmax(dataset.params)
    y_scaled = data_scaler.transform_standard(np.log(dataset.data))
    x_train, y_train ,x_valid, y_valid, x_test, y_test = build_train_test_split(x_scaled, y_scaled, factor=15, n_test= 100)
    # shuffle train data
    idx = np.random.permutation(len(x_train))
    x_train, y_train = x_train[idx], y_train[idx]
    valid_idx = np.random.permutation(len(x_valid))
    x_valid, y_valid = x_valid[valid_idx], y_valid[valid_idx]
    num_train = int(dataset_size * 0.9)
    return torch.tensor(x_train[:num_train], dtype=torch.float32), torch.tensor(y_train[:num_train], dtype=torch.float32).unsqueeze(1), torch.tensor(x_valid[:dataset_size-num_train], dtype=torch.float32), torch.tensor(y_valid[:dataset_size-num_train], dtype=torch.float32).unsqueeze(1), (x_test, y_test), scalers

def prepare_dataloaders(x, y, v_x, v_y, testxy, batch_size):
    train_dataset = AugmentDataset(x, y, train_transform)
    val_dataset = AugmentDataset(v_x, v_y)
    
    test_x, test_y = testxy
    test_dataset = TensorDataset(torch.tensor(test_y, dtype=torch.float32).unsqueeze(1),
                                 torch.tensor(test_x, dtype=torch.float32))
    
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8),
            DataLoader(val_dataset, batch_size=batch_size, num_workers=8),
            DataLoader(test_dataset, batch_size=32, num_workers=8))


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Lambda(lambda img: transforms.functional.rotate(img, random.choice([0, 90, 180, 270]))),
])

class AugmentDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label