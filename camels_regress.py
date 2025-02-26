
# %%

import numpy as np
from pathlib import Path
# name of the file
data_dir = Path('/share/gpu0/asaoulis/cmd/')
IllustrisLH = data_dir / 'params_SB28_IllustrisTNG.txt'
# read the data
IllustrisLHfparams = np.loadtxt(IllustrisLH)[:, :2]

nbodyLH = data_dir / 'params_SB28_Nbody_IllustrisTNG.txt'
nbodyLHfparams = np.loadtxt(nbodyLH)[:, :2]


# %%
# load the data
IllustrisLH= data_dir / 'Maps_Mcdm_IllustrisTNG_SB28_z=0.00.npy'
illustris_data = np.load(IllustrisLH)

nbodyLH= data_dir / 'Maps_Mtot_Nbody_IllustrisTNG_SB28_z=0.00.npy'
nbody_data = np.load(nbodyLH)

# %%
illustris_data.shape

# %%
# repeat the params for the number of maps
factor = 15
IllustrisLHfparams_repeated = np.repeat(IllustrisLHfparams, factor, axis=0)
nbodyLHfparams_repeated = np.repeat(nbodyLHfparams, factor, axis=0)


class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None
    def fit(self, X, axis=None):
        if axis is not None:
            self.min = X.min(axis=axis)
            self.max = X.max(axis=axis)
        else:
            self.min = X.min()
            self.max = X.max()
    def transform(self, X):
        return (X - self.min) / (self.max - self.min)
    def inverse_transform(self, X):
        return X * (self.max - self.min) + self.min
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X, axis=None):
        """Compute the mean and standard deviation for scaling."""
        if axis is not None:
            self.mean = X.mean(axis=axis)
            self.std = X.std(axis=axis)
        else:
            self.mean = X.mean()
            self.std = X.std()

    def transform(self, X):
        """Scale the input data using the computed mean and std deviation."""
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        """Reverse the transformation back to the original scale."""
        return X * self.std + self.mean

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_gradient_magnitude

def create_dummy_variables(x, y):
    # mean_val = y.mean(axis=(1, 2))
    # var_val = y.var(axis=(1, 2))
    # skew_val = np.array([skew(field.flatten()) for field in y])
    kurt_val = np.array([kurtosis(field.flatten()) for field in y])
    
    fft_power = np.abs(fftshift(fft2(y), axes=(1, 2)))**2
    high_freq_power = fft_power[:, 120:136, 120:136].mean(axis=(1, 2))
    
    grad_mag_mean = np.array([gaussian_gradient_magnitude(field, sigma=1).mean() for field in y])

    x = np.hstack([x,
                   kurt_val[:, None], high_freq_power[:, None], grad_mag_mean[:, None]])
    
    return x, y

nbody_x_repeated = nbodyLHfparams_repeated
illustris_x_repeated = IllustrisLHfparams_repeated
param_scaler = MinMaxScaler()
param_scaler.fit(illustris_x_repeated[:, :2], axis=0)
expensive_x = param_scaler.transform(illustris_x_repeated[:, :2])
cheap_x = param_scaler.transform(nbody_x_repeated[:, :2])

exp_data_scaler = StandardScaler()
exp_data_scaler.fit(np.log(illustris_data))
expensive_y = exp_data_scaler.transform(np.log(illustris_data))

cheap_data_scaler = StandardScaler()
cheap_data_scaler.fit(np.log(nbody_data))
cheap_y = cheap_data_scaler.transform(np.log(nbody_data))


def build_exp_train_test_split(expensive_x, expensive_y, N_TEST):
    test_ids = np.random.choice(expensive_x.shape[0], N_TEST, replace=False)
    train_ids = np.array([i for i in range(expensive_x.shape[0]) if i not in test_ids])
    return expensive_x[train_ids], expensive_y[train_ids], expensive_x[test_ids], expensive_y[test_ids]
np.random.seed(0)
expensive_x, expensive_y, expensive_test_x, expensive_test_y = build_exp_train_test_split(expensive_x, expensive_y, 2000)

cheap_x, cheap_y, cheap_test_x, cheap_test_y = build_exp_train_test_split(cheap_x, cheap_y, 2000)


import os
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import wandb
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import transforms

import compressors as models

# Define the augmentations
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

from lightning_modules import RegressionLightningModule, NDELightningModule, GaussianLightningModule

def train(x, y, testxy, epochs=100, batch_size=32, lr=0.0001, scheduler_type='cosine', logger=None):
    dataset = AugmentDataset(x, y, train_transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    test_x, test_y = testxy
    test_dataset = TensorDataset(test_x, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    latent_dim = 2
    # resnet = models.build_resnet(latent_dim, pretrained=True).to(device='cuda')
    resnet = models.model_o3_err(latent_dim, hidden=12, predict_sigmas=True).to(device='cuda')
    model = GaussianLightningModule(resnet, lr, scheduler_type=scheduler_type, batch_size=batch_size,element_names= ["Omega", "sigma8"])
    # model = NDELightningModule(resnet, conditioning_dim=latent_dim, lr=lr, scheduler_type=scheduler_type, batch_size=batch_size,
    #                                 element_names= ["Omega", "sigma8"], test_dataloader = test_loader, optimizer_kwargs = {'weight_decay':0.01})
    monitor_string =f"val_{model.loss_name}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=f"{monitor_string}",
        dirpath=f"/share/gpu0/asaoulis/cmd/checkpoints/run_{wandb.run.name}",
        filename=f"checkpoint-{{epoch:02d}}-{{{monitor_string}:.4f}}",
        save_top_k=10,
        mode="min",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')  # or 'epoch'
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=10,  # Logs every 10 steps
        check_val_every_n_epoch=1,  # Ensures validation runs in every epoch
        val_check_interval=50,  # Runs validation every 25 training steps
        logger=pl.loggers.WandbLogger() if logger else None,
        callbacks=[checkpoint_callback, lr_monitor]
    )
    trainer.fit(model, train_loader, val_loader)

# Run training
dataset_size = 28000
lr = 0.0045
# 'cosine', 'cosine_2mult',
for scheduler_type in [ 'exp']: 
    for i in range(3):
        ids = np.random.permutation(cheap_x.shape[0])
        expensive_train_x_data = torch.tensor(cheap_x[ids], dtype=torch.float32)[:dataset_size, :2]
        expensive_train_y_data = torch.tensor(cheap_y[ids], dtype=torch.float32)[:dataset_size].unsqueeze(1)
        testxy = (
            torch.tensor(cheap_test_y, dtype=torch.float32).unsqueeze(1),
            torch.tensor(cheap_test_x[:, :2], dtype=torch.float32)
        )

        # resnet = model_o3_err(2, hidden=12).to(device='cuda')
        try:
            logger = wandb.init(project="camels-nbody-illustris-improved", name=f"gaussian_npe_{scheduler_type}_lr{lr}_{i}", reinit=True)
            trained_model = train(expensive_train_y_data, expensive_train_x_data, testxy, epochs=200, batch_size=128, lr=lr, logger=logger, scheduler_type=scheduler_type)
        except Exception as e:
            print(e)


