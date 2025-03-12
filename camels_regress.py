
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_gradient_magnitude
import torch
import wandb
import time

NUM_PARAMS = 6
INDICES = (0,1,6,7,8)
DATA_DIR = Path('/share/gpu0/asaoulis/cmd/')

class DatasetLoader:
    def __init__(self, dataset_name, sampling_set = 'SB28'):
        self.dataset_name = dataset_name
        self.sampling_set = sampling_set

        self.params, self.data = self.load_data()
    
    def load_data(self):
        if self.dataset_name == "illustris":
            params_file = DATA_DIR / f'params_{self.sampling_set}_IllustrisTNG.txt'
            data_file = DATA_DIR / f'Maps_Mcdm_IllustrisTNG_{self.sampling_set}_z=0.00.npy'
        elif self.dataset_name == "astrid":
            params_file = DATA_DIR / 'params_LH_Astrid.txt'
            data_file = DATA_DIR / 'Maps_Mcdm_Astrid_LH_z=0.00.npy'
        elif self.dataset_name =="nbody":
            params_file = DATA_DIR / f'params_{self.sampling_set}_Nbody_IllustrisTNG.txt'
            data_file = DATA_DIR / f'Maps_Mtot_Nbody_IllustrisTNG_{self.sampling_set}_z=0.00.npy'
        else:
            raise ValueError("Invalid dataset name. Choose 'cheap' or 'expensive'.")
        print('Loading data from', params_file, data_file)
        params = np.loadtxt(params_file)[:, INDICES]
        data = np.load(data_file)
        return params, data
    
    def get_repeated_params(self, factor=15):
        return np.repeat(self.params, factor, axis=0)

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
    np.random.seed(int(time.time()))
    return x[train_ids], y[train_ids], x[test_ids], y[test_ids]



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

import train.compressors as models

def get_scalers(dataset_name):
    dataset = DatasetLoader(dataset_name)
    x_repeated = dataset.get_repeated_params()
    
    param_scaler = DataScaler()
    param_scaler.fit_minmax(x_repeated)
    
    data_scaler = DataScaler()
    data_scaler.fit_standard(np.log(dataset.data))

    return param_scaler, data_scaler

def get_dataset(dataset_name, dataset_size, scaling_dataset=None, N_TEST=2000):
    if scaling_dataset:
        param_scaler, data_scaler = get_scalers(scaling_dataset)
    else:
        param_scaler, data_scaler = get_scalers(dataset_name)

    dataset = DatasetLoader(dataset_name)
    x_repeated = dataset.get_repeated_params()
    
    x_scaled = param_scaler.transform_minmax(x_repeated)
    y_scaled = data_scaler.transform_standard(np.log(dataset.data))
    
    x_train, y_train, x_test, y_test = build_train_test_split(x_scaled, y_scaled, N_TEST)
    
    ids = np.random.permutation(x_train.shape[0])
    train_x_data = torch.tensor(x_train[ids], dtype=torch.float32)[:dataset_size]
    train_y_data = torch.tensor(y_train[ids], dtype=torch.float32)[:dataset_size].unsqueeze(1)
    test_data = (torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
                 torch.tensor(x_test, dtype=torch.float32))
    return train_x_data, train_y_data, test_data


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

from train.lightning_modules import RegressionLightningModule, NDELightningModule, GaussianLightningModule

def train(x, y, testxy, model = None, epochs=100, batch_size=32, lr=0.0001, scheduler_type='cosine', extra_blocks=0, checkpoint_path=None, logger=None):
    train_loader, val_loader, test_loader = prepare_dataloaders(x, y, testxy, batch_size)
    if model is None:
        latent_dim = 128
        # resnet = models.build_resnet(latent_dim, pretrained=True).to(device='cuda')
        # embedding_model = models.build_resnet(latent_dim, pretrained=True).to(device='cuda')
        embedding_model = models.model_o3_err(latent_dim, hidden=12).to(device='cuda')
        # model = GaussianLightningModule(resnet, lr, scheduler_type=scheduler_type, batch_size=batch_size,element_names= ["Omega", "sigma8"])
        model = NDELightningModule(embedding_model, conditioning_dim=latent_dim, lr=lr, scheduler_type=scheduler_type, batch_size=batch_size,
                                        element_names= ["Omega", "sigma8"], test_dataloader = test_loader, optimizer_kwargs = {'weight_decay':0.01, 'betas':(0.9, 0.999)},
                                        checkpoint_path=checkpoint_path, num_extra_blocks=extra_blocks, scheduler_kwargs={'warmup':1000,'gamma':0.98})
    fit_model(model, epochs, logger, train_loader, val_loader)

def fit_model(model, epochs, logger, train_loader, val_loader):
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
        # val_check_interval=17,  # Runs validation every 25 training steps
        logger=pl.loggers.WandbLogger() if logger else None,
        callbacks=[checkpoint_callback, lr_monitor]
    )
    trainer.fit(model, train_loader, val_loader)

def prepare_dataloaders(x, y, testxy, batch_size):
    dataset = AugmentDataset(x, y, train_transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    test_x, test_y = testxy
    test_dataset = TensorDataset(test_x, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=8)
    return train_loader,val_loader,test_loader

def train_model(dataset_name, dataset_size=600, lr=0.00001, epochs=60, batch_size=10, extra_blocks=0, checkpoint_path=None):

    train_x_data, train_y_data, test_data = get_dataset(dataset_name, dataset_size)
    for scheduler_type in ['cyclic']:
        for i in range(1):
            try:
                logger = wandb.init(project="camels-nbody-illustris-cosmo", name=f"{dataset_name}_{scheduler_type}_eb{extra_blocks}_lr{lr}_ds{dataset_size}_{i}", reinit=True)
                trained_model = train(train_y_data, train_x_data, test_data, epochs=epochs, batch_size=batch_size, lr=lr,
                                       logger=logger, scheduler_type=scheduler_type, checkpoint_path=checkpoint_path,
                                       extra_blocks=extra_blocks)
            except Exception as e:
                print(e)

train_model('illustris', dataset_size=25000, lr=0.0005, batch_size=128, epochs=400)

# FINETUNING STAGE 
dataset_size = 600
lr = 0.00002#
# latent_dim = 2
latent_dim = 128
batch_size = 128
# pretrain_checkpoint_path = "/share/gpu0/asaoulis/cmd/checkpoints/run_test/checkpoint-epoch=15-val_log_prob=-3.1550.ckpt" # no wd
# pretrain_checkpoint_path = "/share/gpu0/asaoulis/cmd/checkpoints/run_pretrain_nde_resnset_wd_cyclic_0/checkpoint-epoch=22-val_log_prob=-3.0133.ckpt" # no wd 2
# pretrain_checkpoint_path = "/share/gpu0/asaoulis/cmd/checkpoints/run_nbody_cyclic_lr0.0002_ds13000_1/checkpoint-epoch=195-val_log_prob=-6.4656.ckpt" # nbody
# pretrain_checkpoint_path = "/share/gpu0/asaoulis/cmd/checkpoints/run_astrid_cyclic_eb0_lr0.0002_ds13000_0/checkpoint-epoch=238-val_log_prob=-7.4763.ckpt"
pretrain_checkpoint_path = "/share/gpu0/asaoulis/cmd/checkpoints/run_nbody_cyclic_eb0_lr0.0002_ds25000_0/checkpoint-epoch=323-val_log_prob=-7.2038.ckpt"
# 5 cosmo param
scaling_dataset= 'nbody'
# 'cosine', 'cosine_2mult',
dataset_name = "illustris"
scheduler_type = 'cyclic'
for extra_blocks in [4]:
    for i in range(1):
        train_x, train_y, test_data = get_dataset(dataset_name, dataset_size, scaling_dataset)
        train_loader, val_loader, test_loader = prepare_dataloaders(train_y, train_x, test_data, batch_size)
        # embedding_model = models.build_resnet(latent_dim, pretrained=True).to(device='cuda')
        embedding_model = models.model_o3_err(latent_dim, hidden=12).to(device='cuda')
        model = NDELightningModule(embedding_model, conditioning_dim=latent_dim, lr=lr, scheduler_type=scheduler_type,
                                        element_names= ["Omega", "sigma8"], test_dataloader = test_loader, optimizer_kwargs = {'weight_decay':0.01, 'betas':(0.9, 0.999)},
                                        checkpoint_path=pretrain_checkpoint_path, scheduler_kwargs={'warmup':250, 'gamma':0.96})
        model.append_maf_blocks(train_x, train_y, num_extra_blocks=extra_blocks, bounds=(0,1), device='cuda', init_scale=0.02)
        try:
            logger = wandb.init(project="camels-nbody-illustris-cosmo", name=f"finetune_{dataset_name}_{scaling_dataset}_{scheduler_type}_eb{extra_blocks}_bs{batch_size}_lr{lr}_ds{dataset_size}_{i}", reinit=True)
            trained_model = fit_model(model, 250, logger, train_loader, val_loader)
        except Exception as e:
            import traceback
            traceback.print_exception(e)
