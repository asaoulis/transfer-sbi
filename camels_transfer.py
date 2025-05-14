
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

# %%
import matplotlib.pyplot as plt
objs = {'IllustrisTNG': (illustris_data, IllustrisLHfparams), 'Nbody': (nbody_data, nbodyLHfparams)}
for name, (data, params) in objs.items():
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        for j in range(3):
            params = IllustrisLHfparams[i*3+j][:2]
            ax[i,j].set_title(f"$\Omega_m = {params[0]}, \sigma_8 = {params[1]}$")
            ax[i, j].imshow(np.log(data[i*3+j]), cmap='magma')
            ax[i, j].axis('off')
    plt.suptitle(name)
            
    plt.show()


# %%
from train.models import MAFPretrainFineTune, TrainConfig

# %%
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

# nbody_x_repeated, _ = create_dummy_variables(nbodyLHfparams_repeated[:10000], np.log10(nbody_data[:10000]))
# illustris_x_repeated, _ = create_dummy_variables(IllustrisLHfparams_repeated[:10000], np.log10(illustris_data[:10000]))

nbody_x_repeated, _ = create_dummy_variables(nbodyLHfparams_repeated, np.log10(nbody_data))
illustris_x_repeated, _ = create_dummy_variables(IllustrisLHfparams_repeated, np.log10(illustris_data))


param_scaler = MinMaxScaler()
param_scaler.fit(illustris_x_repeated[:, :4], axis=0)
expensive_x = param_scaler.transform(illustris_x_repeated[:, :4])
cheap_x = param_scaler.transform(nbody_x_repeated[:, :4])

exp_data_scaler = MinMaxScaler()
exp_data_scaler.fit(np.log10(illustris_data))
expensive_y = exp_data_scaler.transform(np.log10(illustris_data))

cheap_data_scaler = MinMaxScaler()
cheap_data_scaler.fit(np.log10(nbody_data))
cheap_y = cheap_data_scaler.transform(np.log10(nbody_data))
# expensive_y = cheap_data_scaler.transform(np.log10(illustris_data))


# %%
# exp_data_scaler.fit(np.log10(illustris_data))
# expensive_y = exp_data_scaler.transform(np.log10(illustris_data))
# cheap_y = cheap_data_scaler.transform(np.log10(nbody_data))

def build_exp_train_test_split(expensive_x, expensive_y, N_TEST):
    test_ids = np.random.choice(expensive_x.shape[0], N_TEST, replace=False)
    train_ids = np.array([i for i in range(expensive_x.shape[0]) if i not in test_ids])
    return expensive_x[train_ids], expensive_y[train_ids], expensive_x[test_ids], expensive_y[test_ids]
np.random.seed(0)
expensive_x, expensive_y, expensive_test_x, expensive_test_y = build_exp_train_test_split(expensive_x, expensive_y, 2000)

cheap_x, cheap_y, cheap_test_x, cheap_test_y = build_exp_train_test_split(cheap_x, cheap_y, 2000)


# %%
import torch
import torch.nn as nn
from torchvision import models

def build_resnet(num_outputs, pretrained=True):
    
    resnet = models.resnet18(pretrained=pretrained)


    # Copy weights from the original layer
    original_weights = resnet.conv1.weight.data

    # Average the weights across the RGB channels
    new_weights = original_weights.mean(dim=1, keepdim=True)

    # Replace the conv1 layer and assign the new weights
    resnet.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=resnet.conv1.out_channels,
        kernel_size=resnet.conv1.kernel_size,
        stride=resnet.conv1.stride,
        padding=resnet.conv1.padding,
        bias=resnet.conv1.bias is not None
    )
    resnet.conv1.weight.data = new_weights

    # add two fc layers
    resnet.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_outputs),

    )
    return resnet

def build_convnext(num_outputs, pretrained=True):
    convnext = models.convnext_tiny(pretrained=pretrained)
    
    # Get the original first convolution layer
    original_conv = convnext.features[0][0]  # First conv layer in ConvNeXt
    original_weights = original_conv.weight.data
    
    # Average the weights across RGB channels
    new_weights = original_weights.mean(dim=1, keepdim=True)
    
    # Replace first conv layer with single-channel input
    convnext.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )
    convnext.features[0][0].weight.data = new_weights
    
    # Modify classifier head
    in_features = convnext.classifier[2].in_features
    convnext.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, num_outputs),
    )
    
    return convnext
# %%
from train.models import TrainConfig
from typing import NamedTuple

class TrainConfig(NamedTuple):

    flow_type : str
    lr : float
    finetune_lr : float
    num_initial_blocks : int
    num_extra_blocks : int
    optimizer: torch.optim.Optimizer = None 
    resblocks: bool = False
    size: int = 400
    pretrain_size : int = 10000
    pretrain_wd : float = 0.01
    only_finetune_extra_blocks : bool = False
    conditioning_dim : int = 4
    # create a unique wandb name
    def wandb_name(self):
        return f"{self.flow_type}_{self.pretrain_size}pds_{self.only_finetune_extra_blocks}only_{self.size}ds_{self.num_initial_blocks}ib_{self.num_extra_blocks}eb_{self.lr}lr_{self.resblocks}res_{self.pretrain_wd}wd"

pretrained_state_dicts = {}
training_config = TrainConfig(flow_type='nsf', lr=0.0003, finetune_lr=0, num_initial_blocks=2, num_extra_blocks=0, size=600, pretrain_size=10000, pretrain_wd=0.01, resblocks=False)
# shuffle the data


# %%
idx = np.random.permutation(cheap_x.shape[0])
cheap_x = cheap_x[idx]
cheap_y = cheap_y[idx]
resnet = build_resnet(10)
cheap_x_pretrain = torch.tensor(cheap_x[:training_config.pretrain_size], dtype=torch.float32)
cheap_y_pretrain = torch.tensor(cheap_y[:training_config.pretrain_size], dtype=torch.float32)

# %%
cheap_y_pretrain.unsqueeze(1).shape

# %%
import wandb
from copy import deepcopy
from train.models import prep_test_dataloader

model_builder = {'convnext': build_convnext, 'resnet': build_resnet}
confs = [('resnet', 0.00001, 256)]
resnet_sizes  = [128]
use_scheduler = True
test_dataloader = prep_test_dataloader( expensive_test_x, expensive_test_y[:, np.newaxis], batch_size=128) 
for resnet_size in resnet_sizes:
    for (model_type, lr, bs) in confs:
        training_config = TrainConfig(flow_type='nsf', conditioning_dim=resnet_size, lr=lr, finetune_lr=0, num_initial_blocks=4, num_extra_blocks=0, size=600, pretrain_size=28000, pretrain_wd=0.01, resblocks=False)

        add_string = f"{model_type}_lat{resnet_size}_nopre"
        idx = np.random.permutation(expensive_x.shape[0])
        cheap_x_data = expensive_x[idx]
        cheap_y_data = expensive_y[idx]
        resnet = model_builder[model_type](resnet_size, pretrained=False)
        cheap_x_pretrain = torch.tensor(cheap_x_data[:training_config.pretrain_size], dtype=torch.float32)
        cheap_y_pretrain = torch.tensor(cheap_y_data[:training_config.pretrain_size], dtype=torch.float32)
        maf = MAFPretrainFineTune(training_config, embedding_net=resnet, device='cuda')
        try:
            logger = wandb.init(project="camels-nbody-illustris-transfer", name=f'accurate_{add_string}_{training_config.pretrain_size}' , reinit=True)
            print('Starting training...', flush=True)
            state_dict = maf.pretrain( cheap_x_pretrain, cheap_y_pretrain.unsqueeze(1), lr=training_config.lr, test_dataloader=test_dataloader, logger=logger, scheduler=use_scheduler, batch_size=bs)

        except Exception as e:
            import traceback
            print(traceback.format_exc())
        finally:
            # wandb.finish()
            pass
