from transfer_sbi.nbodykit.correlation import pk_to_xi, xi_to_pk
from scipy.interpolate import interp1d
from nbodykit.lab import  LinearMesh
from classy import Class

import numpy as np
cosmo_class = Class()
k0 = 1.e-4 # in h/Mpc
hubble =  0.6711
omega_b = 0.049
omega_cdm = 0.3
ns =  0.9624
sigma8 =  0.834
parameters = [omega_b*hubble**2, omega_cdm*hubble**2, hubble, ns, sigma8]
DEFAULT_COSMOLOGY = {'omega_b': parameters[0],
                'omega_cdm': parameters[1],
                'h': parameters[2],
                'n_s': parameters[3],
                'sigma8': parameters[4],
                'output': 'mPk',
                #'c_min': parameters[5],
                'non linear': 'hmcode',
                'z_max_pk': 50,
                'P_k_max_h/Mpc': 30,
                #'Omega_k': 0.,
                #'N_ncdm' : 0,
                #'N_eff' : 3.046,
            }

from nbodykit.lab import  ArrayMesh, FFTPower

def calc_ps_from_field_nbodykit(delta, BoxSize=[25, 25], kmin=1.e-1, kmax=20, dk=7e-3, already_overdensity=False):
    if already_overdensity:
        overdensity = delta
    else:
        overdensity = delta/np.mean(delta)
        overdensity -= 1.0 
    field_mesh = ArrayMesh(overdensity, BoxSize=BoxSize)
    r_2d = FFTPower(field_mesh, mode='1d', kmin=kmin, kmax=kmax)#, dk=dk)
    return r_2d.power['k'], r_2d.power['power'].real

def create_log_normal_matter_density_from_field(field, cosmo_parameters = {}, BoxSize=25, kmin=0.1, kmax=20):
    k_values_, power_spectrum = extract_power_spectrum_from_field(field, cosmo_parameters, BoxSize, kmin, kmax)
    field_mean = np.mean(field)
    matter_density = sample_matter_density_from_spectrum(k_values_, power_spectrum, field_mean)
    return matter_density

def sample_matter_density_from_spectrum(k_values_, power_spectrum, field_mean):
    cf_class = pk_to_xi(k_values_, power_spectrum, extrap=True)
    r = np.logspace(-3,3, int(1e5))

    def cf_g(r):
        return np.log(1+cf_class(r))

    # then it should be easy to use the same transformation as above, just inverse, to obtain a Gaussian-like power spectrum
    Pclass_g = xi_to_pk(r, cf_g(r), extrap=True)
    g_field = LinearMesh(Pclass_g, Nmesh=[256, 256], BoxSize=25., unitary_amplitude=True ).preview() - 1 
    gaussian_stddev = np.std(g_field.flatten())
    ln_field_mixed = np.exp(g_field-gaussian_stddev**2/2)
    matter_density = field_mean * (ln_field_mixed )
    return np.log10(matter_density)

def compute_theory_power_spectrum(cosmo_parameters, kmin=k0, kmax=20):
    merged_cosmology = {**DEFAULT_COSMOLOGY, **cosmo_parameters}
    cosmo_class.set(merged_cosmology)
    cosmo_class.compute()
    # k_values, ps_values = compute_power_spectrum(nbody_data.data[400])
    k_values_theory = np.logspace(np.log10(kmin*hubble), np.log10(kmax*hubble)-1e-10, 500) # these are in 1/Mpc
    power_spectrum_theory = np.empty(500)
    for ind in range(500):
        power_spectrum_theory[ind] = cosmo_class.pk(k_values_theory[ind], 0)
    k_values_theory /= hubble  # so that it is in h/Mpc
    power_spectrum_theory *= hubble**3  # so that it is in Mpc/h**3
    return k_values_theory, power_spectrum_theory

def extract_power_spectrum_from_field(field, cosmo_parameters, BoxSize=25, kmin=0.1, kmax=20):
    k_values, ps_values = calc_ps_from_field_nbodykit(field, BoxSize=BoxSize, kmin=kmin, kmax=kmax)
    k_values, ps_values = k_values[10:-10], ps_values[10:-10]

    k_values_theory, power_spectrum_theory = compute_theory_power_spectrum(cosmo_parameters, kmin=k0, kmax=k_values[0])

    k_values_interp = np.concatenate([k_values_theory, k_values])
    ps_values_interp = np.concatenate([power_spectrum_theory*ps_values[0]/power_spectrum_theory[-1], ps_values])

    n_points = 1000 # k_values.shape[0] seems to fail!
    f2 = interp1d(k_values_interp, ps_values_interp, kind='linear')
    k_values_ = np.logspace(np.log10(k_values_interp[0]+1e-10), np.log10(9.99), n_points)
    power_spectrum = np.empty(n_points)
    for ind in range(n_points):
        power_spectrum[ind] = f2(k_values_[ind])
    smooth = 30
    power_spectrum = np.convolve(power_spectrum, np.ones(smooth)/smooth, mode='valid')
    k_values_ = k_values_[smooth//2-1:-smooth//2]
    return k_values_, power_spectrum

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from .data.data import DataScaler

# Define the Custom Dataset for log_normal case
class CustomMatterDataset(Dataset):
    def __init__(self, saved_params, ps_values, field_means, k, scalers=(None,None), transform=None):
        self.saved_params = np.array(saved_params)
        self.ps_values = ps_values
        self.field_means = field_means
        self.k = k
        self.transform = transform  # Optional transforms

        # Store scalers
        self.param_scaler, self.data_scaler = scalers

        # create a self.tensors dummy to avoid errors
        dummy_x = torch.stack([self.__getitem__(0)[0] for _ in range(2)])
        dummy_y = torch.stack([self.__getitem__(0)[1] for _ in range(2)])
        self.tensors = (dummy_x, dummy_y)
    def __len__(self):
        return len(self.saved_params)

    def __getitem__(self, idx):
        p = self.ps_values[idx]
        field_mean = self.field_means[idx]
        y = self.saved_params[idx]

        # Generate matter density (input image)
        x = sample_matter_density_from_spectrum(self.k, p, field_mean)

        # Normalize x if scaler exists
        if self.param_scaler:
            x = self.data_scaler.transform_standard(x)

        # Normalize y if scaler exists
        if self.data_scaler:
            y = self.param_scaler.transform_minmax(y)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if self.transform:
            x = self.transform(x)

        return x.unsqueeze(0), y
import time
import numpy as np
def fit_log_normal_scalers(train_loader):
    param_scaler = DataScaler()  # Standard scaler for x
    data_scaler = DataScaler()  # Min-max scaler for y

    x_samples, y_samples = [], []
    torch.manual_seed(0)
    np.random.seed(0)
    for x, y in train_loader:
        x_samples.append(x.numpy().reshape(x.shape[0], -1))  # Flatten images
        y_samples.append(y.numpy())

        if len(x_samples) * x.shape[0] >= 1000:  # Stop after collecting ~1K samples
            break
    torch.manual_seed(int(time.time()))
    np.random.seed(int(time.time()))

    x_samples = np.vstack(x_samples)
    y_samples = np.vstack(y_samples)

    param_scaler.fit_minmax(y_samples)
    data_scaler.fit_standard(x_samples)

    return param_scaler, data_scaler

# Function to create the dataset and dataloaders
import torch
from torch.utils.data import DataLoader
import numpy as np

def create_log_normal_dataloaders(saved_params, ps_values, field_means, k, batch_size, scalers):
    total_size = len(saved_params)
    indices = np.random.permutation(total_size)  # Random shuffle

    # Split sizes: 80% train, 10% validation, 10% test
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create separate datasets
    train_dataset = CustomMatterDataset(saved_params[train_indices], ps_values[train_indices], field_means[train_indices], k, scalers)
    val_dataset = CustomMatterDataset(saved_params[val_indices], ps_values[val_indices], field_means[val_indices], k, scalers)
    test_dataset = CustomMatterDataset(saved_params[test_indices], ps_values[test_indices], field_means[test_indices], k, scalers)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader
