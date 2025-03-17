import numpy as np
from sbi import utils as utils
import torch
from sbi import inference as inference
from transfer_sbi.toy.custom_sbi import CustomSNPE_C, build_maf, build_nsf
import torch.nn as nn

# torch dataset and dataloader
from torch.utils.data import Dataset, DataLoader


y1_simulation_4d = lambda x1, x2, x3, x4: x1 + 0.2 * x2**2  +0.05*x4 + 0.1*x3
y2_simulation_4d = lambda x1, x2, x3, x4: x1 + 0.3 * x2**2 -0.05*x3 - 0.1*x4
y3_simulation_4d = lambda x1, x2, x3, x4: x3 + 0.3 * x4
y4_simulation_4d = lambda x1, x2, x3, x4: x4 - 0.3 * x3

def cheap_simulation_4d(x1, x2, x3, x4, add_noise=None):
    noise1 = np.random.normal(0, 0.1, x1.shape) if add_noise is None else add_noise[:, 0]
    noise2 = np.random.normal(0, 0.1, x2.shape) if add_noise is None else add_noise[:, 1]
    noise3 = np.random.normal(0, 0.1, x3.shape) if add_noise is None else add_noise[:, 2]
    noise4 = np.random.normal(0, 0.1, x4.shape) if add_noise is None else add_noise[:, 3]
    add_noise = np.stack([noise1, noise2, noise3, noise4], axis=-1).squeeze()

    y1 = y1_simulation_4d(x1, x2, x3, x4) + noise1
    y2 = y2_simulation_4d(x1, x2, x3, x4) + noise2
    y3 = y3_simulation_4d(x1, x2, x3, x4) + noise3
    y4 = y4_simulation_4d(x1, x2, x3, x4) + noise4
    return np.stack([y1, y2, y3, y4], axis=-1).squeeze(), add_noise

def expensive_simulation_4d(x1, x2, x3, x4, add_noise=None):
    noise1 = np.random.normal(0, 0.1, x1.shape) if add_noise is None else add_noise[:, 0]
    noise2 = np.random.normal(0, 0.1, x2.shape) if add_noise is None else add_noise[:, 1]
    noise3 = np.random.normal(0, 0.1, x3.shape) if add_noise is None else add_noise[:, 2]
    noise4 = np.random.normal(0, 0.1, x4.shape) if add_noise is None else add_noise[:, 3]
    add_noise = np.stack([noise1, noise2, noise3, noise4], axis=-1).squeeze()

    y1 = y1_simulation_4d(x1, x2, x3, x4) + 0.05*x1**3 - 0.05*x2**3 + noise1 * (1 + 0.05*x1**2 + 0.1*x2**3) 
    y2 = y2_simulation_4d(x1, x2, x3, x4) + 0.03*x1 - 0.025 + noise2 * (1 + 0.05*x1 - 0.1*x2**3)
    y3 = y3_simulation_4d(x1, x2, x3, x4) + noise3 * (1 + 0.05*x3 - 0.1*x4**3)
    y4 = y4_simulation_4d(x1, x2, x3, x4) + noise4* (1 + 0.05*x4 - 0.1*x3**3)
    return np.stack([y1, y2, y3, y4], axis=-1).squeeze(), add_noise

y1_simulation_4d = lambda x1, x2, x3, x4: x1 + 0.2 * x2**2  +0.05*x4 + 0.1*x3
y2_simulation_4d = lambda x1, x2, x3, x4: x1 + 0.3 * x2**2 -0.05*x3 - 0.1*x4
y3_simulation_4d = lambda x1, x2, x3, x4: x3 + 0.3 * x4
y4_simulation_4d = lambda x1, x2, x3, x4: x4 - 0.3 * x3

def bad_cheap_simulation_4d(x1, x2, x3, x4, add_noise=None):
    noise1 = np.random.normal(0, 0.1, x1.shape) if add_noise is None else add_noise[:, 0]
    noise2 = np.random.normal(0, 0.1, x2.shape) if add_noise is None else add_noise[:, 1]
    noise3 = np.random.normal(0, 0.1, x3.shape) if add_noise is None else add_noise[:, 2]
    noise4 = np.random.normal(0, 0.1, x4.shape) if add_noise is None else add_noise[:, 3]
    add_noise = np.stack([noise1, noise2, noise3, noise4], axis=-1).squeeze()

    y1 = y1_simulation_4d(x1, x2, x3, x4) + noise1
    y2 = y2_simulation_4d(x1, x2, x3, x4) + noise2
    y3 = y3_simulation_4d(x1, x2, x3, x4) + noise3
    y4 = y4_simulation_4d(x1, x2, x3, x4) + noise4
    return np.stack([y1, y2, y3, y4], axis=-1).squeeze(), add_noise

def good_expensive_simulation_4d(x1, x2, x3, x4, add_noise=None):
    noise1 = np.random.normal(0, 0.1, x1.shape) if add_noise is None else add_noise[:, 0]
    noise2 = np.random.normal(0, 0.1, x2.shape) if add_noise is None else add_noise[:, 1]
    noise3 = np.random.normal(0, 0.1, x3.shape) if add_noise is None else add_noise[:, 2]
    noise4 = np.random.normal(0, 0.1, x4.shape) if add_noise is None else add_noise[:, 3]
    add_noise = np.stack([noise1, noise2, noise3, noise4], axis=-1).squeeze()

    y1 = y1_simulation_4d(x1, x2, x3, x4) + 0.2*x1**3 - 0.1*x2**2 + noise1 * (1 + 0.2*x1**2 + 0.1*x2**3) 
    y2 = y2_simulation_4d(x1, x2, x3, x4) + 0.15*x1**3 - 0.025 + noise2 * (1 + 0.15*x1 - 0.3*x2**3)
    y3 = y3_simulation_4d(x1, x2, x3, x4) + 0.15 * x4**3 + noise3 * (1 + 0.05*x3 - 0.1*x4**3)
    y4 = y4_simulation_4d(x1, x2, x3, x4) -0.1 *x4**2 +  noise4* (1 + 0.05*x4 - 0.1*x3**3)
    return np.stack([y1, y2, y3, y4], axis=-1).squeeze(), add_noise


y1_simulation = lambda x1, x2: x1 + 0.2 * x2**2 
y2_simulation = lambda x1, x2: x1 + 0.3 * x2**2


# Simulate an "inexpensive" function with slight tail divergence, using x1 and x2 as inputs
def inexpensive_simulation(x1, x2, add_noise=None):
    # Generate noise if needed
    noise1 = np.random.normal(0, 0.1, x1.shape) if add_noise is None else add_noise[:, 0]
    noise2 = np.random.normal(0, 0.1, x2.shape) if add_noise is None else add_noise[:, 1]
    add_noise = np.stack([noise1, noise2], axis=-1).squeeze()
    # Slightly nonlinear transformations for inexpensive simulation
    y1 = y1_simulation(x1, x2) + noise1
    y2 = y1_simulation(x1, x2) + noise2 
    return np.stack([y1, y2], axis=-1).squeeze(), add_noise

# Simulate an "expensive" function with more significant tail divergence, using x1 and x2 as inputs
def expensive_simulation(x1, x2, add_noise=None):
    # Generate noise if needed
    noise1 = np.random.normal(0, 0.1, x1.shape) if add_noise is None else add_noise[:, 0][:, np.newaxis]
    noise2 = np.random.normal(0, 0.1, x2.shape) if add_noise is None else add_noise[:, 1][:, np.newaxis]
    add_noise = np.stack([noise1, noise2], axis=-1).squeeze()
    # Slightly different nonlinear transformations for expensive simulation
    y1 = y1_simulation(x1, x2) + 0.05*x1**3 - 0.05*x2**3 + noise1 * (1 + 0.05*x1**2 + 0.1*x2**3) 
    y2 = y2_simulation(x1, x2) + 0.03*x1 - 0.025 + noise2 * (1 + 0.05*x1 - 0.1*x2**3)
    return np.stack([y1, y2], axis=-1).squeeze(), add_noise

# Define a prior distribution (here, using a uniform distribution as an example)
def pretrain_model(x_data, y_data):
    prior = utils.BoxUniform(low=-1.0 * torch.ones(2), high=1.0 * torch.ones(2))

    # Initialize the posterior estimator with MAF (Masked Autoregressive Flow)
    inference_method = inference.SNPE(prior=prior)

    # Run the inference on inexpensive simulation data
    inference_method.append_simulations(x_data, y_data, exclude_invalid_x=False)
    # create a custom adam optimizer with weight decay
    # from torch.optim import Adam
    # optimizer = Adam(list(inference_method._neural_net.parameters()), lr=0.0005, weight_decay=1e-4)
    # inference_method.optimizer = optimizer
    density_estimator = inference_method.train()
    posterior_inexpensive = inference_method.build_posterior(density_estimator)
    pretrained_state_dict = inference_method._neural_net.state_dict()

    return posterior_inexpensive, pretrained_state_dict

def train_and_log_test(x_data, y_data, test_x_data, test_y_data, logger, dim=2, embedding_net=None, inference_class=CustomSNPE_C, bounds=(1., -1.), device=None, batch_size = None, use_scheduler=False):
    prior = utils.BoxUniform(low=bounds[0] * torch.ones(dim), high=bounds[1] * torch.ones(dim), device=device)

    test_x_data = torch.tensor(test_x_data, dtype=torch.float32)
    test_y_data = torch.tensor(test_y_data, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(test_x_data, test_y_data)
    batch_size = len(dataset) if batch_size is None else batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    import torch.nn as nn
    embedding_net = nn.Identity() if embedding_net is None else embedding_net

    # Initialize the posterior estimator with MAF (Masked Autoregressive Flow)
    inference_method = inference_class(prior=prior, device=device)
    # Run the inference on inexpensive simulation data
    inference_method.append_simulations(x_data, y_data, exclude_invalid_x=False, data_device='cpu')
    # create a custom adam optimizer with weight decay
    from torch.optim import Adam, lr_scheduler
    net= build_nsf(x_data, y_data, embedding_net=embedding_net, z_score_x=None, z_score_y=None, use_residual_blocks=False, hidden_features=64, num_transforms=4)
    # lr = 0.00008
    lr = 0.001
    wd = 1e-4
    print(lr, wd)
    optimizer = Adam(list(net.parameters()), lr=lr, weight_decay=wd)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8) if use_scheduler else None

    # inference_method.optimizer = optimizer
    density_estimator = inference_method.train(network=net, optimizer=[optimizer], stop_after_epochs=20,test_dataloader=dataloader, logger=logger, training_batch_size=batch_size, scheduler=scheduler)
    posterior_inexpensive = inference_method.build_posterior(density_estimator)
    pretrained_state_dict = inference_method._neural_net.state_dict()

    return posterior_inexpensive, pretrained_state_dict