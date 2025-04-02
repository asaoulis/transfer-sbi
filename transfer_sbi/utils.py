import pickle
from .train.compressors import _MODEL_BUILDERS
from .train.lightning_modules import NDELightningModule
from .data.dataset import get_dataset, prepare_dataloaders

def prepare_data_and_model(config, data_parameters = None):
    if data_parameters is None:
        scalers, train_loader, val_loader, test_loader = prepare_data_parameters(config)
    else:
        scalers, train_loader, val_loader, test_loader = data_parameters
        
        # Select the correct backbone dynamically
    embedding_model = _MODEL_BUILDERS[config.model_type](config.latent_dim).to(device='cuda')
        
    model = NDELightningModule(
            embedding_model, conditioning_dim=config.latent_dim, lr=config.lr, scheduler_type=config.scheduler_type,
            element_names=["Omega", "sigma8"], test_dataloader=test_loader, optimizer_kwargs=config.optimizer_kwargs, num_extra_blocks=config.extra_blocks,
            checkpoint_path=config.checkpoint_path, scheduler_kwargs={**config.scheduler_kwargs, **{'warmup': min(config.dataset_size // 4, 1000)}}
        )
    
    return train_loader,val_loader, model, scalers

from torch.utils.data import DataLoader
from .log_normal import create_log_normal_dataloaders, fit_log_normal_scalers, CustomMatterDataset

# Main function to prepare data
def prepare_data_parameters(config):
    if config.dataset_name == "log_normal":
        with open(config.log_normal_dataset_path, "rb") as f:
            log_normal_data = pickle.load(f)
        saved_params, ps_values, field_means, k = log_normal_data
        # Create train loader first
        train_loader, _, _ = create_log_normal_dataloaders(
            saved_params, ps_values, field_means, k, config.batch_size, scalers=(None, None)
        )

        param_scaler, data_scaler = fit_log_normal_scalers(train_loader)
        train_loader, val_loader, test_loader = create_log_normal_dataloaders(
            saved_params, ps_values, field_means, k, config.batch_size, scalers=(param_scaler, data_scaler)
        )
        return (param_scaler, data_scaler), train_loader, val_loader, test_loader

    else:
        # Default dataset loading
        train_x, train_y, valid_x, valid_y, test_data, scalers = get_dataset(
            config.dataset_name, config.dataset_suite, config.dataset_size, config.scaling_dataset, config.data_seed
        )
        train_loader, val_loader, test_loader = prepare_dataloaders(
            train_y, train_x, valid_y, valid_x, test_data, config.batch_size
        )
        return scalers, train_loader, val_loader, test_loader

