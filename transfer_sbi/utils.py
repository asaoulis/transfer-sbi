
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

def prepare_data_parameters(config):
    train_x, train_y, valid_x, valid_y, test_data, scalers = get_dataset(config.dataset_name, config.dataset_suite, config.dataset_size, config.scaling_dataset)
    train_loader, val_loader, test_loader = prepare_dataloaders(train_y, train_x, valid_y, valid_x, test_data, config.batch_size)
    return scalers,train_loader,val_loader,test_loader

