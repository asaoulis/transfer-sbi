import torch
import wandb
from torch.utils.data import DataLoader, random_split, TensorDataset
import lightning as pl

from data.dataset import get_dataset, AugmentDataset, train_transform
from compressors import _MODEL_BUILDERS
from lightning_modules import NDELightningModule

def prepare_dataloaders(x, y, testxy, batch_size):
    dataset = AugmentDataset(x, y, train_transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    test_x, test_y = testxy
    test_dataset = TensorDataset(torch.tensor(test_x, dtype=torch.float32),
                                 torch.tensor(test_y, dtype=torch.float32))
    
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8),
            DataLoader(val_dataset, batch_size=batch_size, num_workers=8),
            DataLoader(test_dataset, batch_size=128, num_workers=8))

def fit_model(model, epochs, logger, train_loader, val_loader, experiment_name):
    monitor_string = f"val_{model.loss_name}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=f"{monitor_string}",
        dirpath=f"/share/gpu0/asaoulis/cmd/checkpoints/{experiment_name}/run_{wandb.run.name}",
        filename=f"checkpoint-{{epoch:02d}}-{{{monitor_string}:.4f}}",
        save_top_k=3,
        mode="min",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        logger=pl.loggers.WandbLogger() if logger else None,
        callbacks=[checkpoint_callback, lr_monitor],
    )
    
    trainer.fit(model, train_loader, val_loader)

def train_model(config, pretrain=False):
    for i in range(config.repeats):
        train_x, train_y, test_data = get_dataset(config.dataset_name, config.dataset_suite, config.dataset_size, config.scaling_dataset)
        train_loader, val_loader, test_loader = prepare_dataloaders(train_y, train_x, test_data, config.batch_size)
        
        # Select the correct backbone dynamically
        embedding_model = _MODEL_BUILDERS[config.model_type](config.latent_dim).to(device='cuda')
        
        model = NDELightningModule(
            embedding_model, conditioning_dim=config.latent_dim, lr=config.lr, scheduler_type=config.scheduler_type,
            element_names=["Omega", "sigma8"], test_dataloader=test_loader, optimizer_kwargs=config.optimizer_kwargs,
            checkpoint_path=config.checkpoint_path, scheduler_kwargs={'warmup': 250 if pretrain else 1000, **config.scheduler_kwargs}
        )
        
        if not pretrain:
            model.append_maf_blocks(
                train_x, train_y, num_extra_blocks=config.extra_blocks, bounds=(0,1), device='cuda', init_scale=0.02
            )
        
        logger = wandb.init(
            project="camels-nbody-illustris-cosmo",
            name=f"{'pretrain' if pretrain else 'finetune'}_{config.dataset_name}_{config.scheduler_type}_{i}",
            reinit=True
        )

        fit_model(model, config.epochs, logger, train_loader, val_loader, config.experimental_name)

