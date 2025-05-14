import ml_collections

def get_default_config():
    config = ml_collections.ConfigDict()
    config.dataset_name = "illustris"
    config.dataset_suite = "LH"
    config.scaling_dataset = None
    config.dataset_size = 25000
    config.lr = 0.0005
    config.epochs = 400
    config.batch_size = 128
    config.latent_dim = 128
    config.extra_blocks = 0
    config.checkpoint_path = None
    config.scheduler_type = 'exp'
    config.model_type = "o3"
    config.optimizer_kwargs = {'weight_decay': 0.01, 'betas': (0.9, 0.999)}
    config.scheduler_kwargs = {'warmup': 250, 'gamma': 0.9}
    config.repeats = 6
    config.experiment_name = None
    config.data_seed = None
    config.log_normal_dataset_path = None
    config.unpaired = False
    config.freeze_cnn = False
    config.match_string = None
    return config