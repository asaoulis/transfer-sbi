
experiments = {

# LH experiments

    "scratch_LH_small":
        {"dataset_size": [200, 400, 800, 1600],
            "lr": 0.00005,
            "epochs": 250,
            "batch_size": 64},
    "scratch_LH_long":
        {"dataset_size": [200, 400, 800, 1600, 3200, 6400],
        "lr": 0.0002,
        "epochs": 200,
        "batch_size": 64,
        "scheduler_kwargs": {'warmup': 1000, 'gamma': 0.98}},
    "baseline_LH":
        {"dataset_size": [15000],
        "lr": 0.0002,
        "repeats": 5,
        "scheduler_type": "cyclic",
        "dataset_suite": "LH",
        "epochs": 250,
        "batch_size": 64,
        "scheduler_kwargs": {'warmup': 1000, 'gamma': 0.995}},
    "pretrain_nbody":
        {"dataset_size": [15000],
        "dataset_name": 'nbody',
        "lr": 0.0002,
        "epochs": 300,
        "batch_size": 128,
        "scheduler_type": "cyclic"},
    "baseline_SB": {
        "dataset_size": [15000],
        "lr": 0.0002,
        "epochs": 500,
        "batch_size": 128,
        "scheduler_type": "cyclic",
        "dataset_suite" : "SB28",
        },
    "finetune_LH_illustris":
        {"dataset_size": [200, 400, 800, 1600],
        "lr": 0.00001,
        "epochs": 100,
        "batch_size": 64,
        "scheduler_type": "exp",
        "dataset_suite": "LH",
        "checkpoint_path": "/share/gpu0/asaoulis/cmd/checkpoints/pretrain_nbody",
        "match_string":"ds15000"
        },
    "finetune_random_LH_illustris":
        {"dataset_size": [200, 400, 800, 1600],
        "lr": 0.00001,
        "epochs": 100,
        "batch_size": 64,
        "scheduler_type": "exp",
        "dataset_suite": "LH",
        "checkpoint_path": "/share/gpu0/asaoulis/cmd/checkpoints/pretrain_nbody",
        "match_string":"ds15000"
        },
    "anneal_LH_illustris":
        {"dataset_size": [15000],
        "lr": 0.00001,
        "epochs": 50,
        "batch_size": 64,
        "scheduler_type": "exp",
        "dataset_suite": "LH",
        "checkpoint_path": "/share/gpu0/asaoulis/cmd/checkpoints/baseline_LH",
        "match_string":"LH_cyclic_ds15000"
        },
    "anneal_random_LH_illustris":
        {"dataset_size": [15000],
        "lr": 0.00001,
        "epochs": 50,
        "batch_size": 64,
        "scheduler_type": "exp",
        "dataset_suite": "LH",
        "checkpoint_path": "/share/gpu0/asaoulis/cmd/checkpoints/baseline_LH",
        "match_string":"LH_cyclic_ds15000"
        },
# SB28 experiments
    "scratch_SB_training_slow":
        {"dataset_size": [200, 400, 800, 1600, 3200, 6400, 12800],
        "lr": 0.0002,
        "scheduler_type": "exp",
        "dataset_suite": "SB28",
        "epochs": 200,
        "batch_size": 64,
        "scheduler_kwargs": {'warmup': 1000, 'gamma': 0.98}},
    "baseline_SB":
        {"dataset_size": [30000],
        "lr": 0.0002,
        "scheduler_type": "cylic",
        "dataset_suite": "SB28",
        "epochs": 300,
        "batch_size": 64,
        "scheduler_kwargs": {'warmup': 1000, 'gamma': 0.995}},
    "pretrain_nbody_SB":
        {"dataset_size": [30000],
        "dataset_name": 'nbody',
        "lr": 0.0002,
        "scheduler_type": "cyclic",
        "dataset_suite": "SB28",
        "epochs": 300,
        "batch_size": 64,
        "scheduler_kwargs": {'warmup': 250, 'gamma': 0.999}},
}
