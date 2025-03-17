
experiments = {
    "scratch_LH_small":
        {"dataset_size": [200, 400, 800, 1600],
            "lr": 0.00005,
            "epochs": 250,
            "batch_size": 64},
    "scratch_LH_training_long":
        {"dataset_size": [200, 400, 800, 1600, 3200, 6400],
        "lr": 0.0002,
        "epochs": 300,
        "batch_size": 64},
    "baseline_LH":
        {"dataset_size": [15000],
        "lr": 0.00004,
        "epochs": 250,
        "batch_size": 128,
        "scheduler_type": "exp",
        "scheduler_kwargs": {'warmup': 250, 'gamma': 0.999}},
    "pretrain_nbody":
        {"dataset_size": [15000],
        "dataset_name": 'nbody',
        "lr": 0.0002,
        "epochs": 500,
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
}
