from config.default import get_default_config
from transfer_sbi.train.utils import train_model

def retrieve_first_list_from_experiments(experiments):
    list_key = None
    list_values = None
    for key, value in experiments.items():
        if isinstance(value, list):
            list_key = key
            list_values = value
            break
    return list_key,list_values


experiments = {
    "scratch_LH_training":
        {"dataset_size": [200, 600, 1500, 4500, 10000],
        "lr": 0.0005,
        "epochs": 200,
        "batch_size": 64},
    "baseline_LH":
        {"dataset_size": 28000,
        "lr": 0.0005,
        "epochs": 500,
        "batch_size": 128,
        "scheduler_type": "cyclic"}
}

if __name__ == "__main__":
    # Extract the list key and its values
    experiment_name = "dataset_size_LH_experiment"
    list_key, list_values = retrieve_first_list_from_experiments(experiments[experiment_name])

    for value in list_values:
        config = get_default_config()
        config.experiment_name = experiment_name
        # Set the non-list values
        for key, val in experiments.items():
            if key != list_key:
                setattr(config, key, val)
        # Set the list value
        setattr(config, list_key, value)
        print(f"Running experiment with {list_key}={value}")
        train_model(config)