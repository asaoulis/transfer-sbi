from config.default import get_default_config
from config.experiments import experiments
import sys
sys.path.append('/home/asaoulis/projects/transfer_sbi/transfer/')
from transfer_sbi.eval.utils import evaluate_best_checkpoint, parse_results, load_best_model_and_build_posterior
from transfer_sbi.eval.ensemble import evaluate_best_ensemble
# set fontsize to 18
from pathlib import Path
import numpy as np

# check if cuda is available
import torch
print(torch.cuda.is_available())

def retrieve_first_list_from_experiments(experiments):
    list_key = None
    list_values = None
    for key, value in experiments.items():
        if isinstance(value, list):
            list_key = key
            list_values = value
            break
    return list_key,list_values

def load_config(experiment_name):
    experiment_config = experiments[experiment_name]
    list_key, list_values = retrieve_first_list_from_experiments(experiment_config)

    config = get_default_config()
    config.experiment_name = experiment_name
    # Set the non-list values
    for key, val in experiment_config.items():
        if key != list_key:
            setattr(config, key, val)
    return config

def load_accurate_ensemble_samples(config):
    experiment_path = f"/share/gpu0/asaoulis/cmd/checkpoints/{config.experiment_name}"
    filepath = Path(experiment_path) / 'accurate_ensemble_samples.npy'
    if filepath.exists():
        accurate_ensemble_samples = np.load(filepath, allow_pickle=True)
    else:
        accurate_ensemble_samples = None
    return accurate_ensemble_samples

from transfer_sbi.utils import prepare_data_parameters

base_config = 'finetune_log_normal_LH'

experiments_to_evaluate = [
    # "anneal_SB_illustris", 
     "finetune_log_normal_LH", 
    #  "scratch_SB_training_slow",
    #  "baseline_SB"
]

config = load_config(base_config)
data_parameters = prepare_data_parameters(config)
accurate_ensemble_samples = load_accurate_ensemble_samples(config)

if __name__ == "__main__":
    
    for experiment_name in experiments.keys():
        if experiment_name in experiments_to_evaluate: 
            config = load_config(experiment_name)
            evaluate_best_checkpoint(config, data_parameters, accurate_ensemble_samples)