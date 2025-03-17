from config.default import get_default_config
from config.experiments import experiments
import sys
sys.path.append('/home/asaoulis/projects/transfer_sbi/transfer/')
from transfer_sbi.eval.utils import evaluate_best_checkpoint

def retrieve_first_list_from_experiments(experiments):
    list_key = None
    list_values = None
    for key, value in experiments.items():
        if isinstance(value, list):
            list_key = key
            list_values = value
            break
    return list_key,list_values

if __name__ == "__main__":
    # Extract the list key and its values
    experiment_name = "anneal_LH_illustris"
    experiment_config = experiments[experiment_name]
    list_key, list_values = retrieve_first_list_from_experiments(experiment_config)

    config = get_default_config()
    config.experiment_name = experiment_name
    # Set the non-list values
    for key, val in experiment_config.items():
        if key != list_key:
            setattr(config, key, val)
    evaluate_best_checkpoint(config)