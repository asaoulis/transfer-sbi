import os
import torch
import glob
import re
import json
import wandb
import pytorch_lightning as pl
from ..utils import prepare_data_and_model

from .fom import compute_fom

def find_best_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint-epoch=*-val_log_prob=*.ckpt"))
    
    best_checkpoint = None
    best_val_loss = float("inf")
    
    # Updated regex pattern to match any number of digits for epoch and allow negative float for loss
    loss_pattern = re.compile(r"checkpoint-epoch=(\d+)-val_log_prob=(-?\d+\.\d+).ckpt")
    
    for ckpt in checkpoint_files:
        match = loss_pattern.search(ckpt)
        if match:
            epoch = int(match.group(1))  # Extract epoch number
            val_loss = float(match.group(2))  # Extract validation loss
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = ckpt
    
    return best_checkpoint, best_val_loss


def mock_evaluation(model):
    with torch.no_grad():
        test_log_prob = model.compute_avg_log_prob()
    return {"test_log_prob": test_log_prob}


def evaluate_best_checkpoint(config, data_parameters = None):
    experiment_path = f"/share/gpu0/asaoulis/cmd/checkpoints/{config.experiment_name}"
    if not os.path.exists(experiment_path):
        return
    run_folders = [os.path.join(experiment_path, d) for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]
    print(f'Running eval on {config.experiment_name} with {len(run_folders)} runs')
    results = {}
    
    for run_folder in run_folders:
        best_checkpoint, best_val_loss = find_best_checkpoint(run_folder)
        config.checkpoint_path = best_checkpoint
        
        if best_checkpoint:            
            _, _, model, _ = prepare_data_and_model(config, data_parameters)
            
            model.to("cuda")
            model.eval()
            metrics = mock_evaluation(model)
            
            results[run_folder] = {
                "best_checkpoint": best_checkpoint,
                "best_val_loss": best_val_loss,
                "metrics": metrics
            }
            
            results_path = os.path.join(run_folder, "evaluation_results.json")
            with open(results_path, "w") as f:
                json.dump(results[run_folder], f, indent=4)
    
    return results


import os
import json
import numpy as np

def parse_results(experiment_name, base_path="/share/gpu0/asaoulis/cmd/checkpoints"):
    """
    Parses evaluation results from checkpoint directories and computes mean and standard error for metrics.
    
    Args:
        experiment_name (str): The name of the experiment.
        base_path (str): The base directory where experiment folders are stored.
    
    Returns:
        dict: A dictionary where the keys are the base run names (without _i suffix) and values contain
              mean and standard error for each metric.
    """
    experiment_path = os.path.join(base_path, experiment_name)
    run_folders = [os.path.join(experiment_path, d) for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]
    
    results = {}
    aggregated_results = {}
    
    for run_folder in run_folders:
        results_file = os.path.join(run_folder, "evaluation_results.json")
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                results_data = json.load(f)
                lowest_level_folder = os.path.basename(run_folder)  # Extract the last folder name
                results[lowest_level_folder] = results_data["metrics"]
    
    # Aggregate results by base run name (without _i suffix)
    for run_name, metrics in results.items():
        base_name = "_".join(run_name.split("_")[:-1]) if run_name.split("_")[-1].isdigit() else run_name
        
        if base_name not in aggregated_results:
            aggregated_results[base_name] = {}
        
        for metric, value in metrics.items():
            if metric not in aggregated_results[base_name]:
                aggregated_results[base_name][metric] = []
            aggregated_results[base_name][metric].append(value)
    
    # Compute mean and standard error for each metric
    final_results = {}
    for base_name, metrics in aggregated_results.items():
        final_results[base_name] = {}
        for metric, values in metrics.items():
            values = np.array(values)
            mean = np.mean(values)
            stderr = np.std(values, ddof=1) / np.sqrt(len(values))  # Standard error
            final_results[base_name][metric] = {"mean": mean, "stderr": stderr}
    
    return final_results

def load_best_model_and_build_posterior(config, ds_string_match="", data_parameters = None):
    experiment_path = f"/share/gpu0/asaoulis/cmd/checkpoints/{config.experiment_name}"
    run_folders = [os.path.join(experiment_path, d) for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]

    best_model_path = None
    best_val_loss = float("inf")
    best_model = None

    for run_folder in run_folders:
        if ds_string_match not in run_folder:
            continue
        results_path = os.path.join(run_folder, "evaluation_results.json")
        
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
            
            val_loss = results.get("best_val_loss", float("inf"))
            checkpoint_path = results.get("best_checkpoint", None)
            
            if checkpoint_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = checkpoint_path

    if best_model_path:
        print(f"Loading best model from: {best_model_path} with val loss: {best_val_loss}")

        config.checkpoint_path = best_model_path
        print(best_model_path,best_model_path)
        _, _, best_model, scalers = prepare_data_and_model(config, data_parameters)  # Assuming a function exists to get config
        best_model.load_from_checkpoint(best_model_path)
        best_model.to("cuda" if torch.cuda.is_available() else "cpu")
        best_model.eval()

        # Build posterior
        posterior = best_model.build_posterior_object()
        return posterior, best_model.test_dataloader, scalers
    else:
        print("No valid checkpoints found.")
        return None
