"""
Robustness testing module for quantized neural networks
Tests model performance under weight noise conditions
"""

import numpy as np
import torch
import torch.nn as nn
from utils import set_seed


def add_percentage_gaussian_noise_to_weights(model, percent):
    """
    Adds Gaussian noise to each weight with std = percent * abs(weight).
    
    Args:
        model: PyTorch model to add noise to
        percent: Noise level as percentage of weight magnitude
        
    Returns:
        dict: Original weights before adding noise
    """
    original_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            std = torch.clamp(percent * param.data.abs(), min=1e-9)
            noise = torch.normal(mean=0.0, std=std).to(param.device)
            original_weights[name] = param.data.clone()
            param.data += noise
    return original_weights


def restore_original_weights(model, original_weights):
    """
    Restore model weights from saved dictionary.
    
    Args:
        model: PyTorch model to restore weights for
        original_weights: Dictionary of original weights
    """
    for name, param in model.named_parameters():
        if name in original_weights:
            param.data.copy_(original_weights[name])


def test_with_weight_noise(model, test_loader, device, criterion, noise_levels_percent, 
                          repeats=10, seed=40):
    """
    Test model robustness under different levels of weight noise.
    
    Args:
        model: PyTorch model to test
        test_loader: Test data loader
        device: Device to run computations on
        criterion: Loss function
        noise_levels_percent: List of noise levels (as fractions, e.g., 0.1 for 10%)
        repeats: Number of repetitions per noise level
        seed: Base random seed
        
    Returns:
        list: List of tuples (noise_level, mean_accuracy, std_accuracy)
    """
    noise_accuracy = []

    for percent in noise_levels_percent:
        print(f"\nTesting with weight noise: {percent * 100:.1f}%")
        accuracies = []

        for r in range(repeats):
            set_seed(seed + r)  # Ensure deterministic noise per repeat

            # Add noise
            original_weights = add_percentage_gaussian_noise_to_weights(model, percent)

            # Evaluation
            model.eval()
            correct, total, total_loss = 0, 0, 0.0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100. * correct / total
            accuracies.append(accuracy)

            # Restore weights
            restore_original_weights(model, original_weights)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        noise_accuracy.append((percent, mean_acc, std_acc))

        print(f"Avg Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}% over {repeats} runs")

    return noise_accuracy


def run_robustness_analysis(models_dict, test_loader, device, criterion, 
                          noise_levels=None, repeats=10, seed=40):
    """
    Run comprehensive robustness analysis on multiple models.
    
    Args:
        models_dict: Dictionary of models to test {'model_name': model}
        test_loader: Test data loader
        device: Device to run computations on
        criterion: Loss function
        noise_levels: List of noise levels to test (default: 0% to 20%)
        repeats: Number of repetitions per noise level
        seed: Base random seed
        
    Returns:
        dict: Results dictionary with robustness data for each model
    """
    if noise_levels is None:
        noise_levels = np.round(np.arange(0.0, 0.21, 0.01), 2).tolist()
    
    results = {}
    
    for model_name, model in models_dict.items():
        print(f"\n{'='*60}")
        print(f"ROBUSTNESS TESTING: {model_name.upper()}")
        print(f"{'='*60}")
        
        robustness_results = test_with_weight_noise(
            model, test_loader, device, criterion, noise_levels, repeats, seed
        )
        
        results[model_name] = {
            'noise_levels': noise_levels,
            'results': robustness_results,
            'mean_accuracies': [result[1] for result in robustness_results],
            'std_accuracies': [result[2] for result in robustness_results]
        }
    
    return results


def calculate_robustness_metrics(robustness_results):
    """
    Calculate robustness metrics from test results.
    
    Args:
        robustness_results: Results from run_robustness_analysis
        
    Returns:
        dict: Robustness metrics for each model
    """
    metrics = {}
    
    for model_name, results in robustness_results.items():
        noise_levels = results['noise_levels']
        mean_accs = results['mean_accuracies']
        
        
        # Calculate accuracy drop at specific noise levels
        baseline_acc = mean_accs[0]  # 0% noise
        acc_drop_5 = baseline_acc - mean_accs[5] if len(mean_accs) > 5 else 0  # 5% noise
        acc_drop_10 = baseline_acc - mean_accs[10] if len(mean_accs) > 10 else 0  # 10% noise
        acc_drop_20 = baseline_acc - mean_accs[20] if len(mean_accs) > 20 else 0  # 20% noise
        
        metrics[model_name] = {
            'baseline_accuracy': baseline_acc,
            'accuracy_drop_5%': acc_drop_5,
            'accuracy_drop_10%': acc_drop_10,
            'accuracy_drop_20%': acc_drop_20
        }
    
    return metrics


def print_robustness_summary(robustness_metrics):
    """
    Print a summary of robustness metrics.
    
    Args:
        robustness_metrics: Results from calculate_robustness_metrics
    """
    print(f"\n{'='*60}")
    print("ROBUSTNESS ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    print(f"{'Model':<15} {'Baseline':<10} {'5% Drop':<10} {'10% Drop':<11} {'20% Drop':<11} {'AUC':<10}")
    print("-" * 70)
    
    for model_name, metrics in robustness_metrics.items():
        print(f"{model_name:<15} {metrics['baseline_accuracy']:<10.2f} "
              f"{metrics['accuracy_drop_5%']:<10.2f} {metrics['accuracy_drop_10%']:<11.2f} "
              f"{metrics['accuracy_drop_20%']:<11.2f} {metrics['auc_robustness']:<10.2f}")


def clean_state_dict(state_dict):
    """
    Clean state dictionary by removing common prefixes.
    Used when loading models with different naming conventions.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        dict: Cleaned state dictionary
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace('resnet.', '')  # Remove resnet prefix
        k = k.replace('.module', '')  # Remove .module part
        k = k.replace("model.", "")   # Remove model prefix
        new_state_dict[k] = v
    return new_state_dict