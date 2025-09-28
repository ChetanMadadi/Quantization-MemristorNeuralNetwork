"""
Script to run robustness testing on trained models
Tests models with quantization levels and weight noise
"""

import torch
import torch.nn as nn
import numpy as np
import copy

from config import *
from models import create_resnet18_model, apply_post_training_quantization
from data_utils import get_data_loaders
from robustness_testing import (run_robustness_analysis, calculate_robustness_metrics, 
                               print_robustness_summary, clean_state_dict)
from visualization import (plot_robustness_analysis, plot_robustness_comparison_bar,
                          plot_test_accuracy_comparison, set_plot_style)
from utils import set_seed, get_device
from config import get_quantization_values

def load_trained_models(model_paths, device):
    """
    Load trained models from saved state dictionaries.
    
    Args:
        model_paths: Dictionary of model paths {'model_name': 'path'}
        device: Device to load models on
        
    Returns:
        dict: Dictionary of loaded models
    """
    models = {}
    
    for model_name, path in model_paths.items():
        try:
            model = create_resnet18_model(NUM_CLASSES).to(device)
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(clean_state_dict(state_dict))
            model.eval()
            models[model_name] = model
            print(f"Successfully loaded {model_name} from {path}")
        except Exception as e:
            print(f"Failed to load {model_name} from {path}: {e}")
    
    return models


def apply_quantization_to_models(models, quant_values, device):
    """
    Apply quantization levels to all loaded models.
    
    Args:
        models: Dictionary of models
        quant_values: Quantization levels to apply
        device: Device for computations
        
    Returns:
        dict: Dictionary of quantized models
    """
    quantized_models = {}
    quant_values_tensor = torch.tensor(quant_values, dtype=torch.float32, device=device)
    
    for model_name, model in models.items():
        # Create a copy to avoid modifying the original
        quantized_model = copy.deepcopy(model)
        apply_post_training_quantization(quantized_model, quant_values_tensor)
        quantized_models[model_name] = quantized_model
        print(f"Applied quantization to {model_name}")
    
    return quantized_models


def run_robustness_experiment(model_paths=None, noise_levels=None, repeats=10):
    """
    Run complete robustness experiment on trained models.
    
    Args:
        model_paths: Dictionary of model paths (optional, uses default names if None)
        noise_levels: List of noise levels to test (optional)
        repeats: Number of repetitions per noise level
        
    Returns:
        tuple: (robustness_results, robustness_metrics)
    """
    print("="*60)
    print("ROBUSTNESS TESTING EXPERIMENT")
    print("="*60)
    
    # Set up
    set_seed(RANDOM_SEED)
    set_plot_style()
    device = get_device()
    
    # Default model paths (assuming models were saved by main experiment)
    if model_paths is None:
        model_paths = {
            'ste': 'ste_model.pth',
            'alpha_blend': 'alpha_blend_model.pth',
            'baseline': 'baseline_model.pth'
        }
    
    # Load test data
    _, _, test_loader = get_data_loaders()
    criterion = nn.CrossEntropyLoss()
    
    # Load trained models
    models = load_trained_models(model_paths, device)
    
    if not models:
        print("No models loaded successfully. Cannot proceed with robustness testing.")
        return None, None
    
    # Get quantization levels for robustness testing
    quant_values = get_quantization_values()
    print(f"Using {len(quant_values)} quantization levels")
    
    # Apply quantization to all models
    quantized_models = apply_quantization_to_models(models, quant_values, device)
    
    # Create PTQ model from baseline
    if 'baseline' in quantized_models:
        quantized_models['ptq'] = copy.deepcopy(quantized_models['baseline'])
        print("Created PTQ model from quantized baseline")
    
    # Define noise levels if not provided
    if noise_levels is None:
        noise_levels = np.round(np.arange(0.0, 0.21, 0.01), 2).tolist()
    
    # Run robustness analysis
    robustness_results = run_robustness_analysis(
        quantized_models, test_loader, device, criterion, 
        noise_levels=noise_levels, repeats=repeats, seed=RANDOM_SEED
    )
    
    # Calculate robustness metrics
    robustness_metrics = calculate_robustness_metrics(robustness_results)
    
    # Print summary
    print_robustness_summary(robustness_metrics)
    
    # Create visualizations
    print("\nGenerating robustness plots...")
    plot_test_accuracy_comparison({}, robustness_results, save_path="robustness_test_accuracy.png")
    plot_robustness_analysis(robustness_results, save_path="robustness_analysis.png")
    plot_robustness_comparison_bar(robustness_metrics, save_path="robustness_metrics.png")
    
    # Save results
    torch.save({
        'robustness_results': robustness_results,
        'robustness_metrics': robustness_metrics,
        'noise_levels': noise_levels,
        'quantization_levels': quant_values.tolist()
    }, 'robustness_experiment_results.pth')
    
    print("\nRobustness experiment completed!")
    print("Results saved to 'robustness_experiment_results.pth'")
    
    return robustness_results, robustness_metrics


if __name__ == "__main__":
    # Example usage with custom model paths
    # model_paths = {
    #     'ste': '/path/to/ste_model.pth',
    #     'alpha_blend': '/path/to/ab_model.pth',
    #     'baseline': '/path/to/baseline_model.pth'
    # }
    
    # Run with default settings
    robustness_results, robustness_metrics = run_robustness_experiment()
    
    # Run with custom settings
    # robustness_results, robustness_metrics = run_robustness_experiment(
    #     model_paths=model_paths,
    #     noise_levels=np.arange(0.0, 0.11, 0.01).tolist(),
    #     repeats=5
    # )