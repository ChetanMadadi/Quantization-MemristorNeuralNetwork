# -*- coding: utf-8 -*-
"""
Utility functions for experiments
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import BASE_LR, LR_MIN_SCALER, MOMENTUM, WEIGHT_DECAY


def set_seed(seed):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Uncomment for fully deterministic behavior (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get available device (CUDA if available, otherwise CPU)
    
    Returns:
        torch.device: Device to use for computation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def create_optimizer_and_scheduler(model, epochs):
    """
    Create optimizer and learning rate scheduler
    
    Args:
        model: Model to optimize
        epochs: Total number of training epochs
    
    Returns:
        tuple: (optimizer, scheduler)
    """
    optimizer = optim.SGD(
        model.parameters(), 
        lr=BASE_LR, 
        momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=BASE_LR * LR_MIN_SCALER
    )
    
    return optimizer, scheduler


def get_criterion():
    """
    Get loss function
    
    Returns:
        Loss function
    """
    return nn.CrossEntropyLoss()


def print_model_summary(model, model_name):
    """
    Print model summary including parameter count
    
    Args:
        model: PyTorch model
        model_name: Name of the model for display
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{model_name} Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model: {model}")


def save_results(results_dict, filename):
    """
    Save training results to file
    
    Args:
        results_dict: Dictionary containing training results
        filename: Output filename
    """
    import json
    
    with open(filename, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Results saved to {filename}")


def load_results(filename):
    """
    Load training results from file
    
    Args:
        filename: Input filename
    
    Returns:
        dict: Training results
    """
    import json
    
    with open(filename, 'r') as f:
        results = json.load(f)
    
    return results