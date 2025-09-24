# -*- coding: utf-8 -*-
"""
Main script for memristor-based CNN quantization experiments
Compares four different approaches:
1. Straight-Through Estimator (STE) quantization
2. Alpha Blending quantization
3. Full-precision baseline
4. Post-Training Quantization (PTQ)
"""

import copy
import torch
from config import EPOCHS, RANDOM_SEED, get_quantization_values
from data_utils import get_data_loaders
from models import SteModel, AlphaBlendModel, create_resnet18_model
from training import train_ste, train_alpha_blend, train_full_precision, test
from quantization import post_training_quantize
from utils import set_seed, get_device, create_optimizer_and_scheduler, get_criterion, print_model_summary


def main():
    """
    Main experiment function
    """
    # Set random seed for reproducibility
    set_seed(RANDOM_SEED)
    
    # Get device and data loaders
    device = get_device()
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Get quantization values from memristor conductance states
    quant_values = get_quantization_values()
    print(f"Number of quantization levels: {len(quant_values)}")
    
    # Get loss function
    criterion = get_criterion()
    
    # Initialize results storage
    results = {
        'ste': {'val_acc': [0], 'train_loss': []},
        'alpha_blend': {'val_acc': [0], 'train_loss': []},
        'full_precision': {'val_acc': [0], 'train_loss': []},
        'ptq': {'test_acc': None}
    }
    
    print("="*80)
    print("MEMRISTOR-BASED CNN QUANTIZATION EXPERIMENTS")
    print("="*80)
    
    # 1. Train STE Model
    print("\n1. Training STE (Straight-Through Estimator) Model")
    print("-" * 50)
    
    ste_model = SteModel(quant_values, device).to(device)
    print_model_summary(ste_model, "STE Model")
    
    ste_optimizer, ste_scheduler = create_optimizer_and_scheduler(ste_model, EPOCHS)
    
    train_ste(
        ste_model, train_loader, val_loader, criterion, 
        ste_optimizer, ste_scheduler, device,
        results['ste']['val_acc'], results['ste']['train_loss'], EPOCHS
    )
    
    # 2. Train Alpha Blend Model
    print("\n2. Training Alpha Blend Model")
    print("-" * 50)
    
    ab_model = AlphaBlendModel(quant_values, device).to(device)
    print_model_summary(ab_model, "Alpha Blend Model")
    
    ab_optimizer, ab_scheduler = create_optimizer_and_scheduler(ab_model, EPOCHS)
    
    train_alpha_blend(
        ab_model, train_loader, val_loader, criterion,
        ab_optimizer, ab_scheduler, device,
        results['alpha_blend']['val_acc'], results['alpha_blend']['train_loss'], EPOCHS
    )
    
    # 3. Train Full-Precision Model (Baseline)
    print("\n3. Training Full-Precision Baseline Model")
    print("-" * 50)
    
    fp_model = create_resnet18_model().to(device)
    print_model_summary(fp_model, "Full-Precision Model")
    
    fp_optimizer, fp_scheduler = create_optimizer_and_scheduler(fp_model, EPOCHS)
    
    train_full_precision(
        fp_model, train_loader, val_loader, criterion,
        fp_optimizer, fp_scheduler, device,
        results['full_precision']['val_acc'], results['full_precision']['train_loss'], EPOCHS
    )
    
    # 4. Create Post-Training Quantized Model
    print("\n4. Creating Post-Training Quantized Model")
    print("-" * 50)
    
    ptq_model = copy.deepcopy(fp_model)
    quant_values_tensor = torch.tensor(quant_values, dtype=torch.float32, device=device)
    post_training_quantize(ptq_model, quant_values_tensor)
    print_model_summary(ptq_model, "Post-Training Quantized Model")
    
    # Test all models
    print("\n" + "="*80)
    print("FINAL TEST RESULTS")
    print("="*80)
    
    print("\nSTE Model Test Results:")
    ste_test_acc = test(ste_model, test_loader, device, criterion)
    
    print("\nAlpha Blend Model Test Results:")
    ab_test_acc = test(ab_model, test_loader, device, criterion)
    
    print("\nFull-Precision Model Test Results:")
    fp_test_acc = test(fp_model, test_loader, device, criterion)
    
    print("\nPost-Training Quantized Model Test Results:")
    ptq_test_acc = test(ptq_model, test_loader, device, criterion)
    
    # Store test accuracies
    results['ste']['test_acc'] = ste_test_acc
    results['alpha_blend']['test_acc'] = ab_test_acc
    results['full_precision']['test_acc'] = fp_test_acc
    results['ptq']['test_acc'] = ptq_test_acc
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'Test Accuracy':<15}")
    print("-" * 40)
    print(f"{'STE':<25} {ste_test_acc:<15.2f}%")
    print(f"{'Alpha Blend':<25} {ab_test_acc:<15.2f}%")
    print(f"{'Full-Precision':<25} {fp_test_acc:<15.2f}%")
    print(f"{'Post-Training Quantized':<25} {ptq_test_acc:<15.2f}%")
    
    # Save models
    print("\nSaving trained models...")
    torch.save(ste_model.state_dict(), 'ste_model_final.pth')
    torch.save(ab_model.state_dict(), 'ab_model_final.pth')
    torch.save(fp_model.state_dict(), 'fp_model_final.pth')
    torch.save(ptq_model.state_dict(), 'ptq_model_final.pth')
    
    print("All models saved successfully!")
    
    return results


if __name__ == "__main__":
    results = main()