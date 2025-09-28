"""
Main script to run memristor-based quantization experiments
Compares STE, Alpha Blending, Baseline, and Post-Training Quantization methods
"""

import copy
import torch

from config import *
from models import SteModel, AlphaBlendModel, create_resnet18_model
from data_utils import get_data_loaders
from training import train_ste_model, train_alpha_blend_model, train_baseline_model, test
from utils import set_seed, get_device, create_optimizer_and_scheduler, save_results, get_criterion
from quantization import post_training_quantize


def run_experiment():
    """
    Run complete quantization experiment comparing all methods
    """
    print("="*60)
    print("MEMRISTOR-BASED QUANTIZATION EXPERIMENT")
    print("="*60)
    
    # Set random seed for reproducibility
    set_seed(RANDOM_SEED)
    
    # Get device and data loaders
    device = get_device()
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Get quantization levels from memristor conductance states
    quant_values = get_quantization_values()
    print(f"Using {len(quant_values)} quantization levels")
    
    # Loss function
    criterion = get_criterion()
    
    # Dictionary to store all results
    results = {
        'ste': {'val_acc': [], 'train_loss': [], 'test_acc': 0},
        'alpha_blend': {'val_acc': [], 'train_loss': [], 'test_acc': 0},
        'baseline': {'val_acc': [], 'train_loss': [], 'test_acc': 0},
        'ptq': {'test_acc': 0}
    }
    
    print("\n" + "="*60)
    print("1. TRAINING STE MODEL")
    print("="*60)
    
    # 1. STE Model
    ste_model = SteModel(quant_values, device, NUM_CLASSES).to(device)
    
    optimizer_ste, scheduler_ste = create_optimizer_and_scheduler(
        ste_model, BASE_LR, MOMENTUM, WEIGHT_DECAY, EPOCHS, LR_MIN_SCALER
    )
    
    val_acc_ste, train_loss_ste = train_ste_model(
        ste_model, train_loader, val_loader, criterion, optimizer_ste, 
        scheduler_ste, device, EPOCHS
    )
    
    test_acc_ste = test(ste_model, test_loader, device, criterion)
    results['ste']['val_acc'] = val_acc_ste
    results['ste']['train_loss'] = train_loss_ste
    results['ste']['test_acc'] = test_acc_ste
    
    print("\n" + "="*60)
    print("2. TRAINING ALPHA BLEND MODEL")
    print("="*60)
    
    # 2. Alpha Blend Model
    ab_model = AlphaBlendModel(quant_values, device, NUM_CLASSES).to(device)
    
    optimizer_ab, scheduler_ab = create_optimizer_and_scheduler(
        ab_model, BASE_LR, MOMENTUM, WEIGHT_DECAY, EPOCHS, LR_MIN_SCALER
    )
    
    val_acc_ab, train_loss_ab = train_alpha_blend_model(
        ab_model, train_loader, val_loader, criterion, optimizer_ab, 
        scheduler_ab, device, EPOCHS
    )
    
    test_acc_ab = test(ab_model, test_loader, device, criterion)
    results['alpha_blend']['val_acc'] = val_acc_ab
    results['alpha_blend']['train_loss'] = train_loss_ab
    results['alpha_blend']['test_acc'] = test_acc_ab
    
    print("\n" + "="*60)
    print("3. TRAINING BASELINE MODEL")
    print("="*60)
    
    # 3. Baseline Model (No Quantization)
    baseline_model = create_resnet18_model(num_classes=NUM_CLASSES).to(device)
    
    optimizer_baseline, scheduler_baseline = create_optimizer_and_scheduler(
        baseline_model, BASE_LR, MOMENTUM, WEIGHT_DECAY, EPOCHS, LR_MIN_SCALER
    )
    
    val_acc_baseline, train_loss_baseline = train_baseline_model(
        baseline_model, train_loader, val_loader, criterion, optimizer_baseline, 
        scheduler_baseline, device, EPOCHS
    )
    
    test_acc_baseline = test(baseline_model, test_loader, device, criterion)
    results['baseline']['val_acc'] = val_acc_baseline
    results['baseline']['train_loss'] = train_loss_baseline
    results['baseline']['test_acc'] = test_acc_baseline
    
    print("\n" + "="*60)
    print("4. POST-TRAINING QUANTIZATION")
    print("="*60)
    
    # 4. Post-Training Quantization
    ptq_model = copy.deepcopy(baseline_model)
    quant_values_tensor = torch.tensor(quant_values, dtype=torch.float32, device=device)
    post_training_quantize(ptq_model, quant_values_tensor)
    
    test_acc_ptq = test(ptq_model, test_loader, device, criterion)
    results['ptq']['test_acc'] = test_acc_ptq
    
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    
    print(f"STE Model Test Accuracy: {test_acc_ste:.2f}%")
    print(f"Alpha Blend Model Test Accuracy: {test_acc_ab:.2f}%")
    print(f"Baseline Model Test Accuracy: {test_acc_baseline:.2f}%")
    print(f"Post-Training Quantized Model Test Accuracy: {test_acc_ptq:.2f}%")
    
    # Save models
    print("\nSaving trained models...")
    torch.save(ste_model.state_dict(), 'ste_model.pth')
    torch.save(ab_model.state_dict(), 'alpha_blend_model.pth')
    torch.save(baseline_model.state_dict(), 'baseline_model.pth')
    torch.save(ptq_model.state_dict(), 'ptq_model.pth')
    
    # Save results
    save_results(results, 'experiment_results.json')
    
    return results


if __name__ == "__main__":
    results = run_experiment()