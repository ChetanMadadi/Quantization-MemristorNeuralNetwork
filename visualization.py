"""
Visualization utilities for quantization experiments
Creates plots for training curves and robustness analysis
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(results, save_path=None):
    """
    Plot training loss and validation accuracy curves for all models.
    
    Args:
        results: Results dictionary from main experiment
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot training loss
    for model_name in ['ste', 'alpha_blend', 'baseline']:
        if model_name in results and 'train_loss' in results[model_name]:
            epochs = range(1, len(results[model_name]['train_loss']) + 1)
            ax1.plot(epochs, results[model_name]['train_loss'], 
                    label=f'{model_name.replace("_", " ").title()}', marker='o', markersize=3)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot validation accuracy
    for model_name in ['ste', 'alpha_blend', 'baseline']:
        if model_name in results and 'val_acc' in results[model_name]:
            epochs = range(0, len(results[model_name]['val_acc']))
            ax2.plot(epochs, results[model_name]['val_acc'], 
                    label=f'{model_name.replace("_", " ").title()}', marker='o', markersize=3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def plot_robustness_analysis(robustness_results, save_path=None):
    """
    Plot robustness analysis results showing accuracy vs noise level.
    
    Args:
        robustness_results: Results from run_robustness_analysis
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    colors = {'ste': 'blue', 'alpha_blend': 'red', 'baseline': 'green', 'ptq': 'orange'}
    markers = {'ste': 'o', 'alpha_blend': 's', 'baseline': '^', 'ptq': 'd'}
    labels = {'ste': 'STE', 'alpha_blend': 'Alpha Blend', 'baseline': 'Baseline', 'ptq': 'PTQ'}
    
    for model_name, results in robustness_results.items():
        noise_levels_percentage = [level * 100 for level in results['noise_levels']]
        mean_accs = results['mean_accuracies']
        std_accs = results['std_accuracies']
        
        color = colors.get(model_name, 'black')
        marker = markers.get(model_name, 'o')
        label = labels.get(model_name, model_name.title())
        
        # Plot mean accuracy with error bars
        plt.errorbar(noise_levels_percentage, mean_accs, yerr=std_accs,
                    label=f'{label}', marker=marker, color=color,
                    capsize=3, capthick=1, markersize=4, linewidth=2)
    
    plt.xlabel('Noise Level (%)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Robustness: Accuracy vs Weight Noise Level', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 20)
    
    # Add baseline reference lines
    if 'baseline' in robustness_results:
        baseline_acc = robustness_results['baseline']['mean_accuracies'][0]
        plt.axhline(y=baseline_acc, color='gray', linestyle='--', alpha=0.5, 
                   label=f'Baseline Accuracy ({baseline_acc:.1f}%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Robustness plot saved to {save_path}")
    
    plt.show()


def plot_robustness_comparison_bar(robustness_metrics, save_path=None):
    """
    Create a bar plot comparing robustness metrics across models.
    
    Args:
        robustness_metrics: Results from calculate_robustness_metrics
        save_path: Optional path to save the plot
    """
    models = list(robustness_metrics.keys())
    metrics = ['accuracy_drop_5%', 'accuracy_drop_10%', 'accuracy_drop_20%']
    metric_labels = ['5% Noise', '10% Noise', '20% Noise']
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [robustness_metrics[model][metric] for model in models]
        ax.bar(x + i * width, values, width, label=label, alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy Drop (%)')
    ax.set_title('Robustness Comparison: Accuracy Drop at Different Noise Levels')
    ax.set_xticks(x + width)
    ax.set_xticklabels([model.replace('_', ' ').title() for model in models])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [robustness_metrics[model][metric] for model in models]
        for j, v in enumerate(values):
            ax.text(j + i * width, v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Robustness comparison saved to {save_path}")
    
    plt.show()


def plot_test_accuracy_comparison(results, robustness_results=None, save_path=None):
    """
    Create a bar plot comparing test accuracies of all models.
    
    Args:
        results: Results from main experiment
        robustness_results: Optional robustness results for noise-free accuracy
        save_path: Optional path to save the plot
    """
    models = []
    accuracies = []
    
    # Extract test accuracies from main results
    model_labels = {'ste': 'STE', 'alpha_blend': 'Alpha Blend', 'baseline': 'Baseline', 'ptq': 'PTQ'}
    
    for model_name, label in model_labels.items():
        if model_name in results and 'test_acc' in results[model_name]:
            models.append(label)
            accuracies.append(results[model_name]['test_acc'])
    
    # If robustness results are available, use those for more precise values
    if robustness_results:
        models_robust = []
        accuracies_robust = []
        for model_name, label in model_labels.items():
            if model_name in robustness_results:
                models_robust.append(label)
                accuracies_robust.append(robustness_results[model_name]['mean_accuracies'][0])
        
        if models_robust:
            models, accuracies = models_robust, accuracies_robust
    
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green', 'orange'][:len(models)]
    bars = plt.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Test Accuracy Comparison Across Models', fontsize=14)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Test accuracy comparison saved to {save_path}")
    
    plt.show()


def create_comprehensive_report(results, robustness_results, robustness_metrics, save_dir=None):
    """
    Create a comprehensive visual report of all experiments.
    
    Args:
        results: Main experiment results
        robustness_results: Robustness analysis results
        robustness_metrics: Calculated robustness metrics
        save_dir: Directory to save plots (optional)
    """
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    print("Creating comprehensive visual report...")
    
    # 1. Training curves
    plot_training_curves(results, 
                        save_path=f"{save_dir}/training_curves.png" if save_dir else None)
    
    # 2. Test accuracy comparison
    plot_test_accuracy_comparison(results, robustness_results,
                                 save_path=f"{save_dir}/test_accuracy_comparison.png" if save_dir else None)
    
    # 3. Robustness analysis
    plot_robustness_analysis(robustness_results,
                           save_path=f"{save_dir}/robustness_analysis.png" if save_dir else None)
    
    # 4. Robustness metrics comparison
    plot_robustness_comparison_bar(robustness_metrics,
                                  save_path=f"{save_dir}/robustness_metrics.png" if save_dir else None)
    
    print("Visual report completed!")
    if save_dir:
        print(f"All plots saved to {save_dir}/")


def set_plot_style():
    """
    Set consistent plotting style for all figures.
    """
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10