"""
Complete experimental pipeline that runs both training and robustness analysis
This is the main entry point for reproducing all paper results
"""

import os
from main import run_experiment
from run_robustness_experiment import run_robustness_experiment
from visualization import create_comprehensive_report, set_plot_style
from utils import save_results
import torch


def run_complete_experiment(output_dir="experiment_results", 
                           skip_training=False, 
                           skip_robustness=False,
                           robustness_repeats=10):
    """
    Run the complete experimental pipeline including training and robustness testing.
    
    Args:
        output_dir: Directory to save all results and plots
        skip_training: If True, skip training and load existing models
        skip_robustness: If True, skip robustness testing
        robustness_repeats: Number of repetitions for robustness testing
    """
    print("="*80)
    print("COMPLETE MEMRISTOR-BASED QUANTIZATION EXPERIMENT PIPELINE")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    original_dir = os.getcwd()
    os.chdir(output_dir)
    
    try:
        set_plot_style()
        
        # Phase 1: Training Experiment
        if not skip_training:
            print("\n" + "="*60)
            print("PHASE 1: TRAINING EXPERIMENT")
            print("="*60)
            
            training_results = run_experiment()
            
            if training_results is None:
                print("Training experiment failed. Aborting complete experiment.")
                return None
                
        else:
            print("\n" + "="*60)
            print("PHASE 1: SKIPPING TRAINING (Loading existing results)")
            print("="*60)
            
            try:
                training_results = torch.load('experiment_results.json')
                print("Loaded existing training results")
            except FileNotFoundError:
                print("No existing training results found. Please run training first.")
                return None
        
        # Phase 2: Robustness Testing
        if not skip_robustness:
            print("\n" + "="*60)
            print("PHASE 2: ROBUSTNESS TESTING")
            print("="*60)
            
            robustness_results, robustness_metrics = run_robustness_experiment(
                repeats=robustness_repeats
            )
            
            if robustness_results is None:
                print("Robustness experiment failed.")
                robustness_results = {}
                robustness_metrics = {}
        else:
            print("\n" + "="*60)
            print("PHASE 2: SKIPPING ROBUSTNESS TESTING")
            print("="*60)
            robustness_results = {}
            robustness_metrics = {}
        
        # Phase 3: Comprehensive Reporting
        print("\n" + "="*60)
        print("PHASE 3: GENERATING COMPREHENSIVE REPORT")
        print("="*60)
        
        try:
            create_comprehensive_report(training_results, robustness_results, robustness_metrics)
            print("Comprehensive visual report generated successfully!")
        except Exception as e:
            print(f"Error generating comprehensive report: {e}")
        
        # Phase 4: Save Combined Results
        print("\n" + "="*60)
        print("PHASE 4: SAVING COMBINED RESULTS")
        print("="*60)
        
        combined_results = {
            'training_results': training_results,
            'robustness_results': robustness_results,
            'robustness_metrics': robustness_metrics,
            'experiment_config': {
                'skip_training': skip_training,
                'skip_robustness': skip_robustness,
                'robustness_repeats': robustness_repeats
            }
        }
        
        save_results(combined_results, 'combined_experiment_results.pth')
        
        # Print final summary
        print_experiment_summary(training_results, robustness_metrics)
        
        print(f"\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"All results saved in: {os.path.abspath('.')}")
        print("Files generated:")
        print("  - combined_experiment_results.pth (all results)")
        print("  - ste_model.pth, alpha_blend_model.pth, baseline_model.pth, ptq_model.pth (models)")
        print("  - training_curves.png, test_accuracy_comparison.png (training plots)")
        print("  - robustness_analysis.png, robustness_metrics.png (robustness plots)")
        
        return combined_results
        
    except Exception as e:
        print(f"Error in complete experiment: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        os.chdir(original_dir)


def print_experiment_summary(training_results, robustness_metrics):
    """
    Print a comprehensive summary of all experimental results.
    
    Args:
        training_results: Results from training experiment
        robustness_metrics: Robustness metrics from robustness testing
    """
    print(f"\n" + "="*60)
    print("FINAL EXPERIMENT SUMMARY")
    print("="*60)
    
    print("\n1. TRAINING RESULTS:")
    print("-" * 30)
    if training_results:
        for model_name in ['ste', 'alpha_blend', 'baseline', 'ptq']:
            if model_name in training_results and 'test_acc' in training_results[model_name]:
                acc = training_results[model_name]['test_acc']
                print(f"  {model_name.replace('_', ' ').title():<15}: {acc:.2f}%")
    
    print("\n2. ROBUSTNESS RESULTS:")
    print("-" * 30)
    if robustness_metrics:
        print(f"{'Model':<12} {'Baseline':<10} {'5% Noise':<10} {'10% Noise':<10} {'20% Noise':<10}")
        print("-" * 52)
        for model_name, metrics in robustness_metrics.items():
            print(f"{model_name.title():<12} {metrics['baseline_accuracy']:<10.2f} "
                  f"{metrics['accuracy_drop_5%']:<10.2f} {metrics['accuracy_drop_10%']:<10.2f} "
                  f"{metrics['accuracy_drop_20%']:<10.2f}")
    
    print("\n3. KEY FINDINGS:")
    print("-" * 30)
    if training_results and robustness_metrics:
        # Find best performing model in clean conditions
        best_clean = max(training_results.keys(), 
                        key=lambda x: training_results[x].get('test_acc', 0) 
                        if x != 'ptq' else training_results[x].get('test_acc', 0))
        
        # Find most robust model (lowest accuracy drop at 10% noise)
        if robustness_metrics:
            most_robust = min(robustness_metrics.keys(), 
                            key=lambda x: robustness_metrics[x]['accuracy_drop_10%'])
            
            print(f"  • Best clean performance: {best_clean.replace('_', ' ').title()}")
            print(f"  • Most robust to noise: {most_robust.replace('_', ' ').title()}")
            
            # Calculate average robustness
            avg_drops = {name: metrics['accuracy_drop_10%'] 
                        for name, metrics in robustness_metrics.items()}
            print(f"  • Average 10% noise drop: {sum(avg_drops.values())/len(avg_drops):.2f}%")


def quick_experiment():
    """
    Run a quick version of the experiment with reduced parameters for testing.
    """
    print("Running quick experiment for testing...")
    return run_complete_experiment(
        output_dir="quick_test_results",
        robustness_repeats=3
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete memristor quantization experiment')
    parser.add_argument('--output-dir', default='experiment_results', 
                       help='Directory to save results (default: experiment_results)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training phase and use existing models')
    parser.add_argument('--skip-robustness', action='store_true',
                       help='Skip robustness testing phase')
    parser.add_argument('--robustness-repeats', type=int, default=10,
                       help='Number of repetitions for robustness testing (default: 10)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test version with reduced parameters')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_experiment()
    else:
        run_complete_experiment(
            output_dir=args.output_dir,
            skip_training=args.skip_training,
            skip_robustness=args.skip_robustness,
            robustness_repeats=args.robustness_repeats
        )