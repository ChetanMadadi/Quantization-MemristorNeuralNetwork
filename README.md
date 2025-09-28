# Memristor-Based Neural Network Quantization

This repository contains the implementation of memristor-based quantization methods for neural networks, comparing four different approaches:

1. **STE (Straight Through Estimator)** - Quantization-aware training with gradient pass-through
2. **Alpha Blending** - Progressive quantization with linear annealing
3. **Baseline** - Standard full-precision training
4. **PTQ (Post-Training Quantization)** - Naive quantization after training

The implementation also includes comprehensive robustness testing to evaluate model performance under weight noise conditions.

## Overview

The quantization levels are derived from actual memristor conductance states (100 positive + 100 negative states), normalized to the range [-1, 1]. The experiments use ResNet-18 on CIFAR-10 dataset.

For robustness testing, variable quantization levels are generated based on differences between conductance states, providing a more comprehensive evaluation of model stability.

## File Structure

```
├── config.py                       # Configuration and hyperparameters
├── quantization_layers.py          # Core quantization layer implementations
├── models.py                       # Model definitions for all approaches
├── data_utils.py                   # Data loading and preprocessing
├── training.py                     # Training and evaluation functions
├── utils.py                        # Utility functions
├── main.py                         # Main training experiment script
├── robustness_testing.py           # Robustness analysis functions
├── visualization.py                # Plotting and visualization utilities
├── run_robustness_experiment.py    # Robustness testing script
├── run_complete_experiment.py      # Full experiment pipeline
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Quick Start

### Option 1: Run Complete Experiment Pipeline (Recommended)

Run both training and robustness testing with comprehensive reporting:

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete experiment
python run_complete_experiment.py

# Or with custom settings
python run_complete_experiment.py --output-dir my_results --robustness-repeats 5
```

### Option 2: Run Individual Components

**Training only:**

```bash
python main.py
```

**Robustness testing only** (requires trained models):

```bash
python run_robustness_experiment.py
```

### Option 3: Quick Test

For quick testing with reduced parameters:

```bash
python run_complete_experiment.py --quick
```

## Results

The complete experiment generates:

- **Model Files**: `ste_model.pth`, `alpha_blend_model.pth`, `baseline_model.pth`, `ptq_model.pth`
- **Training Results**: `experiment_results.pth` - Training history and test accuracies
- **Robustness Results**: `robustness_experiment_results.pth` - Noise robustness analysis
- **Combined Results**: `combined_experiment_results.pth` - All results in one file
- **Visualizations**:
  - `training_curves.png` - Training loss and validation accuracy
  - `test_accuracy_comparison.png` - Test accuracy comparison
  - `robustness_analysis.png` - Accuracy vs noise level
  - `robustness_metrics.png` - Robustness comparison bar chart

## Key Components

### Quantization Methods

#### 1. Straight Through Estimator (STE)

- Forward pass: Quantize weights to nearest memristor state
- Backward pass: Pass gradients unchanged
- Implemented in `STEQuantizer` class

#### 2. Alpha Blending

- Blends full-precision and quantized weights: `w = α * w_quant + (1-α) * w_fp`
- Linear annealing: α increases from 0 to 1 during training
- Gradients flow only through full-precision path. so as alpha increases, the gradient passing from w_fp becomes smaller.

#### 3. Post-Training Quantization

- Quantizes trained baseline model weights without retraining
- Simple nearest-neighbor mapping to quantization levels

### Robustness Testing

The robustness testing evaluates model performance under Gaussian weight noise:

- **Noise Levels**: 0% to 20% in 1% increments
- **Noise Model**: Gaussian noise with (mean = 0) and (std deviation = percentage × |weight| )
- **Repetitions**: 10 runs per noise level (configurable)
- **Metrics**: Mean accuracy, standard deviation, accuracy drops

### Model Architecture

- **Base Model:** ResNet-18 adapted for CIFAR-10
  - Modified first conv layer: 3×3 kernel, stride=1, padding=1
  - Removed max pooling
  - 10-class output for CIFAR-10

### Quantization Levels

#### Fixed Quantization (Training)

- **Source:** Real memristor conductance measurements
- **Range:** 200 total levels (100 positive + 100 negative)
- **Normalization:** Scaled to [-1, 1] range

#### Variable Quantization (Robustness Testing)

- **Source:** Differences between all conductance pairs
- **Range:** ~10,000 unique levels
- **Purpose:** Tests robustness under more diverse quantization

## Configuration

Key hyperparameters in `config.py`:

```python
EPOCHS = 50
BASE_LR = 0.1
BATCH_SIZE = 64
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
SEED = 40  # For reproducibility
```

## Command Line Options

The complete experiment script supports various options:

```bash
python run_complete_experiment.py [OPTIONS]

Options:
  --output-dir DIR          Output directory (default: experiment_results)
  --skip-training          Skip training phase, use existing models
  --skip-robustness        Skip robustness testing phase
  --robustness-repeats N   Number of repetitions for robustness (default: 10)
  --quick                  Run quick test with reduced parameters
```

## Expected Results

The implementation reproduces the methodology from the research paper:

### Training Results

- **STE**: Quantization-aware training with competitive accuracy
- **Alpha Blending**: Progressive quantization approach
- **Baseline**: Full-precision upper bound
- **PTQ**: Simple post-training quantization baseline

### Robustness Results

- Comprehensive noise robustness analysis
- Comparison of quantization methods under noise
- Identification of most robust quantization approach

## Usage Notes

- **Reproducibility:** Fixed random seed (40) for consistent results
- **Device Support:** Automatic CUDA/CPU detection
- **Data:** CIFAR-10 automatically downloaded on first run
- **Memory:** Models and results saved automatically
- **Monitoring:** Progress printed during training and testing
- **Visualization:** Automatic plot generation if matplotlib available

## Research Context

This implementation corresponds to the quantization methodology presented in the associated research paper on memristor-based neural network quantization. The memristor conductance states represent realistic device limitations and provide a hardware-aware quantization scheme.

The robustness testing evaluates the practical viability of these quantization methods under realistic noise conditions that may occur in memristor-based hardware implementations.

## Citation

If you use this code in your research, please cite the corresponding paper:

```
[Paper citation to be added]
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in `config.py`
2. **Missing models for robustness testing**: Run training first or use `--skip-robustness`
3. **Visualization errors**: Install matplotlib or run without visualization
4. **Long runtime**: Use `--quick` for testing or reduce `--robustness-repeats`

### Performance Tips

- Use GPU for faster training
- Reduce `robustness_repeats` for quicker robustness testing
- Skip phases you don't need using command line flags
