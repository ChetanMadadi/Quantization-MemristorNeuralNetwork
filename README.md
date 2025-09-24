# Memristor-Based CNN Quantization

This repository contains the implementation of quantization methods for convolutional neural networks using memristor conductance states. The code compares four different approaches on CIFAR-10 dataset using ResNet-18 architecture. To evaluate the approaches more critically, this project considers two different encoding schemes, which result in 2 different levels of quantization.

## Methods Implemented

1. **Straight-Through Estimator (STE)**: Quantizes weights during training using straight-through gradient estimation
2. **Alpha Blending**: Gradually transitions from full-precision to quantized weights during training.
3. **Full-Precision Baseline**: Standard training without quantization
4. **Post-Training Quantization (PTQ)**: Applies quantization after full-precision training

## Project Structure

```
├── config.py              # Configuration and memristor conductance states
├── quantization.py         # Quantization layers and functions
├── models.py              # Model definitions
├── data_utils.py          # Data loading and preprocessing
├── training.py            # Training functions for different methods
├── utils.py               # Utility functions
├── main.py                # Main experiment script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Memristor Conductance States

The quantization levels are based on actual memristor conductance measurements, containing:

- 100 positive conductance states
- 100 negative conductance states (symmetric)
- Total of 200 discrete quantization levels
- Normalized to range [-1, 1]

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Running Experiments

```bash
python main.py
```

This will:

1. Train all four models sequentially
2. Save the best model checkpoints during training
3. Evaluate all models on the test set
4. Print comparison results
5. Save final trained models

### Configuration

Modify `config.py` to change:

- Training hyperparameters (epochs, learning rate, etc.)
- Batch sizes
- Model architecture parameters
- Memristor conductance states

## Key Features

### Quantization Implementation

- **STEQuantizer**: Custom autograd function implementing straight-through estimation
- **SteLayer**: Wrapper for applying STE quantization to Conv2d/Linear layers
- **AlphaBlendLayer**: Implements gradual quantization using alpha blending
- **Post-training quantization**: Direct weight quantization after training

### Training Features

- Cosine annealing learning rate scheduling
- Data augmentation with RandAugment
- Automatic model checkpointing (saves best validation accuracy)
- Comprehensive logging and progress tracking

### Model Architecture

- ResNet-18 adapted for CIFAR-10 (32×32 input)
- Modified first convolution layer (3×3 kernel, stride=1)
- Removed max pooling for small input size
- 10-class output for CIFAR-10

## Results

The code will output comparison table showing test accuracies for all four methods:

```
Method                    Test Accuracy
----------------------------------------
STE                       XX.XX%
Alpha Blend              XX.XX%
Full-Precision           XX.XX%
Post-Training Quantized  XX.XX%
```

## File Descriptions

- **config.py**: Contains all configuration parameters and memristor conductance states
- **quantization.py**: Core quantization implementations (STE, Alpha Blend, PTQ)
- **models.py**: Model definitions for different quantization approaches
- **data_utils.py**: CIFAR-10 data loading with proper train/val/test splits
- **training.py**: Training loops for each quantization method
- **utils.py**: Utility functions for reproducibility and model management
- **main.py**: Main experiment script that runs all comparisons

## Citation

If you use this code in your research, please cite the corresponding paper.
