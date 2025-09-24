"""
Configuration file for memristor-based CNN quantization experiments
"""

import numpy as np

# Training Configuration
EPOCHS = 50
BASE_LR = 0.1
LR_MIN_SCALER = 0.01
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
BATCH_SIZE = 64
VAL_BATCH_SIZE = 1000
TRAIN_VAL_SPLIT = 0.9
RANDOM_SEED = 40
ENCODING_SCHEME = 'single_memristor_encoding'  # Options: 'single_memristor_encoding', 'dual_memristor_encoding'

# Model Configuration
NUM_CLASSES = 10

# Memristor Conductance States
WEIGHT_VALUES = [
    2.51829624e-06, 3.72715294e-06, 5.15766442e-06, 6.78002834e-06,
    8.52160156e-06, 1.05501363e-05, 1.20569021e-05, 1.37612224e-05,
    1.54674053e-05, 1.71978027e-05, 1.88592821e-05, 2.05431134e-05,
    2.21747905e-05, 2.38902867e-05, 2.55443156e-05, 2.71685421e-05,
    2.89119780e-05, 3.05920839e-05, 3.23131680e-05, 3.39020044e-05,
    3.54871154e-05, 3.71560454e-05, 3.88976187e-05, 4.06000763e-05,
    4.20231372e-05, 4.37367707e-05, 4.54373658e-05, 4.72106040e-05,
    4.88851219e-05, 5.06695360e-05, 5.22565097e-05, 5.39645553e-05,
    5.58234751e-05, 5.74663281e-05, 5.90961426e-05, 6.08209521e-05,
    6.24023378e-05, 6.39092177e-05, 6.54719770e-05, 6.73215836e-05,
    6.87241554e-05, 7.05458224e-05, 7.21868128e-05, 7.36229122e-05,
    7.51297921e-05, 7.67670572e-05, 7.82366842e-05, 7.99763948e-05,
    8.17514956e-05, 8.33496451e-05, 8.50055367e-05, 8.62907618e-05,
    8.78404826e-05, 8.93604010e-05, 9.06307250e-05, 9.19308513e-05,
    9.32049006e-05, 9.49427485e-05, 9.64347273e-05, 9.78037715e-05,
    9.96198505e-05, 1.01117417e-04, 1.02177262e-04, 1.03591010e-04,
    1.05306506e-04, 1.06660649e-04, 1.07808039e-04, 1.09545887e-04,
    1.10885128e-04, 1.12274662e-04, 1.13626942e-04, 1.15199015e-04,
    1.16379932e-04, 1.17568299e-04, 1.18700787e-04, 1.20515004e-04,
    1.21075660e-04, 1.22969970e-04, 1.24694780e-04, 1.26129016e-04,
    1.27758831e-04, 1.29126012e-04, 1.30319968e-04, 1.31307170e-04,
    1.32871792e-04, 1.33739784e-04, 1.33996829e-04, 1.33790076e-04,
    1.32936984e-04, 1.32476911e-04, 1.32355839e-04, 1.35647133e-04,
    1.34376809e-04, 1.35060400e-04, 1.35516748e-04, 1.36040151e-04,
    1.37157738e-04, 1.37412921e-04, 1.39065087e-04, 1.39588490e-04
]

def get_quantization_values():
    """
    Process memristor conductance states to create quantization values
    
    Returns:
        np.ndarray: Normalized quantization values in range [-1, 1]
    """
    # Create symmetric values (positive and negative)
    dfe = (
        [x for x in WEIGHT_VALUES] + [-x for x in WEIGHT_VALUES]
        if ENCODING_SCHEME == 'single_memristor_encoding'
        else [x - y for x in WEIGHT_VALUES for y in WEIGHT_VALUES]
    )
    dfe = np.array(dfe)
    
    # Normalize to range [-1, 1]
    dfe_current_min = np.min(dfe)
    dfe_current_max = np.max(dfe)
    dfe_current_normalized = 2 * (dfe - dfe_current_min) / (dfe_current_max - dfe_current_min) - 1
    dfe_current_normalized.sort()
    
    return dfe_current_normalized

# CIFAR-10 Dataset Statistics
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]