# -*- coding: utf-8 -*-
"""
Model definitions for memristor-based CNN quantization
"""

import torch
import torch.nn as nn
import torchvision.models as models
from quantization import replace_layers_with_ste, replace_layers_with_alpha_blend


def create_resnet18_model(num_classes=10):
    """
    Create ResNet-18 model adapted for CIFAR-10
    
    Args:
        num_classes: Number of output classes
    
    Returns:
        Modified ResNet-18 model
    """
    model = models.resnet18(weights=None)
    # Modify for CIFAR-10 (32x32 input)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class SteModel(nn.Module):
    """
    ResNet-18 model with Straight-Through Estimator quantization
    """
    
    def __init__(self, quant_values, device, num_classes=10):
        super(SteModel, self).__init__()
        quant_values = torch.tensor(quant_values, dtype=torch.float32, device=device)
        
        self.model = create_resnet18_model(num_classes)
        replace_layers_with_ste(self.model, quant_values)

    def forward(self, x):
        return self.model(x)


class AlphaBlendModel(nn.Module):
    """
    ResNet-18 model with Alpha Blending quantization
    """
    
    def __init__(self, quant_values, device, num_classes=10):
        super(AlphaBlendModel, self).__init__()
        quant_values = torch.tensor(quant_values, dtype=torch.float32, device=device)
        self.model = create_resnet18_model(num_classes)
        replace_layers_with_alpha_blend(self.model, quant_values, alpha=0.0)

    def forward(self, x):
        return self.model(x)