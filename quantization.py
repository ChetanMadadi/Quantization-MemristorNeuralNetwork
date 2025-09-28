
"""
Quantization layers and functions for memristor-based CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STEQuantizer(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) quantization function
    Forward pass quantizes to nearest memristor conductance state
    Backward pass uses straight-through gradient estimation
    """
    
    @staticmethod
    def forward(ctx, input, quant_values):
        quant_values = quant_values.to(input.device)
        input_flat = input.view(-1, 1)

        # Vectorized nearest neighbor search
        idx = torch.searchsorted(quant_values, input_flat)
        idx = idx.clamp(0, len(quant_values) - 1)

        # Get left and right neighbors safely
        left_idx = (idx - 1).clamp(min=0)
        right_idx = idx

        left = quant_values[left_idx]
        right = quant_values[right_idx]

        # Choose the closest one
        closer_to_left = (input_flat - left).abs() < (input_flat - right).abs()
        final_idx = torch.where(closer_to_left, left_idx, right_idx)

        output = quant_values[final_idx].view_as(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradients unchanged
        return grad_output, None


class SteLayer(nn.Module):
    """
    Layer wrapper for Straight-Through Estimator quantization
    """
    
    def __init__(self, module, quant_values):
        super().__init__()
        self.module = module
        self.quant_values = quant_values

    def forward(self, x):
        w_fp = self.module.weight
        quant_weight = STEQuantizer.apply(w_fp, self.quant_values)

        if isinstance(self.module, nn.Conv2d):
            return nn.functional.conv2d(
                x, quant_weight, self.module.bias,
                stride=self.module.stride,
                padding=self.module.padding,
                dilation=self.module.dilation,
                groups=self.module.groups
            )
        elif isinstance(self.module, nn.Linear):
            return nn.functional.linear(x, quant_weight, self.module.bias)
        else:
            raise NotImplementedError


class AlphaBlendLayer(nn.Module):
    """
    Alpha blending layer for gradual quantization
    Blends between full-precision and quantized weights
    """
    
    def __init__(self, module, quant_values, alpha=0.0):
        super().__init__()
        self.module = module
        self.quant_values = quant_values
        self.alpha = alpha

    def forward(self, x):
        w_fp = self.module.weight
        q_w = STEQuantizer.apply(w_fp, self.quant_values).detach()  # No gradient flow here
        w_blend = self.alpha * q_w + (1 - self.alpha) * w_fp

        if isinstance(self.module, nn.Conv2d):
            return nn.functional.conv2d(
                x, w_blend, self.module.bias,
                stride=self.module.stride,
                padding=self.module.padding,
                dilation=self.module.dilation,
                groups=self.module.groups
            )
        elif isinstance(self.module, nn.Linear):
            return nn.functional.linear(x, w_blend, self.module.bias)
        else:
            raise NotImplementedError


def quantize_tensor(input, quant_values):
    """
    Quantize tensor to nearest memristor conductance states
    Used for post-training quantization
    
    Args:
        input: Input tensor to quantize
        quant_values: Available quantization values
    
    Returns:
        Quantized tensor
    """
    input_flat = input.view(-1, 1)
    quant_values = quant_values.to(input.device)  
    idx = torch.searchsorted(quant_values, input_flat)
    idx = torch.clamp(idx, 0, len(quant_values) - 1)

    left_idx = (idx - 1).clamp(min=0)
    right_idx = idx

    left = quant_values[left_idx]
    right = quant_values[right_idx]

    # Choose the closest one
    closer_to_left = (input_flat - left).abs() < (input_flat - right).abs()
    final_idx = torch.where(closer_to_left, left_idx, right_idx)

    return quant_values[final_idx].view_as(input)


def post_training_quantize(model, quant_values):
    """
    Apply post-training quantization to model weights
    
    Args:
        model: PyTorch model to quantize
        quant_values: Available quantization values
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                module.weight.data = quantize_tensor(module.weight.data, quant_values)


def replace_layers_with_ste(model, quant_values):
    """
    Replace Conv2d and Linear layers with STE quantized versions
    
    Args:
        model: PyTorch model
        quant_values: Available quantization values
    
    Returns:
        Model with STE quantized layers
    """
    for name, module in model.named_children():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            setattr(model, name, SteLayer(module, quant_values))
        else:
            replace_layers_with_ste(module, quant_values)


def replace_layers_with_alpha_blend(model, quant_values, alpha=0.0):
    """
    Replace Conv2d and Linear layers with alpha blending versions
    
    Args:
        model: PyTorch model
        quant_values: Available quantization values
        alpha: Blending factor (0 = full precision, 1 = fully quantized)
    """
    for name, module in model.named_children():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            setattr(model, name, AlphaBlendLayer(module, quant_values, alpha))
        else:
            replace_layers_with_alpha_blend(module, quant_values, alpha)


def anneal_alpha(epoch, total_epochs):
    """
    Linear annealing schedule for alpha blending
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
    
    Returns:
        Alpha value for current epoch
    """
    return min(1.0, epoch / total_epochs)