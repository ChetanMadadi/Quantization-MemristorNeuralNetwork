# -*- coding: utf-8 -*-
"""
Data loading and preprocessing utilities
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from config import CIFAR10_MEAN, CIFAR10_STD, BATCH_SIZE, VAL_BATCH_SIZE, TRAIN_VAL_SPLIT


class TransformDataset(Dataset):
    """
    Wrapper to apply transform to a subset
    """
    
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y

    def __len__(self):
        return len(self.subset)


def get_transforms():
    """
    Get data transforms for training and validation/test
    
    Returns:
        tuple: (train_transform, test_val_transform)
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.autoaugment.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    test_val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])
    
    return train_transform, test_val_transform


def get_data_loaders():
    """
    Create data loaders for CIFAR-10 dataset
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_transform, test_val_transform = get_transforms()
    
    # Load datasets
    train_val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=test_val_transform, download=True
    )

    # Split training data into train and validation
    train_size = int(TRAIN_VAL_SPLIT * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_subset, val_subset = random_split(train_val_dataset, [train_size, val_size])

    # Apply transforms to subsets
    train_dataset = TransformDataset(train_subset, train_transform)
    val_dataset = TransformDataset(val_subset, test_val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader