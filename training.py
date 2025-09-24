# -*- coding: utf-8 -*-
"""
Training functions for different quantization methods
"""

import torch
from quantization import AlphaBlendLayer, anneal_alpha


def train_ste(model, train_loader, val_loader, criterion, optimizer, scheduler, 
              device, val_acc_list, train_loss_list, epochs):
    """
    Train model with Straight-Through Estimator quantization
    
    Args:
        model: STE quantized model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to run on
        val_acc_list: List to store validation accuracies
        train_loss_list: List to store training losses
        epochs: Number of training epochs
    """
    best_val_acc = 0.0
    best_model_path = "best_ste_model.pth"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"---- Learning Rate: {current_lr:.6f} -----")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(images), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        train_loss_list.append(avg_loss)
        
        val_acc = test(model, val_loader, device, criterion)
        val_acc_list.append(val_acc)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best STE model with validation accuracy: {val_acc:.2f}%")

    # Load the best model after training
    model.load_state_dict(torch.load(best_model_path))
    print("Loaded best STE model after training completed.")


def train_alpha_blend(model, train_loader, val_loader, criterion, optimizer, scheduler,
                      device, val_acc_list, train_loss_list, epochs):
    """
    Train model with Alpha Blending quantization
    
    Args:
        model: Alpha Blend quantized model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to run on
        val_acc_list: List to store validation accuracies
        train_loss_list: List to store training losses
        epochs: Number of training epochs
    """
    best_val_acc = 0.0
    best_model_path = "best_ab_model.pth"

    for epoch in range(epochs):
        # Update alpha for all alpha blend layers
        alpha = anneal_alpha(epoch, epochs)
        for module in model.modules():
            if isinstance(module, AlphaBlendLayer):
                module.alpha = alpha
        
        model.train()
        running_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"---- Learning Rate: {current_lr:.6f} -----")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(images), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        train_loss_list.append(avg_loss)
        
        val_acc = test(model, val_loader, device, criterion)
        val_acc_list.append(val_acc)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best AB model with validation accuracy: {val_acc:.2f}%")

    # Load the best model after training
    model.load_state_dict(torch.load(best_model_path))
    print("Loaded best AB model after training completed.")


def train_full_precision(model, train_loader, val_loader, criterion, optimizer, scheduler,
                         device, val_acc_list, train_loss_list, epochs):
    """
    Train full-precision model (baseline)
    
    Args:
        model: Full-precision model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to run on
        val_acc_list: List to store validation accuracies
        train_loss_list: List to store training losses
        epochs: Number of training epochs
    """
    best_val_acc = 0.0
    best_model_path = "best_fp_model.pth"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"---- Learning Rate: {current_lr:.6f} -----")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(images), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        train_loss_list.append(avg_loss)
        
        val_acc = test(model, val_loader, device, criterion)
        val_acc_list.append(val_acc)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best FP model with validation accuracy: {val_acc:.2f}%")

    # Load the best model after training
    model.load_state_dict(torch.load(best_model_path))
    print("Loaded best FP model after training completed.")


def test(model, test_loader, device, criterion):
    """
    Evaluate model on test/validation set
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        criterion: Loss function
    
    Returns:
        float: Test accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
    
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy