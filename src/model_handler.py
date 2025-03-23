"""
model_handler.py

Contains functions to load, fine-tune, save, and reload a pretrained deep learning model.
"""

import torch
import torch.nn as nn
from torchvision import models


def get_device():
    """
    Returns the available device ('cuda' if available, else 'cpu').

    Returns:
        torch.device: Computation device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path=None, feature_extraction=True, num_classes=3):
    """
    Loads a pretrained ResNet50 model and modifies the final layer for skin lesion classification.

    Args:
        model_path (str): Optional path to a saved model state dict.
        feature_extraction (bool): If True, freeze convolutional base.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: The modified model.
    """
    # Load pretrained ResNet50
    model = models.resnet50(pretrained=True)

    if feature_extraction:
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    device = get_device()
    model = model.to(device)

    if model_path:
        # Load model state dict if provided
        model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def save_model(model, save_path):
    """
    Saves the model state dictionary.

    Args:
        model (torch.nn.Module): The model to save.
        save_path (str): Path where the model will be saved.
    """
    torch.save(model.state_dict(), save_path)


def reload_model(save_path, feature_extraction=True, num_classes=3):
    """
    Reloads the model from disk.

    Args:
        save_path (str): Path to the saved model state dict.
        feature_extraction (bool): If True, freeze convolutional base.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: The reloaded model.
    """
    return load_model(
        model_path=save_path,
        feature_extraction=feature_extraction,
        num_classes=num_classes,
    )
