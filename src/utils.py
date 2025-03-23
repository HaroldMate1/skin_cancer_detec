"""
utils.py

Provides utility functions for visualization and other helper tasks.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def denormalize(tensor, mean, std):
    """
    Denormalizes an image tensor for visualization.

    Args:
        tensor (torch.Tensor): Normalized image tensor.
        mean (list): Mean used in normalization.
        std (list): Standard deviation used in normalization.

    Returns:
        np.array: Denormalized image in numpy format.
    """
    tensor = tensor.cpu().clone().detach()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return np.transpose(tensor.numpy(), (1, 2, 0))


def visualize_prediction(image_tensor, prediction, confidence, label_map):
    """
    Displays the image with an overlay showing the predicted label and confidence score.

    Args:
        image_tensor (torch.Tensor): Preprocessed image tensor.
        prediction (int): Predicted class index.
        confidence (float): Confidence score.
        label_map (dict): Mapping from class indices to labels.
    """
    # Denormalize image for display
    image_np = denormalize(
        image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    plt.imshow(image_np)
    plt.title(f"Prediction: {label_map[prediction]} ({confidence:.2f} confidence)")
    plt.axis("off")
    plt.show()


def get_label_from_prediction(preds):
    """
    Converts model output logits to a label and confidence score.

    Args:
        preds (torch.Tensor): Output logits from the model.

    Returns:
        tuple: (predicted class index, confidence score)
    """
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(preds, dim=1)
    confidence, prediction = torch.max(probs, 1)
    return prediction.item(), confidence.item()
