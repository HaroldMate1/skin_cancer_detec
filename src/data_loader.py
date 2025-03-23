# This file contains the code for loading and preprocessing images for skin lesion classification.
"""
data_loader.py

Handles image loading, preprocessing, and augmentation for skin lesion classification.
"""

import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# Define ImageNet statistics for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image(image_path):
    """
    Loads an image from disk.

    Args:
        image_path (str): Path to the image file.

    Returns:
        PIL.Image.Image: Loaded image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def preprocess_image(image, target_size=(224, 224)):
    """
    Resizes, converts to tensor, and normalizes the image.

    Args:
        image (PIL.Image.Image): Input image.
        target_size (tuple): Desired output size.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    preprocess = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return preprocess(image)


def augment_image(image):
    """
    Applies random augmentations to the input image.

    Augmentations include horizontal/vertical flip, rotation, and color jitter.
    Useful for training to improve model robustness.

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        PIL.Image.Image: Augmented image.
    """
    augmentation = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
        ]
    )
    return augmentation(image)


def load_images_from_directory(directory, target_size=(224, 224)):
    """
    Loads and preprocesses all images in a given directory.

    Args:
        directory (str): Path to the directory containing images.
        target_size (tuple): Target size for image resizing.

    Returns:
        list: List of preprocessed image tensors.
    """
    images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(directory, filename)
            img = load_image(image_path)
            img = preprocess_image(img, target_size=target_size)
            images.append(img)
    return images
