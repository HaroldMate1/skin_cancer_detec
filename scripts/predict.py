#!/usr/bin/env python
"""
predict.py

A command-line tool for performing skin lesion classification on a single image or a directory of images.
Outputs the prediction label and confidence score. Optionally visualizes the result.
"""

import os
import click
import torch
from src import data_loader, model_handler, utils

# Define label mapping for output classes.
LABEL_MAP = {0: "benign", 1: "malignant", 2: "needs further analysis"}


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    help="Path to an image file or directory of images.",
)
@click.option(
    "--model",
    "-m",
    "model_path",
    default=None,
    help="Path to a saved model (.pt file).",
)
@click.option(
    "--visualize", "-v", is_flag=True, help="Display the image with prediction overlay."
)
def predict(input_path, model_path, visualize):
    """
    Load image(s), preprocess, perform prediction, and output results.
    """
    device = model_handler.get_device()
    # Load model (using feature extraction mode by default)
    model = model_handler.load_model(
        model_path=model_path, feature_extraction=True, num_classes=3
    )
    model.eval()

    # Determine if input is a file or directory
    image_paths = []
    if os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(input_path, file))
    elif os.path.isfile(input_path):
        image_paths.append(input_path)
    else:
        click.echo("Invalid input path provided.")
        return

    for path in image_paths:
        try:
            img = data_loader.load_image(path)
            input_tensor = data_loader.preprocess_image(img)
            # Add batch dimension and move to device
            input_tensor = input_tensor.unsqueeze(0).to(device)

            # Forward pass
            with torch.no_grad():
                outputs = model(input_tensor)

            pred_class, confidence = utils.get_label_from_prediction(outputs)
            click.echo(
                f"Image: {path} | Prediction: {LABEL_MAP[pred_class]} | Confidence: {confidence:.2f}"
            )

            if visualize:
                # Visualize using the original preprocessed tensor (remove batch dimension)
                utils.visualize_prediction(
                    input_tensor.squeeze(0), pred_class, confidence, LABEL_MAP
                )
        except Exception as e:
            click.echo(f"Error processing {path}: {e}")


if __name__ == "__main__":
    predict()
