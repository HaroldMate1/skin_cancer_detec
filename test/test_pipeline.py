"""
test_pipeline.py

Basic tests to ensure that the data loading and model functions are working as expected.
"""

import os
import pytest
from PIL import Image
import torch
from src import data_loader, model_handler


def test_load_image(tmp_path):
    # Create a temporary image file
    img_path = tmp_path / "test.jpg"
    image = Image.new("RGB", (100, 100), color="red")
    image.save(img_path)

    loaded_image = data_loader.load_image(str(img_path))
    assert loaded_image.size == (100, 100)


def test_preprocess_image(tmp_path):
    # Create a temporary image and test preprocessing
    img_path = tmp_path / "test.jpg"
    image = Image.new("RGB", (300, 300), color="blue")
    image.save(img_path)

    img = data_loader.load_image(str(img_path))
    tensor = data_loader.preprocess_image(img)
    # The tensor shape should be (3, 224, 224)
    assert tensor.shape == (3, 224, 224)


def test_load_model():
    # Test loading model and forward pass with a dummy tensor.
    model = model_handler.load_model(feature_extraction=True, num_classes=3)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    # Output shape should be (1, 3)
    assert output.shape == (1, 3)
