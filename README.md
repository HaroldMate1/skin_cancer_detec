# Skin Lesion Classification: AI-based Skin Cancer Detection

This project provides a complete, modular, and production-ready pipeline for the classification of skin lesions using advanced deep learning techniques. Leveraging pretrained convolutional neural network models (such as ResNet50), the pipeline classifies skin lesion images into diagnostic categories (benign, malignant, or requiring further analysis) and helps in preliminary risk assessment for skin cancer.

## Project Highlights

- **End-to-end pipeline**: From image preprocessing, model loading, and fine-tuning, to deployment and prediction.
- **Modular and Scalable**: Easily adaptable to different datasets and additional classes or models.
- **User-friendly Interface**: Command-line tool for quick and efficient image classification.
- **Visualization**: Includes optional visualization capabilities for interpreting model predictions.
- **Testing and CI Integration**: Basic tests and GitHub Actions CI workflow ensure robustness and ease of maintenance.

## Project Structure

```
skin_lesion_classification/
├── data/                      # Raw images and datasets
├── models/                    # Trained and serialized model files
├── notebooks/                 # Exploratory and analysis notebooks
├── scripts/
│   └── predict.py             # CLI for inference
├── src/
│   ├── data_loader.py         # Image processing and augmentation
│   ├── model_handler.py       # Model training, loading, and saving
│   └── utils.py               # Utility functions and visualization
├── tests/                     # Pytest-based tests
├── .github/workflows/ci.yml   # GitHub Actions CI configuration
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/skin_lesion_classification.git
cd skin_lesion_classification
```

### 2. Create Virtual Environment

- **Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

- **Linux/Mac:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Classify Images

Run predictions on a single image or a directory:

```bash
python scripts/predict.py --input path/to/image_or_directory --visualize
```

- `--input`: Path to an image or directory of images.
- `--visualize`: (Optional) Display images with prediction overlays.

## Testing

Run the tests:

```bash
pytest tests/
```

## Continuous Integration

Automated testing is performed on each push to ensure pipeline integrity using GitHub Actions.

## Ethical Disclaimer

> **Important:** This model is intended strictly for research and educational purposes. It has not undergone clinical validation. Clinical decisions must always be confirmed by qualified medical professionals.

## References

- Pretrained models provided by [Torchvision](https://pytorch.org/vision/stable/models.html).
- Recommended dataset: [ISIC Archive](https://www.isic-archive.com/).

## License

MIT License


