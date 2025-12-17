# Fast PDF Layout Segmentation with Transformer-Based Models

This is an ML4S project for course Machine Learning (CS-433) at EPFL, a deep learning project for token-level classification and segmentation in PDF documents using Transformer-based models.

---

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Data Loader Usage](#data-loader-usage)
- [Prediction & Visualization](#prediction--visualization)
- [Experiments](#experiments)

---

## Overview

This project implements a two-stage pipeline for PDF document understanding:

1. **Token Classification**: Classifies each token into 11 categories (Formula, Footnote, List item, Table, Picture, Title, Text, Page header, Section header, Caption, Page footer)
2. **Segmentation**: Identifies segment boundaries between tokens

The project includes:
- Custom data loader compatible with DocLayNet dataset and PDFFeatures format
- Transformer-based models for token classification and segmentation
- LightGBM baseline models
- Comprehensive experiments and ablation studies
- Visualization pipeline for predictions

---

## Setup

### Prerequisites

- **Python 3.11** (required)
- `conda` for environment management
- `pdftohtml` (on macOS: `brew install poppler`)

### Installation

```bash
# Create conda environment
conda create -n pdf_token python=3.11
conda activate pdf_token

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

```
pyPDF2
datasets
reportlab
torch
transformers
tqdm
matplotlib
pdf-features  # Installed via git
```

### Pretrained Models
To run some of the experiments, you can use the pre-trained models in the `model branch`
[Download pretrained model](https://github.com/USER/REPO/raw/models/pretrained_models/your_model.bin)
- `best_larger_train_deep.pth` and `best_segment_model.pth` are the pre-trained Transformer Models. You can use them for prediction on real PDFs using `predict_and_visualize.py`
- `models/token_type_example_model.model` and `paragraph_extractor_example_model_.model` are baseline LighGBM Models trained on new dataset. You can use them to reproduce the results using `lightgbm_experiments.ipynb`

---

## Project Structure

```
.
├── model.py                    # Token classification model and feature extractor
├── model_segment.py            # Segmentation model
├── dataloader.py              # Data loading and preprocessing
├── predict_and_visualize.py   # Prediction pipeline with visualization
├── experiments.ipynb          # Main experiments and ablation studies
├── lightgbm_experiments.ipynb # LightGBM baseline experiments
├── lightgbm_train.py          # LightGBM model training
├── requirements.txt           # Python dependencies
├── pdf-labeled-data/          # Auto-generated labeled data (PDFFeatures format)
└── models/                    # Saved model checkpoints
```

### Key Files

- **`model.py`**: Contains `TransformerTagger` and `FeatureExtractor` classes for token classification
- **`model_segment.py`**: Contains `TransformerTaggerSegment` for segment boundary detection
- **`dataloader.py`**: Provides `DocLayNetDataset` and data processing utilities
- **`predict_and_visualize.py`**: End-to-end pipeline for PDF processing and visualization
- **`experiments.ipynb`**: Comprehensive experiments including ablation studies
- **`lightgbm_train.py`**: Training script for LightGBM baseline models


---

## Data Loader Usage

The `dataloader.py` module provides flexible data loading capabilities:

### DocLayNetDataset

```python
from dataloader import DocLayNetDataset

# Load dataset
dataset = DocLayNetDataset(split="test")  # or "train", "val"

# Access individual samples
pdf_features = dataset[0]

# The dataset automatically:
# 1. Downloads DocLayNet data (first time only)
# 2. Extracts tokens from PDFs
# 3. Combines dataset labels with tokens
# 4. Generates pdf-labeled-data/ directory
```

### PDFFeatures Format Compatibility

The data loader automatically generates `pdf-labeled-data/` directory in a format compatible with the [PdfFeatures](https://github.com/huridocs/pdf-features) repository. This allows:

- Direct usage with PDFFeatures tools
- Standalone data processing without the full dataset
- Easy integration with other PDF processing pipelines

### Visualization Options

```python
# Visualize original dataset labels
dataset.visualize_item(idx, output_path="labels.pdf")

# Visualize extracted tokens (unlabeled)
dataset.visualize_tokens(idx, output_path="tokens.pdf", labels=False)

# Visualize tokens with combined labels
dataset.visualize_tokens(idx, output_path="labeled_tokens.pdf", labels=True)
```

---

## Prediction & Visualization

The `predict_and_visualize.py` script provides an end-to-end pipeline for processing PDFs and visualizing predictions.

### Basic Usage

```bash
python predict_and_visualize.py \
    --pdf input.pdf \
    --model models/best_model.pth \
    --segment_model models/best_segment_model.pth \
    --output result.pdf
```

### Command Line Arguments

```bash
--pdf PATH              Input PDF file path (required)
--model PATH           Token classification model path (default: best_larger_train_deep.pth)
--segment_model PATH   Segmentation model path (default: best_segment_model.pth)
--output PATH          Output PDF path (default: output.pdf)
--device DEVICE        Device to use: auto/cpu/cuda/mps (default: auto)
--one_page_only        Process only the first page (flag)
```

### Pipeline Details

The script performs:
1. **PDF Loading**: Extracts tokens from input PDF
2. **Token Classification**: Predicts token type for each token
3. **Segmentation**: Identifies segment boundaries
4. **Visualization**: Generates annotated PDF with:
   - Colored bounding boxes for each token type
   - Segment boundaries highlighted
   - Category labels

### Example

```bash
# Process a single PDF with visualization
python predict_and_visualize.py \
    --pdf test_pdfs/report.pdf \
    --output annotated_paper.pdf \
    --device cpu \
    --model models/best_larger_train_deep.pth
    --segment_model models/best_segment_model.pth
```

### Token Categories

| ID | Category | Description |
|----|----------|-------------|
| 0 | Formula | Mathematical formulas |
| 1 | Footnote | Footnotes |
| 2 | List item | Bulleted/numbered lists |
| 3 | Table | Table content |
| 4 | Picture | Images/figures |
| 5 | Title | Document title |
| 6 | Text | Body text |
| 7 | Page header | Headers |
| 8 | Section header | Section titles |
| 9 | Caption | Figure/table captions |
| 10 | Page footer | Footers |

---

## Experiments

### Main Experiments

**`experiments.ipynb`** contains:
- Token classification experiments
- Segmentation experiments
- Ablation studies (feature importance, model architecture, hyperparameters)
- Performance metrics and visualizations
- Error analysis

### LightGBM Baseline

**`lightgbm_experiments.ipynb`** and **`lightgbm_train.py`**:
- LightGBM models for token classification
- Feature importance analysis
- Comparison with Transformer models
- Training on new/custom datasets

Run LightGBM training:
```bash
python lightgbm_train.py
```

---



## 🤝 Contributing

This project uses the [PdfFeatures](https://github.com/huridocs/pdf-features) library for PDF processing. The generated `pdf-labeled-data/` directory is fully compatible with PDFFeatures tools.

