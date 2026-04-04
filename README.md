# PGDC: A Physiology-Guided Divide-and-Conquer Framework for Cross-Time Drift-Resilient ECG Biometric Recognition

## Project Introduction

PGDC (Physiology-Guided Divide-and-Conquer) is a framework for cross-time drift-resilient ECG biometric recognition. This project aims to address the drift problem when ECG signals are collected at different times, improving the accuracy and stability of biometric recognition.

## Directory Structure

```
PGDC/
├── compare_method/       # Implementation of comparison methods
│   ├── 2022-TETCI/       # 2022 TETCI conference method
│   ├── 2022-TIM/         # 2022 TIM conference method
│   ├── 2023-TOMM/        # 2023 TOMM conference method
│   ├── 2025-ArXiv/       # 2025 ArXiv paper method
│   ├── 2025-jsen/        # 2025 JSEN method
│   └── 2026-ArXiv/       # 2026 ArXiv paper method
├── model/                # Model definitions
│   ├── compare/          # Comparison models
│   ├── CrossAttention.py # Cross attention module
│   ├── ExpertEncoder.py  # Expert encoder
│   ├── MultiExpertModel.py # Multi-expert model
│   ├── ViT.py            # Vision Transformer
│   └── ...
├── utils/                # Utility classes and data processing
│   ├── dataset/          # Dataset related
│   ├── train/            # Training related
│   ├── trainer/          # Trainers
│   ├── util/             # Utility functions
│   └── __init__.py
└── config.py             # Configuration file
```

## Installation Instructions

### Environment Requirements

- Python 3.8+
- PyTorch 2.4.1
- torchvision 0.19.1
- NumPy 1.24.3
- SciPy 1.10.1
- scikit-learn 1.3.0
- pandas 2.0.3
- matplotlib 3.7.3
- spectrum 0.8.1 (for 2022-TIM method)
- neurokit2 0.2.11
- fastdtw 0.3.4
- wfdb 4.1.2

### Installation Steps

1. Clone the repository
   ```bash
   git clone https://github.com/hustselab511/PGDC.git
   cd PGDC
   ```
2. Create and activate virtual environment
   ```bash
   # Using conda
   conda create -n torch_11.8 python=3.8
   conda activate torch_11.8

   # Or using venv
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   ```
3. Install dependencies
   ```bash
   # Install all dependencies using requirements.txt
   pip install -r requirements.txt
   ```

## Usage

### Run Comparison Methods

```bash
# Run 2022-TIM method
cd compare_method/2022-TIM
python train_test.py

# Run other comparison methods
cd compare_method/[method_directory]
python train_test.py
```

### Train and test PGDC Model

```bash
cd utils/train
python train_test.py
```

<br />

## Configuration

The configuration file `config.py` contains the following main configurations：

- Dataset paths
- Training parameters
- Model parameters
- Preprocessing settings


## Contact

- Project homepage：<https://github.com/hustselab511/PGDC>
- Issue tracking：<https://github.com/hustselab511/PGDC/issues>

## Changelog

### v1.0.0

- Initial version
- Implemented multi-expert model
- Integrated multiple comparison methods
- Supported cross-time drift resilience evaluation

