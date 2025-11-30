# Attention-UNet-BrainTumor-Survival

A complete production-quality project for **Brain Tumor Segmentation using Attention U-Net** combined with **survival prediction** based on tumor features.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [API](#api)
  - [Frontend](#frontend)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)
- [Citation](#citation)

## 🎯 Overview

This project implements:

1. **Attention U-Net** for brain tumor segmentation (based on Oktay et al., 2018)
2. **XGBoost** classifier for survival risk prediction
3. **FastAPI** backend for model serving
4. **Streamlit** frontend for interactive visualization

### Key Features

- ✅ Full Attention U-Net implementation with attention gates
- ✅ Multi-modal MRI support (T1, T1ce, T2, FLAIR)
- ✅ Comprehensive feature extraction (volume, texture, intensity)
- ✅ Survival risk classification (Low/Medium/High)
- ✅ Production-ready API and UI
- ✅ Docker support

## 📐 Architecture

### Attention U-Net

The model follows the architecture from "Attention U-Net: Learning Where to Look for the Pancreas" (Oktay et al., 2018):

- **Encoder-Decoder** structure with skip connections
- **Attention Gates** between encoder-decoder connections
- **Grid-attention** mechanism (not global vector attention)
- **Additive attention** formulation

### Survival Prediction Pipeline

1. Extract tumor features from segmentation mask:
   - Tumor volume
   - Bounding box dimensions
   - Centroid location
   - Intensity statistics
   - Texture features (GLCM)

2. Train XGBoost classifier on extracted features
3. Predict survival risk category

## 🚀 Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM
- 10GB+ free disk space

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd Attention-UNet-BrainTumor-Survival
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n braintumor python=3.9
conda activate braintumor
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📦 Dataset Setup

### BRATS Dataset Download

This project uses the **BRATS (Brain Tumor Segmentation) dataset**:

#### Option 1: BRATS 2020 (Recommended)

1. Visit: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data
2. Download the dataset (contains `.h5` slice files)
3. Extract to `data/BRATS/` directory
4. Convert to NIfTI (only needed for the Kaggle `.h5` release):

   ```bash
   # Converts volume_*_slice_*.h5 into standard BRATS folders
   python data/convert_h5_to_nifti.py \
       --input-dir data/BRATS/BraTs2020_training_data/content/data \
       --output-dir data/BRATS/converted
   ```

5. Update `config.yaml` to point to `data/BRATS/converted`

#### Option 2: BRATS 2018 (already in NIfTI format)

1. Visit: https://www.kaggle.com/datasets/meetnagadia/brats-2018
2. Download the dataset
3. Extract to `data/BRATS/` directory

### Expected Directory Structure

```
data/
└── BRATS/
    └── BraTS2020_TrainingData/  # or "converted/" if using the HDF5 release
        └── MICCAI_BraTS2020_TrainingData/
            ├── BraTS20_Training_001/
            │   ├── BraTS20_Training_001_t1.nii.gz
            │   ├── BraTS20_Training_001_t1ce.nii.gz
            │   ├── BraTS20_Training_001_t2.nii.gz
            │   ├── BraTS20_Training_001_flair.nii.gz
            │   └── BraTS20_Training_001_seg.nii.gz
            ├── BraTS20_Training_002/
            └── ...
```

> **Note:** If you converted the Kaggle `.h5` release, the expected structure lives under `data/BRATS/converted/`.

### Verify Dataset

```bash
# For original BRATS folders
python data/dataset.py data/BRATS

# For converted Kaggle release
python data/dataset.py data/BRATS/converted
```

## 📖 Usage

### Training

#### Step 1: Train Segmentation Model

```bash
python training/train_segmentation.py --config config.yaml
```

**Options:**
- `--config`: Path to config file (default: `config.yaml`)
- `--resume`: Path to checkpoint to resume from

**Training Process:**
1. Model will train on training set
2. Validate on validation set
3. Save best model to `outputs/checkpoints/best_model.pth`
4. Logs saved to `outputs/logs/` (TensorBoard)

**Monitor Training:**
```bash
tensorboard --logdir outputs/logs
```

#### Step 2: Train Survival Prediction Model

```bash
python training/train_survival.py --config config.yaml
```

**Process:**
1. Extracts features from ground truth masks
2. Trains XGBoost classifier
3. Saves model to `outputs/models/survival_model.pkl`

### Inference

#### Command Line Inference

```bash
python inference/predict.py \
    --mri path/to/mri.nii.gz \
    --slice 75 \
    --config config.yaml \
    --seg_model outputs/checkpoints/best_model.pth \
    --surv_model outputs/models/survival_model.pkl
```

**Output:**
- Segmentation mask saved to `outputs/predictions/`
- Overlay visualization
- JSON results with tumor features and survival prediction

#### Python API

```python
from inference.predict import BrainTumorPredictor

# Initialize predictor
predictor = BrainTumorPredictor(config_path="config.yaml")

# Predict
results = predictor.predict("path/to/mri.nii.gz", slice_idx=75)

print(f"Tumor detected: {results['tumor_detected']}")
print(f"Tumor volume: {results['tumor_volume']}")
print(f"Risk category: {results['survival_prediction']['risk_category']}")
```

### API

#### Start FastAPI Server

```bash
python api/main.py
```

Or using uvicorn directly:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Prediction:**
```bash
curl -X POST "http://localhost:8000/predict?slice_idx=75" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/mri.nii.gz"
```

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Frontend

#### Launch Streamlit App

```bash
streamlit run frontend/app.py
```

The app will open in your browser at `http://localhost:8501`

#### Using the UI

1. **Upload MRI**: Click "Browse files" and select a NIfTI file
2. **Select Slice**: Optionally specify slice index (default: middle slice)
3. **Analyze**: Click "🔍 Analyze" button
4. **View Results**:
   - Original MRI
   - Segmentation mask
   - Overlay visualization
   - Survival risk prediction with probabilities
5. **Download**: Download mask and overlay images

## 📁 Project Structure

```
Attention-UNet-BrainTumor-Survival/
├── model/
│   ├── __init__.py
│   ├── attention_unet.py          # Attention U-Net architecture
│   └── survival_model.py           # XGBoost survival predictor
├── data/
│   ├── __init__.py
│   └── dataset.py                  # BRATS dataset loader
├── training/
│   ├── __init__.py
│   ├── train_segmentation.py      # Segmentation training script
│   └── train_survival.py          # Survival model training
├── inference/
│   ├── __init__.py
│   └── predict.py                 # Inference pipeline
├── api/
│   ├── __init__.py
│   └── main.py                    # FastAPI backend
├── frontend/
│   ├── __init__.py
│   └── app.py                     # Streamlit UI
├── utils/
│   ├── __init__.py
│   ├── preprocess.py              # Preprocessing utilities
│   ├── postprocess.py             # Postprocessing utilities
│   ├── feature_extraction.py      # Feature extraction
│   └── metrics.py                 # Evaluation metrics
├── outputs/
│   ├── checkpoints/               # Model checkpoints
│   ├── logs/                      # Training logs
│   ├── models/                    # Saved models
│   └── predictions/               # Prediction outputs
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── .gitignore
└── README.md                      # This file
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

- **Model parameters**: Architecture, filters, depth
- **Training hyperparameters**: Batch size, learning rate, epochs
- **Data settings**: Dataset path, augmentation
- **Inference settings**: Threshold, device
- **API/Frontend settings**: Ports, file size limits

### Key Configuration Options

```yaml
model:
  in_channels: 4        # T1, T1ce, T2, FLAIR
  num_classes: 1          # Binary segmentation
  base_filters: 64
  depth: 4
  attention: true

training_segmentation:
  batch_size: 4
  num_epochs: 100
  learning_rate: 0.0001
  patch_size: [128, 128]

inference:
  threshold: 0.5
  device: "cuda"        # or "cpu"
```

## 📊 Results

### Example Output

```
PREDICTION RESULTS
==================================================
{
  "tumor_detected": true,
  "tumor_volume": 1234.0,
  "bounding_box": {
    "min_x": 45,
    "min_y": 67,
    "max_x": 123,
    "max_y": 145
  },
  "slice_index": 75,
  "survival_prediction": {
    "risk_category": "Medium Risk",
    "probabilities": {
      "low_risk": 0.15,
      "medium_risk": 0.65,
      "high_risk": 0.20
    }
  },
  "mask_path": "outputs/predictions/patient_001_slice75_mask.png",
  "overlay_path": "outputs/predictions/patient_001_slice75_overlay.png"
}
==================================================
```

## 🐳 Docker

### Build Docker Image

```bash
docker build -t braintumor-segmentation .
```

### Run Container

```bash
# Run API
docker run -p 8000:8000 -v $(pwd)/outputs:/app/outputs braintumor-segmentation

# Run with GPU support
docker run --gpus all -p 8000:8000 -v $(pwd)/outputs:/app/outputs braintumor-segmentation
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./outputs:/app/outputs
      - ./data:/app/data
```

Run:
```bash
docker-compose up
```

## 📚 Citation

If you use this code, please cite the original Attention U-Net paper:

```bibtex
@article{oktay2018attention,
  title={Attention U-Net: Learning Where to Look for the Pancreas},
  author={Oktay, Ozan and Schlemper, Jo and Folgoc, Loic Le and Lee, Matthew and Heinrich, Mattias and Misawa, Kazunari and Mori, Kensaku and McDonagh, Steven and Hammerla, Nils Y and Kainz, Bernhard and others},
  journal={arXiv preprint arXiv:1804.03999},
  year={2018}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is provided for research and educational purposes.

## ⚠️ Disclaimer

This software is for research purposes only. It is not intended for clinical use or medical diagnosis.

## 🆘 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` in `config.yaml`
2. **Dataset not found**: Verify dataset path and structure
3. **Model not loading**: Ensure models are trained first
4. **Import errors**: Verify all dependencies are installed

### Getting Help

- Check logs in `outputs/logs/`
- Verify configuration in `config.yaml`
- Ensure dataset is properly formatted
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

---

**Built with ❤️ for medical imaging research**

