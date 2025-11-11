# VitalNet Reproducibility Guide

## 🎯 Overview

This document explains how to reproduce the key findings from the VitalNet paper while respecting intellectual property protection.

## 📊 What Can Be Reproduced

### ✅ Fully Reproducible (Public Code)

1. **Data Preprocessing Pipeline**
   - VitalDB data downloading
   - Signal interpolation and outlier removal
   - Feature extraction (time/frequency domain)
   - **Code**: `data/` module

2. **Evaluation Metrics**
   - Regression metrics (MAE, RMSE, R², CCC)
   - Classification metrics (AUC, Sensitivity, Specificity)
   - **Code**: `utils/metrics.py`

3. **Toy Model Demo**
   - Simplified VitalNet architecture
   - Synthetic data training
   - **Code**: `demo/toy_model.py`

### ⚠️ Partially Reproducible (Methodology Only)

4. **Core Model Architecture**
   - High-level pseudocode provided
   - Key concepts and flow explained
   - **Code**: `demo/method_pseudocode.py`
   - **Note**: Full implementation is proprietary

5. **MPC-Based Dosing**
   - Algorithm framework provided
   - Optimization objective defined
   - **Code**: `demo/method_pseudocode.py`
   - **Note**: Patient-specific parameters are proprietary

### ❌ Not Reproducible (Proprietary)

6. **Production Model**
   - Transformer-CNN fusion details
   - Cross-modal attention mechanisms
   - Pre-trained weights (3,023 cases)
   - **Status**: Will be released upon paper acceptance

---

## 🚀 Step-by-Step Reproducibility

### Step 1: Environment Setup

```bash
git clone https://github.com/RegAItool/vitalnet_anesthesia.git
cd vitalnet_anesthesia
pip install -r requirements.txt
```

### Step 2: Download VitalDB Data

```python
from data.download_vitaldb import download_vitaldb_data

# Download 100 cases for testing
cases = download_vitaldb_data(
    output_dir='./vitaldb_data',
    n_cases=100,
    duration=300  # 5 minutes per case
)
```

**Requirements**:
- Internet connection
- ~500 MB storage for 100 cases
- VitalDB is publicly available: https://vitaldb.net

### Step 3: Preprocess Data

```python
from data.preprocessing import SignalPreprocessor

preprocessor = SignalPreprocessor(sampling_rate=1.0)

config = {
    'interpolate': True,
    'remove_outliers': True,
    'normalize': 'zscore'
}

# Process each signal
processed_signal = preprocessor.preprocess_pipeline(raw_signal, config)
```

**Expected Output**:
- Clean signals with no missing values
- Outliers removed (1-99 percentile)
- Z-score normalized

### Step 4: Extract Features

```python
from data.feature_extraction import FeatureExtractor

extractor = FeatureExtractor(sampling_rate=1.0)

# Extract all features
features = extractor.extract_all_features(processed_signal)

# Features include:
# - Time domain: mean, std, IQR, kurtosis, etc. (14 features)
# - Frequency domain: spectral centroid, entropy, etc. (6 features)
```

**Expected Output**:
- 20 features per signal
- Feature dictionary with descriptive names

### Step 5: Train Toy Model (Demo)

```bash
cd demo
python toy_model.py
```

**Expected Performance**:
- Toy model MAE: ~15-20 (on synthetic data)
- Training time: ~2 minutes on CPU
- **Note**: This is 60-70% of full model performance

**Toy vs. Full Model Comparison**:

| Metric | Toy Model (Demo) | Full VitalNet (Paper) |
|--------|------------------|----------------------|
| MAP MAE | ~15-20 | 8.1 mmHg |
| Architecture | Simplified | Full Transformer-CNN |
| Data | Synthetic | 3,023 real cases |
| Purpose | Demonstration | Production |

### Step 6: Evaluate Your Model

```python
from utils.metrics import evaluate_regression

# For regression (MAP, BIS)
metrics = evaluate_regression(y_true, y_pred)
print(f"MAE: {metrics['mae']:.2f}")
print(f"R²: {metrics['r2']:.4f}")
print(f"CCC: {metrics['ccc']:.4f}")

# For classification (hypotension)
from utils.metrics import evaluate_classification
metrics = evaluate_classification(y_true, y_pred_proba)
print(f"AUC: {metrics['auc']:.4f}")
```

---

## 📋 Reproducibility Checklist

### Data Preprocessing ✅

- [x] VitalDB data downloader (public)
- [x] Signal preprocessing pipeline (public)
- [x] Feature extraction methods (public)
- [x] Can reproduce preprocessing on VitalDB data

### Model Architecture ⚠️

- [x] High-level pseudocode provided
- [x] Toy model demonstration available
- [ ] Full model weights (proprietary, will release post-acceptance)
- [ ] Exact architecture details (proprietary)

### Evaluation ✅

- [x] All metrics implemented
- [x] Evaluation code public
- [x] Can compute same metrics on own models

### Training Pipeline ⚠️

- [x] Training loop pseudocode provided
- [x] Toy training example available
- [ ] Full training code (proprietary)
- [ ] Hyperparameters (some proprietary)

---

## 🎓 Academic Reproducibility Standards

### What We Provide

✅ **Data Access**: VitalDB is publicly available
✅ **Preprocessing Code**: Complete pipeline provided
✅ **Evaluation Metrics**: All metrics with source code
✅ **Method Description**: Detailed pseudocode
✅ **Toy Implementation**: Working demonstration

### What Is Proprietary

🔒 **Core Model**: Transformer-CNN fusion implementation
🔒 **MPC Optimizer**: Patient-specific parameter estimation
🔒 **Pre-trained Weights**: Trained on 3,023 cases
🔒 **Production Code**: Optimized training pipeline

### Why This Approach?

1. **Academic Integrity**: Sufficient information to validate claims
2. **IP Protection**: Core innovations remain proprietary during review
3. **Future Release**: Full code will be released upon acceptance
4. **Standard Practice**: Common in AI papers with commercial potential

---

## 📧 Questions About Reproducibility?

### For Reviewers

If you need clarification on any method or result:
1. Check `demo/method_pseudocode.py` for detailed algorithms
2. Run `demo/toy_model.py` for a working example
3. Contact: yu.han@eng.ox.ac.uk

### For Researchers

To build upon this work:
1. Use our preprocessing pipeline (fully public)
2. Use our evaluation metrics (fully public)
3. Implement your own model using the provided pseudocode
4. Compare against our reported results

---

## 📚 Citation

If you use this code or methodology, please cite:

```bibtex
@article{wu2025vitalnet,
  title={VitalNet: Multimodal Triple-Endpoint Prediction and Personalized Closed-Loop Dosing in Anesthesia},
  author={Wu, Ping and Han, Yu and Wu, Xiaoqi and Zhang, Jing and Li, Yunqi and Jiang, Mengna and Wen, Qingping},
  journal={Under Review},
  year={2025}
}
```

---

## 🔄 Reproducibility Timeline

### Current (Review Period)
- ✅ Data preprocessing: **Fully reproducible**
- ✅ Feature extraction: **Fully reproducible**
- ⚠️ Model architecture: **Pseudocode only**
- ✅ Evaluation: **Fully reproducible**
- ⚠️ Toy model: **Demo available**

### Post-Acceptance
- ✅ Full model code: **Will be released**
- ✅ Pre-trained weights: **Will be released**
- ✅ Training scripts: **Will be released**
- ✅ Production pipeline: **Will be released**

---

**Last Updated**: January 2025
**Status**: Under Peer Review - Partial Release
**Reproducibility Level**: Method validation + Public tooling
