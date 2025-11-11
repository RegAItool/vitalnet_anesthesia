# VitalNet Quick Reference

## 📋 Repository Summary

**Status**: 🟡 Under Peer Review - Partial Release
**Version**: 0.1.0-alpha
**License**: MIT (with proprietary components)

## 🔓 What's Included (Public)

### ✅ Data Processing
- VitalDB data downloader
- Signal preprocessing (interpolation, outlier removal, normalization)
- Feature extraction (time/frequency domain)

### ✅ Evaluation Tools
- Regression metrics (MAE, RMSE, R², CCC)
- Classification metrics (AUC, Sensitivity, Specificity)
- Clinical endpoint evaluation

### ✅ Documentation
- Complete usage guide
- Data format specification
- API documentation
- Working examples

## 🔒 What's NOT Included (Proprietary)

### ❌ Core Algorithms
- Transformer-CNN fusion architecture
- Multi-modal attention mechanisms
- Model Predictive Control (MPC) for dosing
- Patient-specific PK/PD optimization
- Trained model weights

**These will be released after paper acceptance.**

## 🚀 Quick Start (3 Steps)

```bash
# 1. Clone and install
git clone https://github.com/RegAItool/vitalnet_anesthesia.git
cd vitalnet_anesthesia
pip install -r requirements.txt

# 2. Download data
python -c "from data.download_vitaldb import download_vitaldb_data; download_vitaldb_data(n_cases=10)"

# 3. Run demo
python examples/demo_preprocessing.py
```

## 📊 Key Modules

| Module | Purpose | Status |
|--------|---------|--------|
| `data/download_vitaldb.py` | Download VitalDB data | ✅ Public |
| `data/preprocessing.py` | Signal preprocessing | ✅ Public |
| `data/feature_extraction.py` | Extract features | ✅ Public |
| `models/base_model.py` | Model interface | ⚠️ Stub only |
| `utils/metrics.py` | Evaluation metrics | ✅ Public |

## 📈 Reported Performance (From Paper)

### MAP Prediction
- MAE: 8.1 mmHg
- RMSE: 10.4 mmHg
- R²: 0.36

### BIS Prediction
- MAE: 2.8 units
- CCC: 0.91

### Hypotension Risk
- AUC: 0.875
- Sensitivity: 0.82
- Specificity: 0.79

## 🎯 Use Cases

### ✅ What You CAN Do
- Download and preprocess VitalDB data
- Extract standardized features
- Evaluate your own models using our metrics
- Build on our preprocessing pipeline
- Reproduce data preparation steps

### ❌ What You CANNOT Do (Yet)
- Train the VitalNet model (proprietary)
- Use the MPC dosing optimizer (proprietary)
- Access pre-trained weights (not released)

## 📚 Documentation Files

- `README.md` - Main introduction
- `docs/usage_guide.md` - Detailed usage
- `docs/data_format.md` - Data specifications
- `CONTRIBUTING.md` - Contribution guidelines
- `LICENSE` - MIT license with proprietary notice

## 🔗 Important Links

- **Paper**: Under Review
- **VitalDB**: https://vitaldb.net
- **Issues**: https://github.com/RegAItool/vitalnet_anesthesia/issues
- **Contact**: yu.han@eng.ox.ac.uk

## ⚡ Code Snippets

### Download Data
```python
from data.download_vitaldb import download_vitaldb_data
cases = download_vitaldb_data(output_dir='./data', n_cases=100)
```

### Preprocess Signal
```python
from data.preprocessing import SignalPreprocessor
preprocessor = SignalPreprocessor(sampling_rate=1.0)
processed = preprocessor.preprocess_pipeline(raw_signal, config)
```

### Extract Features
```python
from data.feature_extraction import FeatureExtractor
extractor = FeatureExtractor(sampling_rate=1.0)
features = extractor.extract_all_features(signal)
```

### Evaluate Predictions
```python
from utils.metrics import evaluate_regression
metrics = evaluate_regression(y_true, y_pred)
print(f"MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}")
```

## 🛡️ Protection Strategy

### For Reviewers
- Full data pipeline for reproducibility
- Complete evaluation metrics
- Clear documentation of methods

### For Protection
- Core model architecture: **NOT INCLUDED**
- MPC optimizer: **NOT INCLUDED**
- Trained weights: **NOT INCLUDED**
- Only interfaces/stubs provided

## 📧 Support

- **General Questions**: Open GitHub issue
- **Research Collaboration**: yu.han@eng.ox.ac.uk
- **Clinical Inquiries**: wenqp@dmu.edu.cn

## 📖 Citation

```bibtex
@article{wu2025vitalnet,
  title={VitalNet: Multimodal Triple-Endpoint Prediction and Personalized Closed-Loop Dosing in Anesthesia},
  author={Wu, Ping and Han, Yu and others},
  journal={Under Review},
  year={2025}
}
```

---

**Last Updated**: January 2025
**Maintained By**: VitalNet Research Team
