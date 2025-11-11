# VitalNet: Multimodal AI Platform for Anesthesia Monitoring

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**VitalNet** is a multimodal AI platform for real-time anesthesia monitoring and personalized drug dosing optimization. This repository contains the **data preprocessing pipeline** and **basic framework** for research purposes.

> **⚠️ IMPORTANT NOTICE**: This repository is under **peer review** and contains only partial implementation. The core algorithmic components (Transformer-CNN fusion architecture and MPC-based closed-loop control) are **proprietary** and will be released after publication.

## 🎯 Overview

VitalNet addresses critical challenges in anesthesia management:
- **Real-time monitoring** of Mean Arterial Pressure (MAP), Bispectral Index (BIS), and hypotension risk
- **Personalized drug dosing** using model predictive control (MPC)
- **Multi-modal data fusion** combining EEG waveforms and physiological signals

### Key Results
- MAP Prediction: MAE = 8.1 mmHg, R² = 0.36
- Hypotension Risk: AUC = 0.875
- BIS Prediction: MAE = 2.8 units, CCC = 0.91
- Improved time in BIS target range from 58% → 79%
- Reduced hypotensive episodes by 42%

## 📂 Repository Structure

```
vitalnet_anesthesia/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── data/                        # Data utilities (PUBLIC)
│   ├── download_vitaldb.py     # VitalDB data downloader
│   ├── preprocessing.py        # Data preprocessing pipeline
│   └── feature_extraction.py   # Time/frequency feature extraction
├── models/                      # Model framework (PARTIAL)
│   ├── __init__.py
│   ├── base_model.py           # Base model interface
│   └── predictor_stub.py       # Prediction interface (STUB ONLY)
├── utils/                       # Utility functions (PUBLIC)
│   ├── metrics.py              # Evaluation metrics
│   └── visualization.py        # Result visualization
├── examples/                    # Usage examples (PUBLIC)
│   ├── demo_preprocessing.py   # Data preprocessing demo
│   └── demo_evaluation.py      # Evaluation demo
└── docs/                        # Documentation (PUBLIC)
    ├── data_format.md          # Data format specification
    └── usage_guide.md          # Usage instructions
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/RegAItool/vitalnet_anesthesia.git
cd vitalnet_anesthesia

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```python
from data.download_vitaldb import download_vitaldb_data
from data.preprocessing import preprocess_signals

# Download VitalDB data
download_vitaldb_data(output_dir='./vitaldb_raw', n_cases=100)

# Preprocess signals
preprocessed_data = preprocess_signals(
    input_dir='./vitaldb_raw',
    output_dir='./vitaldb_processed'
)
```

### Feature Extraction

```python
from data.feature_extraction import extract_features

# Extract time/frequency features
features = extract_features(
    signal_data=preprocessed_data,
    feature_types=['time', 'frequency', 'physiological']
)
```

## 📊 Dataset

This work uses the **VitalDB** database:
- **Source**: Seoul National University Hospital
- **Size**: 3,023 surgical cases
- **Modalities**: EEG, ECG, arterial pressure, vital signs
- **Access**: https://vitaldb.net

**Citation**:
```
Lee, H. C., & Jung, C. W. (2018). VitalDB, a high-fidelity multi-parameter
vital signs database in surgical patients. Scientific Data, 5(1), 1-8.
```

## 🔒 Proprietary Components (Not Included)

The following components are **NOT** included in this repository due to ongoing peer review:

1. **Multimodal Fusion Architecture**
   - Transformer-CNN fusion backbone
   - Cross-attention mechanisms
   - Multi-scale temporal feature extraction

2. **Closed-Loop Control System**
   - Model Predictive Control (MPC) implementation
   - Individualized PK/PD parameter estimation
   - Real-time dosing optimization algorithms

3. **Trained Models**
   - Pre-trained weights
   - Model checkpoints
   - Calibration parameters

**These components will be released upon paper acceptance.**

## 📖 Documentation

- [Data Format Specification](docs/data_format.md)
- [Usage Guide](docs/usage_guide.md)
- [API Reference](docs/api_reference.md) *(Coming soon)*

## 🧪 Evaluation Metrics

Available evaluation utilities:

```python
from utils.metrics import evaluate_regression, evaluate_classification

# Regression metrics (MAP, BIS prediction)
reg_metrics = evaluate_regression(y_true, y_pred)
# Returns: MAE, RMSE, R², CCC

# Classification metrics (hypotension risk)
clf_metrics = evaluate_classification(y_true, y_pred_proba)
# Returns: AUC, Sensitivity, Specificity, PPV, NPV
```

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Important**: While the code is MIT licensed, the trained models and proprietary algorithms remain confidential until publication.

## 📧 Contact

For questions or collaboration:
- **Corresponding Author**: Dr. Qingping Wen (wenqp@dmu.edu.cn)
- **Lead Developer**: Yu Han (yu.han@eng.ox.ac.uk)

## 🙏 Acknowledgments

- VitalDB team for providing the open-access database
- Department of Anesthesiology, First Affiliated Hospital of Dalian Medical University
- University of Oxford, Institute of Biomedical Engineering

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{wu2025vitalnet,
  title={VitalNet: Multimodal Triple-Endpoint Prediction and Personalized Closed-Loop Dosing in Anesthesia},
  author={Wu, Ping and Han, Yu and Wu, Xiaoqi and Zhang, Jing and Li, Yunqi and Jiang, Mengna and Wen, Qingping},
  journal={Under Review},
  year={2025}
}
```

---

**Last Updated**: January 2025
**Status**: 🟡 Under Peer Review - Partial Release

## 🔬 Reproducibility

VitalNet follows academic reproducibility standards while protecting intellectual property:

### ✅ Fully Reproducible
- **Data preprocessing**: Complete pipeline with VitalDB downloader
- **Feature extraction**: All time/frequency domain methods
- **Evaluation metrics**: All metrics (MAE, RMSE, R², CCC, AUC, etc.)
- **Toy model**: Working demonstration on synthetic data

### ⚠️ Methodology Available
- **Core architecture**: High-level pseudocode provided in `demo/method_pseudocode.py`
- **Training pipeline**: Algorithmic flow documented
- **MPC optimization**: Framework and objectives described

### 🔒 Proprietary (Post-Acceptance Release)
- **Production model**: Full Transformer-CNN fusion implementation
- **Pre-trained weights**: Models trained on 3,023 cases
- **MPC dosing**: Patient-specific parameter optimization

**See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for detailed instructions.**

### Quick Demo

```bash
# Run toy model demonstration
cd demo
python toy_model.py

# Expected output:
# - Toy model trains on synthetic data
# - Achieves ~60-70% of full model performance
# - Demonstrates the methodology
```

**Note**: The toy model is for demonstration only. Paper results use the full proprietary model.

