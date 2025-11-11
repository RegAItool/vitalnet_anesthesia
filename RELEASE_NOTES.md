# VitalNet Release Notes

## Version 0.1.0-alpha (January 2025)

### Status
🟡 **Under Peer Review - Partial Release**

### Overview
This is a **partial release** of VitalNet for peer review transparency and research reproducibility. Core proprietary algorithms are **not included** and will be released upon paper acceptance.

### What's New

#### ✅ Public Components Released

**Data Processing**
- VitalDB data downloader with multi-track support
- Comprehensive signal preprocessing pipeline
  - Missing value interpolation
  - Outlier removal (percentile-based)
  - Bandpass filtering
  - Multiple normalization methods (z-score, min-max, robust)
- Time-domain feature extraction (14 features)
- Frequency-domain feature extraction (6 features)
- Physiological feature extraction (domain-specific)

**Evaluation Metrics**
- Regression metrics: MAE, RMSE, R², CCC, Pearson r, MAPE
- Classification metrics: AUC-ROC, AP, Sensitivity, Specificity, PPV, NPV, F1
- Clinical endpoint evaluation for MAP, BIS, and hypotension risk
- Formatted reporting utilities

**Documentation**
- Comprehensive README with project overview
- Detailed usage guide with examples
- Data format specification
- API documentation
- Contributing guidelines

**Examples**
- Signal preprocessing demo
- Feature extraction demo
- Batch processing demo

#### 🔒 Proprietary Components (NOT Included)

The following components are **proprietary** and will be released after paper acceptance:

1. **VitalNet Core Model**
   - Transformer-CNN fusion architecture
   - Multi-scale temporal feature extraction
   - Cross-modal attention mechanisms
   - Triple-endpoint prediction network

2. **Closed-Loop Control**
   - Model Predictive Control (MPC) implementation
   - Patient-specific PK/PD parameter estimation
   - Real-time dosing optimization algorithms
   - Safety constraint enforcement

3. **Trained Models**
   - Pre-trained model weights
   - Model checkpoints from 3,023 cases
   - Calibration parameters

### Installation

```bash
git clone https://github.com/RegAItool/vitalnet_anesthesia.git
cd vitalnet_anesthesia
pip install -r requirements.txt
```

### Dependencies

- Python >= 3.8
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- SciPy >= 1.7.0
- scikit-learn >= 1.0.0
- TensorFlow >= 2.8.0
- VitalDB >= 1.0.0

### Known Limitations

- Model interfaces are stubs only (return dummy predictions)
- No training functionality for core VitalNet model
- MPC optimizer interface only (no implementation)
- Limited to data preprocessing and evaluation

### Roadmap

#### Post-Acceptance (Planned)
- [ ] Release Transformer-CNN fusion model
- [ ] Release MPC dosing optimizer
- [ ] Release pre-trained weights
- [ ] Add training scripts
- [ ] Add real-time inference examples
- [ ] Web-based demo interface

#### Future Enhancements
- [ ] Support for additional datasets (MIMIC-IV, eICU)
- [ ] Extended feature extraction methods
- [ ] Model explainability tools (SHAP, LIME)
- [ ] Clinical validation studies
- [ ] Multi-center deployment guide

### Breaking Changes
None (initial release)

### Bug Fixes
None (initial release)

### Contributors
- Ping Wu (First Affiliated Hospital of Dalian Medical University)
- Yu Han (University of Oxford)
- Xiaoqi Wu, Jing Zhang, Yunqi Li, Mengna Jiang
- Qingping Wen (Corresponding Author)

### Acknowledgments
- VitalDB team for open-access database
- Department of Anesthesiology, Dalian Medical University
- Institute of Biomedical Engineering, University of Oxford

### License
MIT License (with proprietary component notices)

### Citation

```bibtex
@article{wu2025vitalnet,
  title={VitalNet: Multimodal Triple-Endpoint Prediction and Personalized Closed-Loop Dosing in Anesthesia},
  author={Wu, Ping and Han, Yu and Wu, Xiaoqi and Zhang, Jing and Li, Yunqi and Jiang, Mengna and Wen, Qingping},
  journal={Under Review},
  year={2025}
}
```

### Contact
- **Issues**: https://github.com/RegAItool/vitalnet_anesthesia/issues
- **Email**: yu.han@eng.ox.ac.uk
- **Clinical**: wenqp@dmu.edu.cn

---

**Disclaimer**: This is a research project under peer review. The software is provided "as is" for academic purposes. Clinical use requires proper validation and regulatory approval.
