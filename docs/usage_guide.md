# VitalNet Usage Guide

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/RegAItool/vitalnet_anesthesia.git
cd vitalnet_anesthesia

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Workflow

```python
# 1. Download data
from data.download_vitaldb import download_vitaldb_data

cases = download_vitaldb_data(
    output_dir='./vitaldb_data',
    n_cases=100,
    duration=300
)

# 2. Preprocess signals
from data.preprocessing import SignalPreprocessor

preprocessor = SignalPreprocessor(sampling_rate=1.0)
config = {
    'interpolate': True,
    'remove_outliers': True,
    'normalize': 'zscore'
}

processed_signal = preprocessor.preprocess_pipeline(raw_signal, config)

# 3. Extract features
from data.feature_extraction import FeatureExtractor

extractor = FeatureExtractor(sampling_rate=1.0)
features = extractor.extract_all_features(processed_signal)

# 4. Evaluate predictions
from utils.metrics import evaluate_regression

metrics = evaluate_regression(y_true, y_pred)
print(f"MAE: {metrics['mae']:.2f}")
print(f"R²: {metrics['r2']:.4f}")
```

## Detailed Usage

### 1. Data Downloading

#### Download Vital Signs Data

```python
from data.download_vitaldb import download_vitaldb_data

# Download 100 cases with 5 minutes of data each
cases = download_vitaldb_data(
    output_dir='./vitaldb_data',
    n_cases=100,
    duration=300  # seconds
)
```

#### Download Waveform Data

```python
from data.download_vitaldb import download_waveform_data

# Download high-frequency waveforms for a specific case
success = download_waveform_data(
    caseid=123,
    output_dir='./vitaldb_waveforms',
    duration=60  # seconds
)
```

### 2. Signal Preprocessing

#### Basic Preprocessing

```python
from data.preprocessing import SignalPreprocessor
import numpy as np

# Create preprocessor
preprocessor = SignalPreprocessor(sampling_rate=1.0)

# Your raw signal (with possible NaN and outliers)
raw_hr = np.array([72, 73, np.nan, 150, 71, ...])

# Interpolate missing values
interpolated = preprocessor.interpolate_missing(raw_hr)

# Remove outliers
cleaned = preprocessor.remove_outliers(interpolated)

# Normalize
normalized = preprocessor.normalize_signal(cleaned, method='zscore')
```

#### Complete Pipeline

```python
# Define preprocessing configuration
config = {
    'interpolate': True,
    'remove_outliers': True,
    'bandpass': {
        'lowcut': 0.5,   # Hz
        'highcut': 4.0   # Hz
    },
    'normalize': 'zscore'
}

# Apply complete pipeline
processed = preprocessor.preprocess_pipeline(raw_signal, config)
```

#### Batch Processing

```python
from data.preprocessing import preprocess_vitaldb_case

# Preprocess a case file
preprocessed_df = preprocess_vitaldb_case(
    case_file='vitaldb_data/case_123.csv',
    output_file='vitaldb_processed/case_123.csv',
    config=config
)
```

### 3. Feature Extraction

#### Extract Time-Domain Features

```python
from data.feature_extraction import FeatureExtractor

extractor = FeatureExtractor(sampling_rate=1.0)

# Extract time-domain features
time_features = extractor.extract_time_features(signal)

# Available features:
# - mean, std, median, q25, q75
# - range, iqr, cv, mad
# - mean_abs_diff, rms
# - zero_crossing_rate
# - kurtosis, skewness
```

#### Extract Frequency-Domain Features

```python
# Extract frequency-domain features
freq_features = extractor.extract_frequency_features(signal)

# Available features:
# - dominant_freq, dominant_power
# - total_power
# - spectral_centroid, spectral_spread
# - spectral_entropy
```

#### Extract All Features

```python
# Extract both time and frequency features
all_features = extractor.extract_all_features(signal)

# Returns dict with keys like:
# 'time_mean', 'time_std', ..., 'freq_dominant_freq', ...
```

#### Extract Physiological Features

```python
# Extract domain-specific physiological features
physio_features = extractor.extract_physiological_features(
    hr=hr_signal,
    map_signal=map_signal,
    spo2=spo2_signal
)
```

### 4. Model Interface (Stub Only)

⚠️ **Note**: The actual VitalNet model is proprietary and not included.

```python
from models import VitalNetStub

# Create stub model (for demo purposes only)
model = VitalNetStub()

# This returns dummy predictions
predictions = model.predict_all_endpoints({
    'numerical': feature_array
})

# predictions = {
#     'map': [...],
#     'bis': [...],
#     'hypotension_risk': [...]
# }
```

### 5. Evaluation

#### Regression Evaluation (MAP, BIS)

```python
from utils.metrics import evaluate_regression, print_evaluation_report

# Evaluate predictions
metrics = evaluate_regression(y_true, y_pred)

# Print formatted report
print_evaluation_report(metrics, endpoint_name="MAP Prediction")
```

**Metrics returned:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (R-squared)
- CCC (Concordance Correlation Coefficient)
- Pearson r and p-value
- MAPE (Mean Absolute Percentage Error)

#### Classification Evaluation (Hypotension Risk)

```python
from utils.metrics import evaluate_classification

# Evaluate binary classification
metrics = evaluate_classification(
    y_true,
    y_pred_proba,
    threshold=0.5
)

print_evaluation_report(metrics, endpoint_name="Hypotension Risk")
```

**Metrics returned:**
- AUC-ROC
- Average Precision
- Sensitivity, Specificity
- PPV, NPV
- F1 Score, Accuracy
- Confusion Matrix

#### Evaluate All Endpoints

```python
from utils.metrics import evaluate_clinical_endpoints

results = evaluate_clinical_endpoints(
    map_true, map_pred,
    bis_true, bis_pred,
    hypotension_true, hypotension_pred
)

# results = {
#     'map': {...},
#     'bis': {...},
#     'hypotension': {...}
# }
```

## Examples

### Example 1: End-to-End Preprocessing

```python
import pandas as pd
from data.preprocessing import SignalPreprocessor
from data.feature_extraction import FeatureExtractor

# Load raw data
df = pd.read_csv('vitaldb_data/case_123.csv')

# Setup
preprocessor = SignalPreprocessor(sampling_rate=1.0)
extractor = FeatureExtractor(sampling_rate=1.0)

config = {
    'interpolate': True,
    'remove_outliers': True,
    'normalize': 'zscore'
}

# Process each signal
processed_features = []

for column in ['Solar8000/HR', 'Solar8000/NIBP_MBP', 'Solar8000/SPO2']:
    signal = df[column].values

    # Preprocess
    processed = preprocessor.preprocess_pipeline(signal, config)

    # Extract features
    features = extractor.extract_all_features(processed)

    processed_features.append(features)

print(f"Extracted {len(processed_features)} feature sets")
```

### Example 2: Batch Case Processing

```python
import os
from tqdm import tqdm

input_dir = './vitaldb_data'
output_dir = './vitaldb_processed'

os.makedirs(output_dir, exist_ok=True)

# Get all case files
case_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

# Process each case
for case_file in tqdm(case_files):
    input_path = os.path.join(input_dir, case_file)
    output_path = os.path.join(output_dir, case_file)

    preprocess_vitaldb_case(input_path, output_path, config)
```

## Running the Demos

```bash
# Run preprocessing demo
python examples/demo_preprocessing.py

# This will:
# 1. Demonstrate signal preprocessing
# 2. Show feature extraction
# 3. Perform batch processing
```

## Troubleshooting

### Common Issues

1. **Missing VitalDB data**
   ```
   Solution: Run download_vitaldb_data() first
   ```

2. **NaN in processed signals**
   ```
   Solution: Ensure 'interpolate': True in config
   ```

3. **Memory issues with large datasets**
   ```
   Solution: Process in batches or reduce n_cases
   ```

## Next Steps

1. ✅ Download VitalDB data
2. ✅ Preprocess signals
3. ✅ Extract features
4. ⏳ Train your own model (VitalNet core is proprietary)
5. ✅ Evaluate performance

## Support

For questions or issues:
- GitHub Issues: https://github.com/RegAItool/vitalnet_anesthesia/issues
- Email: yu.han@eng.ox.ac.uk

## Citation

If you use this code, please cite:

```bibtex
@article{wu2025vitalnet,
  title={VitalNet: Multimodal Triple-Endpoint Prediction and Personalized Closed-Loop Dosing in Anesthesia},
  author={Wu, Ping and Han, Yu and Wu, Xiaoqi and Zhang, Jing and Li, Yunqi and Jiang, Mengna and Wen, Qingping},
  journal={Under Review},
  year={2025}
}
```
