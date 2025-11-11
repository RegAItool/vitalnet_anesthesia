# VitalNet Data Format Specification

## Overview

This document describes the data formats used in VitalNet for anesthesia monitoring.

## VitalDB Data Structure

### Raw Data Format

VitalDB data is stored in CSV format with the following structure:

```csv
Time,Solar8000/HR,Solar8000/NIBP_MBP,Solar8000/SPO2,BIS/BIS,...
0.0,72.5,85.3,98.2,45.6,...
1.0,73.1,84.9,98.1,46.2,...
2.0,72.8,85.7,98.3,45.8,...
...
```

**Columns:**
- `Time`: Timestamp in seconds from start of recording
- `Solar8000/HR`: Heart rate (bpm)
- `Solar8000/NIBP_SBP`: Non-invasive systolic blood pressure (mmHg)
- `Solar8000/NIBP_DBP`: Non-invasive diastolic blood pressure (mmHg)
- `Solar8000/NIBP_MBP`: Non-invasive mean blood pressure (mmHg)
- `Solar8000/SPO2`: Oxygen saturation (%)
- `Solar8000/BT`: Body temperature (°C)
- `BIS/BIS`: Bispectral index (0-100)
- `Orchestra/RFTN20_CE`: Remifentanil effect-site concentration (ng/ml)
- `Orchestra/PPF20_CE`: Propofol effect-site concentration (μg/ml)

### Waveform Data Format

High-frequency waveform data (ECG, arterial pressure) is stored in NumPy `.npy` format:

```python
# Shape: (n_samples,)
# Sampling rates:
# - ECG: 500 Hz
# - Arterial Pressure: 100 Hz
# - Plethysmography: 100 Hz
```

## Preprocessed Data Format

After preprocessing, data maintains the same CSV structure but with:
- No missing values (interpolated)
- Outliers removed
- Optional normalization applied

## Feature Matrix Format

Extracted features are organized in a pandas DataFrame:

```python
# Shape: (n_samples, n_features)
# Columns: ['time_mean', 'time_std', ..., 'freq_dominant_freq', ...]
```

### Feature Groups

1. **Time-Domain Features** (14 features per signal)
   - Basic statistics: mean, std, median, q25, q75
   - Variability: range, IQR, CV, MAD
   - Derivative: mean absolute difference, RMS
   - Complexity: zero-crossing rate, kurtosis, skewness

2. **Frequency-Domain Features** (6 features per signal)
   - Dominant frequency and power
   - Total power
   - Spectral centroid and spread
   - Spectral entropy

3. **Physiological Features** (domain-specific)
   - HR trend, MAP trend
   - Hypotension risk indicator
   - Hypoxia risk indicator

## Target Variables

### Regression Targets
- **MAP (Mean Arterial Pressure)**: Continuous value in mmHg
- **BIS (Bispectral Index)**: Continuous value 0-100

### Classification Target
- **Hypotension Risk**: Binary (0 or 1)
  - 1 if MAP < 65 mmHg within next 5 minutes
  - 0 otherwise

## Multi-Modal Input Format

For models using multiple modalities:

```python
{
    'numerical': np.array([...]),  # Shape: (n_samples, n_features)
    'waveform': {
        'ecg': np.array([...]),    # Shape: (n_samples, sequence_length)
        'art': np.array([...]),    # Shape: (n_samples, sequence_length)
    }
}
```

## Data Quality Indicators

Each sample may include quality flags:

- `missing_ratio`: Proportion of missing values in time window
- `artifact_detected`: Boolean flag for detected artifacts
- `signal_quality_index`: 0-1 score for overall signal quality

## Clinical Context

### Patient Demographics (Optional)
- Age (years)
- Weight (kg)
- Height (cm)
- Sex (M/F)
- ASA score (I-V)

### Surgical Context (Optional)
- Surgery type
- Anesthesia type
- Duration

## Example Usage

```python
import pandas as pd
import numpy as np

# Load preprocessed data
df = pd.read_csv('vitaldb_processed/case_123.csv')

# Extract features
from data.feature_extraction import FeatureExtractor

extractor = FeatureExtractor(sampling_rate=1.0)
features = extractor.extract_all_features(df['Solar8000/HR'].values)

# Create input for model
X = np.array([features[key] for key in sorted(features.keys())])
```

## Data Availability

VitalDB data is publicly available at: https://vitaldb.net

**Citation:**
```
Lee, H. C., & Jung, C. W. (2018). VitalDB, a high-fidelity multi-parameter
vital signs database in surgical patients. Scientific Data, 5(1), 1-8.
```

## Notes

- All timestamps are synchronized across modalities
- Missing data is indicated by NaN values in CSV
- Sampling rates may vary; check track metadata
- Some tracks may not be available for all cases
