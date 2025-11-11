#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VitalNet Data Preprocessing Pipeline
======================================
Signal preprocessing and quality control for anesthesia monitoring data.

Author: VitalNet Team
License: MIT
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


class SignalPreprocessor:
    """
    Preprocessing pipeline for physiological signals.
    """

    def __init__(self, sampling_rate=1.0):
        """
        Initialize preprocessor.

        Parameters
        ----------
        sampling_rate : float
            Target sampling rate in Hz (default: 1.0 for vital signs)
        """
        self.sampling_rate = sampling_rate


    def remove_outliers(self, signal_data, lower_percentile=1, upper_percentile=99):
        """
        Remove outliers using percentile-based clipping.

        Parameters
        ----------
        signal_data : array-like
            Input signal
        lower_percentile : float
            Lower percentile for clipping (default: 1)
        upper_percentile : float
            Upper percentile for clipping (default: 99)

        Returns
        -------
        cleaned_signal : ndarray
            Signal with outliers removed
        """
        if len(signal_data) == 0:
            return signal_data

        lower_bound = np.percentile(signal_data, lower_percentile)
        upper_bound = np.percentile(signal_data, upper_percentile)

        cleaned_signal = np.clip(signal_data, lower_bound, upper_bound)

        return cleaned_signal


    def interpolate_missing(self, signal_data, method='linear'):
        """
        Interpolate missing values (NaN) in signal.

        Parameters
        ----------
        signal_data : array-like
            Input signal with possible NaN values
        method : str
            Interpolation method ('linear', 'cubic', 'nearest')

        Returns
        -------
        interpolated_signal : ndarray
            Signal with NaN values interpolated
        """
        signal_array = np.array(signal_data, dtype=float)

        # Find NaN positions
        nan_mask = np.isnan(signal_array)

        if not nan_mask.any():
            return signal_array

        # Get valid indices
        valid_indices = np.where(~nan_mask)[0]

        if len(valid_indices) < 2:
            # Not enough valid points for interpolation
            return np.nan_to_num(signal_array, nan=0.0)

        # Interpolate
        interp_func = interp1d(
            valid_indices,
            signal_array[valid_indices],
            kind=method,
            bounds_error=False,
            fill_value='extrapolate'
        )

        all_indices = np.arange(len(signal_array))
        interpolated_signal = interp_func(all_indices)

        return interpolated_signal


    def apply_bandpass_filter(self, signal_data, lowcut, highcut, order=4):
        """
        Apply Butterworth bandpass filter.

        Parameters
        ----------
        signal_data : array-like
            Input signal
        lowcut : float
            Low cutoff frequency (Hz)
        highcut : float
            High cutoff frequency (Hz)
        order : int
            Filter order (default: 4)

        Returns
        -------
        filtered_signal : ndarray
            Bandpass filtered signal
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = signal.butter(order, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, signal_data)

        return filtered_signal


    def normalize_signal(self, signal_data, method='zscore'):
        """
        Normalize signal using various methods.

        Parameters
        ----------
        signal_data : array-like
            Input signal
        method : str
            Normalization method ('zscore', 'minmax', 'robust')

        Returns
        -------
        normalized_signal : ndarray
            Normalized signal
        """
        signal_array = np.array(signal_data, dtype=float)

        if method == 'zscore':
            mean = np.mean(signal_array)
            std = np.std(signal_array)
            if std > 0:
                normalized_signal = (signal_array - mean) / std
            else:
                normalized_signal = signal_array - mean

        elif method == 'minmax':
            min_val = np.min(signal_array)
            max_val = np.max(signal_array)
            if max_val > min_val:
                normalized_signal = (signal_array - min_val) / (max_val - min_val)
            else:
                normalized_signal = signal_array - min_val

        elif method == 'robust':
            median = np.median(signal_array)
            iqr = np.percentile(signal_array, 75) - np.percentile(signal_array, 25)
            if iqr > 0:
                normalized_signal = (signal_array - median) / iqr
            else:
                normalized_signal = signal_array - median

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized_signal


    def preprocess_pipeline(self, signal_data, config=None):
        """
        Complete preprocessing pipeline.

        Parameters
        ----------
        signal_data : array-like
            Input signal
        config : dict, optional
            Configuration dictionary with keys:
            - 'interpolate': bool
            - 'remove_outliers': bool
            - 'bandpass': dict with 'lowcut', 'highcut'
            - 'normalize': str ('zscore', 'minmax', 'robust')

        Returns
        -------
        processed_signal : ndarray
            Fully preprocessed signal
        """
        if config is None:
            config = {
                'interpolate': True,
                'remove_outliers': True,
                'normalize': 'zscore'
            }

        processed = np.array(signal_data, dtype=float)

        # Step 1: Interpolate missing values
        if config.get('interpolate', False):
            processed = self.interpolate_missing(processed)

        # Step 2: Remove outliers
        if config.get('remove_outliers', False):
            processed = self.remove_outliers(processed)

        # Step 3: Bandpass filter (optional)
        if 'bandpass' in config:
            lowcut = config['bandpass']['lowcut']
            highcut = config['bandpass']['highcut']
            processed = self.apply_bandpass_filter(processed, lowcut, highcut)

        # Step 4: Normalize
        if 'normalize' in config:
            processed = self.normalize_signal(processed, method=config['normalize'])

        return processed


def preprocess_vitaldb_case(case_file, output_file=None, config=None):
    """
    Preprocess a single VitalDB case file.

    Parameters
    ----------
    case_file : str
        Path to input CSV file
    output_file : str, optional
        Path to save preprocessed data
    config : dict, optional
        Preprocessing configuration

    Returns
    -------
    preprocessed_df : DataFrame
        Preprocessed data
    """
    # Load data
    df = pd.read_csv(case_file)

    # Initialize preprocessor
    preprocessor = SignalPreprocessor(sampling_rate=1.0)

    # Preprocess each column
    preprocessed_data = {}

    for column in df.columns:
        if column == 'Time':
            preprocessed_data['Time'] = df['Time'].values
        else:
            signal_data = df[column].values
            processed = preprocessor.preprocess_pipeline(signal_data, config)
            preprocessed_data[column] = processed

    preprocessed_df = pd.DataFrame(preprocessed_data)

    # Save if output path provided
    if output_file:
        preprocessed_df.to_csv(output_file, index=False)
        print(f"Preprocessed data saved to {output_file}")

    return preprocessed_df


if __name__ == '__main__':
    # Example usage

    # Generate sample signal with noise and missing values
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    clean_signal = 70 + 10 * np.sin(2 * np.pi * 0.1 * t)

    # Add noise and missing values
    noisy_signal = clean_signal + np.random.normal(0, 2, len(t))
    noisy_signal[50:60] = np.nan  # Missing values
    noisy_signal[100] = 150  # Outlier

    # Preprocess
    preprocessor = SignalPreprocessor(sampling_rate=10.0)

    config = {
        'interpolate': True,
        'remove_outliers': True,
        'normalize': 'zscore'
    }

    processed_signal = preprocessor.preprocess_pipeline(noisy_signal, config)

    print("Preprocessing demo completed!")
    print(f"Input shape: {noisy_signal.shape}")
    print(f"Output shape: {processed_signal.shape}")
    print(f"NaN count - Before: {np.isnan(noisy_signal).sum()}, After: {np.isnan(processed_signal).sum()}")
