#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VitalNet Feature Extraction
============================
Extract time-domain and frequency-domain features from physiological signals.

Author: VitalNet Team
License: MIT
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    """
    Extract features from physiological signals.
    """

    def __init__(self, sampling_rate=1.0):
        """
        Initialize feature extractor.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate of the signal in Hz
        """
        self.sampling_rate = sampling_rate


    def extract_time_features(self, signal_data):
        """
        Extract time-domain statistical features.

        Parameters
        ----------
        signal_data : array-like
            Input signal

        Returns
        -------
        features : dict
            Dictionary of time-domain features
        """
        signal_array = np.array(signal_data, dtype=float)

        if len(signal_array) < 5:
            return self._empty_time_features()

        features = {}

        # Basic statistics
        features['mean'] = np.mean(signal_array)
        features['std'] = np.std(signal_array)
        features['median'] = np.median(signal_array)
        features['q25'] = np.percentile(signal_array, 25)
        features['q75'] = np.percentile(signal_array, 75)
        features['range'] = np.max(signal_array) - np.min(signal_array)
        features['iqr'] = features['q75'] - features['q25']

        # Variability measures
        features['cv'] = features['std'] / features['mean'] if features['mean'] != 0 else 0
        features['mad'] = np.mean(np.abs(signal_array - features['mean']))

        # Derivative features
        diff_signal = np.diff(signal_array)
        features['mean_abs_diff'] = np.mean(np.abs(diff_signal))
        features['rms'] = np.sqrt(np.mean(signal_array ** 2))

        # Zero-crossing rate
        zero_crossings = np.where(np.diff(np.sign(signal_array - features['mean'])))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal_array)

        # Higher-order statistics
        features['kurtosis'] = stats.kurtosis(signal_array)
        features['skewness'] = stats.skew(signal_array)

        return features


    def extract_frequency_features(self, signal_data):
        """
        Extract frequency-domain features using FFT.

        Parameters
        ----------
        signal_data : array-like
            Input signal

        Returns
        -------
        features : dict
            Dictionary of frequency-domain features
        """
        signal_array = np.array(signal_data, dtype=float)

        if len(signal_array) < 10:
            return self._empty_frequency_features()

        features = {}

        # Detrend signal
        try:
            detrended = signal.detrend(signal_array)
        except:
            detrended = signal_array

        # FFT
        n = len(detrended)
        fft_vals = fft(detrended)
        fft_freq = fftfreq(n, d=1.0/self.sampling_rate)

        # Only positive frequencies
        positive_freq_idx = fft_freq > 0
        fft_magnitude = np.abs(fft_vals[positive_freq_idx])
        frequencies = fft_freq[positive_freq_idx]

        if len(fft_magnitude) == 0:
            return self._empty_frequency_features()

        # Power spectral density
        psd = fft_magnitude ** 2
        total_power = np.sum(psd)

        # Dominant frequency
        dominant_idx = np.argmax(fft_magnitude)
        features['dominant_freq'] = frequencies[dominant_idx]
        features['dominant_power'] = fft_magnitude[dominant_idx]

        # Frequency bands (example for EEG-like analysis)
        # Note: Adjust bands based on your specific signal type
        features['total_power'] = total_power
        features['spectral_centroid'] = np.sum(frequencies * psd) / total_power if total_power > 0 else 0
        features['spectral_spread'] = np.sqrt(
            np.sum(((frequencies - features['spectral_centroid']) ** 2) * psd) / total_power
        ) if total_power > 0 else 0

        # Spectral entropy
        psd_norm = psd / total_power if total_power > 0 else psd
        psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
        features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm)) if len(psd_norm) > 0 else 0

        return features


    def extract_physiological_features(self, hr, map_signal, spo2):
        """
        Extract physiological domain-specific features.

        Parameters
        ----------
        hr : array-like
            Heart rate signal
        map_signal : array-like
            Mean arterial pressure signal
        spo2 : array-like
            Oxygen saturation signal

        Returns
        -------
        features : dict
            Dictionary of physiological features
        """
        features = {}

        # Heart rate features
        if hr is not None and len(hr) > 0:
            hr_array = np.array(hr, dtype=float)
            features['hr_mean'] = np.mean(hr_array)
            features['hr_std'] = np.std(hr_array)
            features['hr_trend'] = self._calculate_trend(hr_array)

        # MAP features
        if map_signal is not None and len(map_signal) > 0:
            map_array = np.array(map_signal, dtype=float)
            features['map_mean'] = np.mean(map_array)
            features['map_std'] = np.std(map_array)
            features['map_trend'] = self._calculate_trend(map_array)

            # Hypotension risk indicator
            features['hypotension_risk'] = np.mean(map_array < 65)

        # SpO2 features
        if spo2 is not None and len(spo2) > 0:
            spo2_array = np.array(spo2, dtype=float)
            features['spo2_mean'] = np.mean(spo2_array)
            features['spo2_min'] = np.min(spo2_array)
            features['hypoxia_risk'] = np.mean(spo2_array < 90)

        return features


    def _calculate_trend(self, signal_data):
        """
        Calculate linear trend of signal.

        Parameters
        ----------
        signal_data : array-like
            Input signal

        Returns
        -------
        slope : float
            Slope of linear regression
        """
        if len(signal_data) < 2:
            return 0.0

        x = np.arange(len(signal_data))
        y = np.array(signal_data, dtype=float)

        # Remove NaN
        valid_mask = ~np.isnan(y)
        if np.sum(valid_mask) < 2:
            return 0.0

        x = x[valid_mask]
        y = y[valid_mask]

        # Linear regression
        slope, _ = np.polyfit(x, y, 1)

        return slope


    def _empty_time_features(self):
        """Return empty time-domain features."""
        return {
            'mean': 0, 'std': 0, 'median': 0, 'q25': 0, 'q75': 0,
            'range': 0, 'iqr': 0, 'cv': 0, 'mad': 0, 'mean_abs_diff': 0,
            'rms': 0, 'zero_crossing_rate': 0, 'kurtosis': 0, 'skewness': 0
        }


    def _empty_frequency_features(self):
        """Return empty frequency-domain features."""
        return {
            'dominant_freq': 0, 'dominant_power': 0, 'total_power': 0,
            'spectral_centroid': 0, 'spectral_spread': 0, 'spectral_entropy': 0
        }


    def extract_all_features(self, signal_data):
        """
        Extract both time and frequency domain features.

        Parameters
        ----------
        signal_data : array-like
            Input signal

        Returns
        -------
        features : dict
            Combined feature dictionary
        """
        time_features = self.extract_time_features(signal_data)
        freq_features = self.extract_frequency_features(signal_data)

        # Combine with prefix
        all_features = {}
        for key, value in time_features.items():
            all_features[f'time_{key}'] = value
        for key, value in freq_features.items():
            all_features[f'freq_{key}'] = value

        return all_features


def extract_features(signal_data, feature_types=['time', 'frequency'], sampling_rate=1.0):
    """
    Convenience function to extract features from signal.

    Parameters
    ----------
    signal_data : array-like
        Input signal
    feature_types : list
        List of feature types to extract ('time', 'frequency')
    sampling_rate : float
        Sampling rate in Hz

    Returns
    -------
    features : dict
        Extracted features
    """
    extractor = FeatureExtractor(sampling_rate=sampling_rate)

    features = {}

    if 'time' in feature_types:
        time_features = extractor.extract_time_features(signal_data)
        features.update({f'time_{k}': v for k, v in time_features.items()})

    if 'frequency' in feature_types:
        freq_features = extractor.extract_frequency_features(signal_data)
        features.update({f'freq_{k}': v for k, v in freq_features.items()})

    return features


if __name__ == '__main__':
    # Example usage

    np.random.seed(42)

    # Generate synthetic heart rate signal
    t = np.linspace(0, 300, 300)  # 5 minutes at 1 Hz
    hr_signal = 70 + 10 * np.sin(2 * np.pi * 0.01 * t) + np.random.normal(0, 2, len(t))

    # Extract features
    extractor = FeatureExtractor(sampling_rate=1.0)

    print("="*60)
    print("Feature Extraction Demo")
    print("="*60)

    # Time-domain features
    print("\n[Time-Domain Features]")
    time_features = extractor.extract_time_features(hr_signal)
    for key, value in time_features.items():
        print(f"  {key:20s}: {value:.4f}")

    # Frequency-domain features
    print("\n[Frequency-Domain Features]")
    freq_features = extractor.extract_frequency_features(hr_signal)
    for key, value in freq_features.items():
        print(f"  {key:20s}: {value:.4f}")

    # All features
    print("\n[Combined Features]")
    all_features = extractor.extract_all_features(hr_signal)
    print(f"  Total features extracted: {len(all_features)}")
