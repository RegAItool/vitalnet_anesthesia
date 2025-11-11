#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VitalNet Preprocessing Demo
============================
Demonstrates the data preprocessing pipeline.

Author: VitalNet Team
License: MIT
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from data.preprocessing import SignalPreprocessor
from data.feature_extraction import FeatureExtractor


def demo_signal_preprocessing():
    """
    Demonstrate signal preprocessing on synthetic data.
    """
    print("="*70)
    print("VitalNet Signal Preprocessing Demo")
    print("="*70)

    # Generate synthetic physiological signal (e.g., heart rate)
    np.random.seed(42)
    t = np.linspace(0, 300, 300)  # 5 minutes at 1 Hz

    # Clean signal: baseline + slow oscillation
    clean_hr = 70 + 10 * np.sin(2 * np.pi * 0.01 * t)

    # Add realistic artifacts
    noisy_hr = clean_hr.copy()
    noisy_hr += np.random.normal(0, 3, len(t))  # Measurement noise
    noisy_hr[50:60] = np.nan  # Missing data
    noisy_hr[100] = 150  # Outlier (artifact)
    noisy_hr[200:210] = np.random.uniform(40, 50, 10)  # Sudden drop

    print("\n[1] Input Signal Statistics")
    print("-"*70)
    print(f"  Length:          {len(noisy_hr)} samples")
    print(f"  Mean:            {np.nanmean(noisy_hr):.2f} bpm")
    print(f"  Std:             {np.nanstd(noisy_hr):.2f} bpm")
    print(f"  Missing values:  {np.isnan(noisy_hr).sum()} samples")
    print(f"  Range:           [{np.nanmin(noisy_hr):.1f}, {np.nanmax(noisy_hr):.1f}] bpm")

    # Create preprocessor
    preprocessor = SignalPreprocessor(sampling_rate=1.0)

    # Define preprocessing configuration
    config = {
        'interpolate': True,
        'remove_outliers': True,
        'normalize': 'zscore'
    }

    print("\n[2] Preprocessing Steps")
    print("-"*70)

    # Step-by-step preprocessing
    print("  ✓ Interpolating missing values...")
    interpolated = preprocessor.interpolate_missing(noisy_hr)
    print(f"    Missing after interpolation: {np.isnan(interpolated).sum()}")

    print("  ✓ Removing outliers (1-99 percentile)...")
    cleaned = preprocessor.remove_outliers(interpolated)
    print(f"    Range after outlier removal: [{np.min(cleaned):.1f}, {np.max(cleaned):.1f}]")

    print("  ✓ Normalizing (z-score)...")
    normalized = preprocessor.normalize_signal(cleaned, method='zscore')
    print(f"    Mean: {np.mean(normalized):.4f}, Std: {np.std(normalized):.4f}")

    # Or use complete pipeline
    print("\n[3] Complete Pipeline")
    print("-"*70)
    processed = preprocessor.preprocess_pipeline(noisy_hr, config)
    print(f"  Processed signal shape: {processed.shape}")
    print(f"  Mean:                   {np.mean(processed):.4f}")
    print(f"  Std:                    {np.std(processed):.4f}")
    print(f"  Missing values:         {np.isnan(processed).sum()}")

    return processed


def demo_feature_extraction():
    """
    Demonstrate feature extraction from physiological signals.
    """
    print("\n" + "="*70)
    print("VitalNet Feature Extraction Demo")
    print("="*70)

    # Generate synthetic signal
    np.random.seed(42)
    t = np.linspace(0, 300, 300)
    hr_signal = 70 + 10 * np.sin(2 * np.pi * 0.01 * t) + np.random.normal(0, 2, len(t))

    # Create feature extractor
    extractor = FeatureExtractor(sampling_rate=1.0)

    print("\n[1] Time-Domain Features")
    print("-"*70)
    time_features = extractor.extract_time_features(hr_signal)
    for key, value in list(time_features.items())[:8]:  # Show first 8
        print(f"  {key:20s}: {value:10.4f}")
    print(f"  ... ({len(time_features)} total features)")

    print("\n[2] Frequency-Domain Features")
    print("-"*70)
    freq_features = extractor.extract_frequency_features(hr_signal)
    for key, value in freq_features.items():
        print(f"  {key:20s}: {value:10.4f}")

    print("\n[3] Physiological Features")
    print("-"*70)

    # Generate multiple signals
    map_signal = np.random.uniform(70, 90, 300)
    spo2_signal = np.random.uniform(95, 100, 300)

    physio_features = extractor.extract_physiological_features(
        hr=hr_signal,
        map_signal=map_signal,
        spo2=spo2_signal
    )

    for key, value in physio_features.items():
        print(f"  {key:20s}: {value:10.4f}")

    print("\n[4] Combined Feature Vector")
    print("-"*70)
    all_features = extractor.extract_all_features(hr_signal)
    print(f"  Total features extracted: {len(all_features)}")
    print(f"  Feature names: {list(all_features.keys())[:5]} ...")

    return all_features


def demo_batch_processing():
    """
    Demonstrate batch processing of multiple cases.
    """
    print("\n" + "="*70)
    print("VitalNet Batch Processing Demo")
    print("="*70)

    # Simulate multiple cases
    n_cases = 5
    case_features = []

    preprocessor = SignalPreprocessor(sampling_rate=1.0)
    extractor = FeatureExtractor(sampling_rate=1.0)

    config = {
        'interpolate': True,
        'remove_outliers': True,
        'normalize': 'zscore'
    }

    print(f"\nProcessing {n_cases} simulated cases...")

    for i in range(n_cases):
        # Generate synthetic case data
        np.random.seed(i)
        t = np.linspace(0, 300, 300)
        hr = 70 + np.random.uniform(-5, 5) + 10 * np.sin(2 * np.pi * 0.01 * t)
        hr += np.random.normal(0, 2, len(t))

        # Preprocess
        processed = preprocessor.preprocess_pipeline(hr, config)

        # Extract features
        features = extractor.extract_all_features(processed)

        case_features.append(features)
        print(f"  ✓ Case {i+1}: {len(features)} features extracted")

    # Convert to DataFrame
    df_features = pd.DataFrame(case_features)

    print(f"\nBatch processing completed!")
    print(f"  Feature matrix shape: {df_features.shape}")
    print(f"  Columns: {list(df_features.columns)[:5]} ...")

    return df_features


if __name__ == '__main__':
    # Run all demos

    # Demo 1: Signal preprocessing
    processed_signal = demo_signal_preprocessing()

    # Demo 2: Feature extraction
    features = demo_feature_extraction()

    # Demo 3: Batch processing
    feature_df = demo_batch_processing()

    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Download real VitalDB data using data/download_vitaldb.py")
    print("  2. Preprocess your data using the SignalPreprocessor")
    print("  3. Extract features using the FeatureExtractor")
    print("  4. Train your own models using the extracted features")
    print("\nNote: The VitalNet core model is proprietary and not included.")
    print("="*70)
