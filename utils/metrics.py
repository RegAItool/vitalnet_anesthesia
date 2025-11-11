#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VitalNet Evaluation Metrics
============================
Comprehensive metrics for anesthesia monitoring model evaluation.

Author: VitalNet Team
License: MIT
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from scipy.stats import pearsonr


def concordance_correlation_coefficient(y_true, y_pred):
    """
    Calculate Lin's Concordance Correlation Coefficient (CCC).

    CCC measures agreement between predicted and true values.
    Commonly used in medical prediction validation.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns
    -------
    ccc : float
        Concordance correlation coefficient
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Remove NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return 0.0

    # Pearson correlation
    cor, _ = pearsonr(y_true, y_pred)

    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    # Variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    # Standard deviations
    sd_true = np.sqrt(var_true)
    sd_pred = np.sqrt(var_pred)

    # CCC formula
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    ccc = numerator / denominator if denominator != 0 else 0.0

    return ccc


def evaluate_regression(y_true, y_pred):
    """
    Comprehensive regression metrics for continuous predictions (MAP, BIS).

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns
    -------
    metrics : dict
        Dictionary of regression metrics
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Remove NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return {
            'mae': 0.0, 'rmse': 0.0, 'r2': 0.0, 'ccc': 0.0,
            'pearson_r': 0.0, 'mape': 0.0, 'n_samples': 0
        }

    metrics = {}

    # Mean Absolute Error
    metrics['mae'] = mean_absolute_error(y_true, y_pred)

    # Root Mean Squared Error
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

    # R-squared
    metrics['r2'] = r2_score(y_true, y_pred)

    # Concordance Correlation Coefficient
    metrics['ccc'] = concordance_correlation_coefficient(y_true, y_pred)

    # Pearson correlation
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    metrics['pearson_r'] = pearson_r
    metrics['pearson_p'] = pearson_p

    # Mean Absolute Percentage Error
    nonzero_mask = y_true != 0
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
        metrics['mape'] = mape
    else:
        metrics['mape'] = 0.0

    # Sample count
    metrics['n_samples'] = len(y_true)

    return metrics


def evaluate_classification(y_true, y_pred_proba, threshold=0.5):
    """
    Classification metrics for binary prediction (hypotension risk).

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1)
    y_pred_proba : array-like
        Predicted probabilities
    threshold : float
        Classification threshold (default: 0.5)

    Returns
    -------
    metrics : dict
        Dictionary of classification metrics
    """
    y_true = np.array(y_true).flatten()
    y_pred_proba = np.array(y_pred_proba).flatten()

    # Remove NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred_proba))
    y_true = y_true[mask].astype(int)
    y_pred_proba = y_pred_proba[mask]

    if len(y_true) < 2:
        return {'auc': 0.0, 'ap': 0.0, 'n_samples': 0}

    # Binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {}

    # AUC-ROC
    try:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    except:
        metrics['auc'] = 0.0

    # Average Precision (AP)
    try:
        metrics['ap'] = average_precision_score(y_true, y_pred_proba)
    except:
        metrics['ap'] = 0.0

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Sensitivity (Recall, True Positive Rate)
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity (True Negative Rate)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Precision (Positive Predictive Value)
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Negative Predictive Value
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # F1 Score
    metrics['f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # Accuracy
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)

    # Confusion matrix components
    metrics['tp'] = int(tp)
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)

    # Sample count
    metrics['n_samples'] = len(y_true)

    return metrics


def evaluate_clinical_endpoints(map_true, map_pred, bis_true, bis_pred,
                                  hypotension_true, hypotension_pred):
    """
    Evaluate all three clinical endpoints used in VitalNet.

    Parameters
    ----------
    map_true : array-like
        True MAP values
    map_pred : array-like
        Predicted MAP values
    bis_true : array-like
        True BIS values
    bis_pred : array-like
        Predicted BIS values
    hypotension_true : array-like
        True hypotension labels (binary)
    hypotension_pred : array-like
        Predicted hypotension probabilities

    Returns
    -------
    results : dict
        Comprehensive evaluation results for all endpoints
    """
    results = {}

    # MAP prediction metrics
    print("Evaluating MAP prediction...")
    results['map'] = evaluate_regression(map_true, map_pred)

    # BIS prediction metrics
    print("Evaluating BIS prediction...")
    results['bis'] = evaluate_regression(bis_true, bis_pred)

    # Hypotension risk classification
    print("Evaluating hypotension risk...")
    results['hypotension'] = evaluate_classification(hypotension_true, hypotension_pred)

    return results


def print_evaluation_report(metrics, endpoint_name="Model"):
    """
    Print formatted evaluation report.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from evaluate_regression or evaluate_classification
    endpoint_name : str
        Name of the endpoint being evaluated
    """
    print("\n" + "="*60)
    print(f"{endpoint_name} Evaluation Report")
    print("="*60)

    if 'mae' in metrics:  # Regression metrics
        print("\nRegression Metrics:")
        print(f"  MAE:         {metrics['mae']:.4f}")
        print(f"  RMSE:        {metrics['rmse']:.4f}")
        print(f"  R²:          {metrics['r2']:.4f}")
        print(f"  CCC:         {metrics['ccc']:.4f}")
        print(f"  Pearson r:   {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4e})")
        print(f"  MAPE:        {metrics['mape']:.2f}%")

    elif 'auc' in metrics:  # Classification metrics
        print("\nClassification Metrics:")
        print(f"  AUC-ROC:     {metrics['auc']:.4f}")
        print(f"  AP:          {metrics['ap']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  PPV:         {metrics['ppv']:.4f}")
        print(f"  NPV:         {metrics['npv']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TP: {metrics['tp']:4d}  FP: {metrics['fp']:4d}")
        print(f"    FN: {metrics['fn']:4d}  TN: {metrics['tn']:4d}")

    print(f"\n  Samples:     {metrics['n_samples']}")
    print("="*60)


if __name__ == '__main__':
    # Example usage

    np.random.seed(42)

    # Generate synthetic data
    n_samples = 1000

    # Regression example (MAP prediction)
    map_true = np.random.uniform(60, 100, n_samples)
    map_pred = map_true + np.random.normal(0, 8, n_samples)

    # Classification example (hypotension)
    hypotension_true = (map_true < 65).astype(int)
    hypotension_pred = 1 / (1 + np.exp(-(-5 + 0.1 * map_true)))  # Sigmoid

    print("VitalNet Metrics Demo")

    # Regression evaluation
    map_metrics = evaluate_regression(map_true, map_pred)
    print_evaluation_report(map_metrics, "MAP Prediction")

    # Classification evaluation
    hypotension_metrics = evaluate_classification(hypotension_true, hypotension_pred)
    print_evaluation_report(hypotension_metrics, "Hypotension Risk")
