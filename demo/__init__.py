"""
VitalNet Demo Module
====================
Demonstration models and pseudocode for academic reproducibility.

This module provides:
1. Toy model for method validation
2. High-level pseudocode for the VitalNet approach
3. Synthetic data generation for testing

Note: These are simplified demonstrations. The full production
VitalNet system is proprietary and will be released upon paper acceptance.
"""

from .toy_model import SimplifiedVitalNet, create_toy_dataset, train_toy_model

__all__ = [
    'SimplifiedVitalNet',
    'create_toy_dataset',
    'train_toy_model',
]
