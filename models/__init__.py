"""
VitalNet Model Framework
=========================
Model interfaces and base classes.

IMPORTANT: This module contains only interface definitions and stubs.
The core Transformer-CNN fusion architecture and MPC implementation
are proprietary and will be released after paper publication.
"""

from .base_model import (
    BasePredictor,
    MultiEndpointPredictor,
    PersonalizedDosingOptimizer,
    VitalNetStub
)

__all__ = [
    'BasePredictor',
    'MultiEndpointPredictor',
    'PersonalizedDosingOptimizer',
    'VitalNetStub',
]

# Version info
__version__ = '0.1.0-alpha'
__status__ = 'Under Review - Partial Release'
