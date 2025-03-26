# fusionflow.py
"""
FusionFlow: A Basic Tensor and Autograd Library

This module serves as the main public entry point to the FusionFlow library,
exposing core functionalities imported from internal modules.
"""

# Import core components from the implementation module
from fusionflow_core import (
    # Core classes
    Tensor,
    Parameter,
    Module,

    # Basic NN components (consider moving to fusionflow.nn later)
    Linear,
    MSELoss,

    # Basic Optimizer (consider moving to fusionflow.optim later)
    SGD,

    # Data Types / Devices (useful constants)
    FFDataType,
    FFDevice,

    # Maybe expose key factory functions directly if desired,
    # though they are class methods on Tensor mostly.
    # e.g., from_numpy (already Tensor.from_numpy)
)

# Optional: Define __all__ to explicitly state the public API
# This affects 'from fusionflow import *' behavior and documentation tools.
__all__ = [
    'Tensor',
    'Parameter',
    'Module',
    'Linear',
    'MSELoss',
    'SGD',
    'FFDataType',
    'FFDevice',
]

# Optional: Define version information
__version__ = "0.1.0" # Example

# You could potentially add top-level convenience functions here later.

# Note: The C library loading and ctypes setup remains within fusionflow_core.py
# This keeps fusionflow.py clean and focused on exposing the API.
