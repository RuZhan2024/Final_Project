"""
core package

This folder contains the reusable "building blocks" used across your repo:
- preprocessing (resample, smoothing, normalization)
- feature building (convert NPZ windows -> model tensors)
- models (TCN / GCN)
- checkpointing + EMA
- metrics + alerting logic
- calibration / yaml IO helpers

-----------------------------------
Python executes __init__.py whenever you import the package, e.g.:

    import core
    from core import preprocess

If we import many submodules here (like `from .models import ...`),
it can:
1) slow down import time
2) create circular-import errors (common in bigger projects)

So we only define package metadata here.
"""

# Package version string (useful when printing debug info or saving in checkpoints).
__version__ = "0.1.0"

# Optional: define what `from core import *` exports.
# We keep it empty to avoid accidental side-effects.
__all__ = ["__version__"]
