"""
eval package

This folder contains evaluation / analysis scripts such as:
- metrics.py (window + event evaluation)
- fit_ops.py (fit OP-1/2/3 thresholds under real alert policy)
- replay_eval.py (deployment-like streaming evaluation)
- mining scripts (hard negatives, near-miss negatives)
- plotting scripts

--------------------------------------
Python executes eval/__init__.py whenever you do:

    import eval

If we import heavy modules here (torch, matplotlib, etc.),
it can slow startup and cause circular imports.

So we only keep package metadata here.
"""

__all__ = []
