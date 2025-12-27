"""Server package for the Fall Detection project.

This package exposes a FastAPI app that:
- receives skeleton windows from the front-end,
- runs inference using TCN / GCN / TCN+GCN modes,
- applies triage + temporal logic to produce alerts,
- (optionally) persists events to a DB.
"""
