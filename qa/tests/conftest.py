from __future__ import annotations

import os


# Keep native math / torch runtime deterministic and avoid shared-memory
# crashes in constrained test environments before any test imports server/app.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
