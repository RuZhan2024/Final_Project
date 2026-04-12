from __future__ import annotations

from typing import Optional


SUPPORTED_DATASETS = {"caucafall", "le2i"}
SUPPORTED_MODEL_CODES = {"TCN", "GCN", "HYBRID"}


def normalize_dataset_code(dataset_code: Optional[str], default: str = "caucafall") -> str:
    ds = str(dataset_code or "").lower().strip()
    if ds in SUPPORTED_DATASETS:
        return ds
    return default


def normalize_model_code(model_code: Optional[str], default: str = "TCN") -> str:
    mc = str(model_code or "").upper().strip()
    if mc in SUPPORTED_MODEL_CODES:
        return mc
    return default


def norm_op_code(op_code: Optional[str]) -> str:
    s = (op_code or "").strip().upper().replace("_", "-")
    if s in {"OP1", "OP-1", "HIGH", "OP-01"}:
        return "OP-1"
    if s in {"OP2", "OP-2", "BALANCED", "OP-02"}:
        return "OP-2"
    if s in {"OP3", "OP-3", "LOW", "OP-03"}:
        return "OP-3"
    return "OP-2"
