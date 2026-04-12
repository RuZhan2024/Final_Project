#!/usr/bin/env python3
"""Check release-critical artifacts for demo/thesis handoff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


REQUIRED_FILES: List[str] = [
    "docs/project_targets/DEPLOYMENT_LOCK.md",
    "artifacts/reports/deployment_lock_validation.md",
    "artifacts/reports/release_snapshot.md",
    "docs/project_targets/THESIS_EVIDENCE_MAP.md",
    "artifacts/reports/tuning/overfit_round1_summary.md",
    "docs/project_targets/GCN_POLICY_ROUND2_RESULTS.md",
    "artifacts/reports/tuning/tcn_round2_results.md",
    "artifacts/registry/overfit_experiment_registry.csv",
    "configs/ops/tcn_caucafall.yaml",
]


def _contains(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return needle in txt


def main() -> None:
    root = Path(".").resolve()
    out_json = root / "artifacts/reports/release_bundle_status.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    checks: List[Dict[str, object]] = []
    for rel in REQUIRED_FILES:
        p = root / rel
        checks.append(
            {
                "id": f"exists:{rel}",
                "path": rel,
                "ok": p.exists(),
                "detail": "found" if p.exists() else "missing",
            }
        )

    lock_validation = root / "artifacts/reports/deployment_lock_validation.md"
    checks.append(
        {
            "id": "lock_validation_pass",
            "path": str(lock_validation.relative_to(root)),
            "ok": _contains(lock_validation, "[x] PASS"),
            "detail": "PASS checked" if _contains(lock_validation, "[x] PASS") else "PASS not checked",
        }
    )

    evidence_map = root / "docs/project_targets/THESIS_EVIDENCE_MAP.md"
    for marker in [
        "Tab-Overfit-Round1-13",
        "Tab-GCN-Policy-R2-14",
        "Tab-TCN-Train-R2-15",
        "Tab-Deploy-Lock-16",
    ]:
        ok = _contains(evidence_map, marker)
        checks.append(
            {
                "id": f"evidence:{marker}",
                "path": str(evidence_map.relative_to(root)),
                "ok": ok,
                "detail": "present" if ok else "missing",
            }
        )

    passed = sum(1 for c in checks if c["ok"])
    total = len(checks)
    status = {
        "ok": passed == total,
        "passed": passed,
        "total": total,
        "checks": checks,
    }
    out_json.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out_json}")
    print(f"[summary] passed={passed}/{total}")


if __name__ == "__main__":
    main()
