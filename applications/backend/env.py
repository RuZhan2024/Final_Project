from __future__ import annotations

"""Lightweight `.env` loader used by backend config startup."""

import os

from pathlib import Path


_ENV_LOADED = False


def load_local_env_files() -> None:
    """Load local env files once, without overriding already-exported variables."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    repo_root = Path(__file__).resolve().parents[1]
    for name in (".env.local.private", ".env.local", ".env"):
        # More specific local files win because earlier loads populate os.environ first.
        path = repo_root / name
        if not path.exists():
            continue
        try:
            for raw_line in path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key or key in os.environ:
                    continue
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", "\""}:
                    value = value[1:-1]
                os.environ[key] = value
        except OSError:
            continue

    _ENV_LOADED = True
