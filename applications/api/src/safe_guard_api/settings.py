from __future__ import annotations

from dataclasses import dataclass
from os import getenv


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class ApiSettings:
    app_name: str = "Safe Guard API"
    app_version: str = "0.1.0"
    allowed_origins: tuple[str, ...] = ("http://localhost:3000",)

    @classmethod
    def from_env(cls) -> "ApiSettings":
        origins = getenv("SAFE_GUARD_ALLOWED_ORIGINS")
        return cls(
            allowed_origins=_split_csv(origins) if origins else cls.allowed_origins,
        )
