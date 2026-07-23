from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SampleRecord:
    text: str
    completion: str
    phi: float
    raw_ari: float | None = None
    capped_ari: float | None = None
    log_weight: float | None = None
    normalized_weight: float | None = None
    token_ids: list[int] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
