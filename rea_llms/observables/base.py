from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObservableScore:
    value: float
    metadata: dict[str, Any]
