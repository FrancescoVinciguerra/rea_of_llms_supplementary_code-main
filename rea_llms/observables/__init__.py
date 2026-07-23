from __future__ import annotations

from rea_llms.observables.ari import ARIObservable


def create_observable(name: str, **kwargs):
    normalized = name.lower().replace("-", "_")
    if normalized in {"ari", "automated_readability_index"}:
        return ARIObservable(**kwargs)
    raise ValueError(f"Unknown observable: {name!r}")


__all__ = ["ARIObservable", "create_observable"]
