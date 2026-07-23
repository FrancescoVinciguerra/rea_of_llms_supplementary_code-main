from __future__ import annotations

from dataclasses import dataclass, field

from textstat import automated_readability_index

from rea_llms.observables.base import ObservableScore


@dataclass
class ARIObservable:
    mode: str = "capped"
    cap: float = 15.0
    backend: str = "exact_cached"
    cache_enabled: bool = True
    name: str = "ari"
    _cache: dict[str, ObservableScore] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in {"raw", "capped"}:
            raise ValueError("mode must be one of {'raw', 'capped'}")
        if self.backend not in {"exact", "exact_cached", "textstat"}:
            raise ValueError("ARI backend must be one of {'exact', 'exact_cached', 'textstat'}")

    def score_text(self, text: str) -> ObservableScore:
        if self.cache_enabled and self.backend == "exact_cached" and text in self._cache:
            return self._cache[text]
        raw = float(automated_readability_index(text))
        capped = min(raw, float(self.cap))
        value = raw if self.mode == "raw" else capped
        score = ObservableScore(
            value=float(value),
            metadata={
                "raw_ari": float(raw),
                "capped_ari": float(capped),
                "ari_mode": self.mode,
                "ari_cap": float(self.cap),
                "backend": self.backend,
            },
        )
        if self.cache_enabled and self.backend == "exact_cached":
            self._cache[text] = score
        return score

    def score_completion(self, prompt: str, completion: str) -> ObservableScore:
        return self.score_text(prompt + completion)

    def score_token_sequences(self, *, tokenizer, prompt: str, token_sequences: list[list[int]], prompt_token_count: int):
        scores = []
        for ids in token_sequences:
            completion = tokenizer.decode(ids[prompt_token_count:], skip_special_tokens=True)
            scores.append(self.score_completion(prompt, completion))
        return scores
