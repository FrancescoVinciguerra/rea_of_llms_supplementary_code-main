from __future__ import annotations

import time
from typing import Any

import torch

from rea_llms.models import set_seed
from rea_llms.observables import create_observable


def decode_completion_batch(tokenizer, completion_ids: list[list[int]]) -> list[str]:
    if hasattr(tokenizer, "batch_decode"):
        return tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in completion_ids]


def direct_ancestral_llm(
    model,
    tokenizer,
    *,
    prompt: str,
    n_samples: int,
    max_new_tokens: int,
    seed: int,
    ari_cap: float = 15.0,
    batch_size: int = 32,
) -> dict[str, Any]:
    if n_samples < 0:
        raise ValueError("n_samples must be >= 0")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    set_seed(seed)
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_token_count = int(prompt_ids.shape[1])
    observable = create_observable("ari", mode="capped", cap=ari_cap, backend="exact_cached")
    completions: list[str] = []
    raw_ari: list[float] = []
    capped_ari: list[float] = []
    token_ids: list[list[int]] = []
    start = time.perf_counter()

    for offset in range(0, n_samples, batch_size):
        current_batch = min(batch_size, n_samples - offset)
        input_ids = prompt_ids.repeat(current_batch, 1)
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_length=prompt_token_count + max_new_tokens,
                do_sample=True,
                top_k=None,
                temperature=1.0,
                eos_token_id=None,
                pad_token_id=getattr(tokenizer, "pad_token_id", None),
            )
        batch_completion_ids = generated[:, prompt_token_count:].detach().cpu().tolist()
        batch_completions = decode_completion_batch(tokenizer, batch_completion_ids)
        for ids, completion in zip(batch_completion_ids, batch_completions):
            score = observable.score_completion(prompt, completion)
            token_ids.append([int(x) for x in ids])
            completions.append(completion)
            raw_ari.append(float(score.metadata["raw_ari"]))
            capped_ari.append(float(score.metadata["capped_ari"]))

    return {
        "completion_texts": completions,
        "completions": completions,
        "completion_token_ids": token_ids,
        "token_ids": token_ids,
        "raw_ari": raw_ari,
        "capped_ari": capped_ari,
        "runtime_seconds": float(time.perf_counter() - start),
        "generated_token_transitions": int(n_samples * max_new_tokens),
        "prompt": prompt,
        "seed": int(seed),
    }
