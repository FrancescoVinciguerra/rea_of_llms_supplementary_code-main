from __future__ import annotations

import random

import numpy as np
import torch
import transformers


DEFAULT_PROMPT = "Once upon a time, in a big forest, there lived a rhinoc"
DEFAULT_MODEL_NAME = "roneneldan/TinyStories-8M"
DEFAULT_TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"


def resolve_device(device: str = "auto") -> torch.device:
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be one of {'auto', 'cuda', 'cpu'}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    tokenizer_name: str | None = None,
    *,
    local_files_only: bool = False,
    device: str = "auto",
):
    selected_device = resolve_device(device)
    tokenizer_name = tokenizer_name or model_name
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name,
        local_files_only=local_files_only,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model.to(selected_device)
    model.eval()
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
