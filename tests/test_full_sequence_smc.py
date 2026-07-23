from __future__ import annotations

import math

import numpy as np
import torch

from full_sequence_smc_experiment.run_smc_sampler import parse_args
from full_sequence_smc_experiment.smc_sampler import (
    FullSequenceSMCConfig,
    acceptance_from_log_ratio,
    ess_over_n_from_log_weights,
    rare_event_indicator,
    run_full_sequence_smc,
)
from rea_llms.observables.base import ObservableScore


class TinyTokenizer:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[1]], dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens=True):
        pieces = {0: " a", 1: "Once", 2: " longer", 3: "."}
        return "".join(pieces.get(int(token), " x") for token in token_ids)

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in token_ids]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids):
        batch, length = input_ids.shape
        logits = torch.zeros((batch, length, 4), dtype=torch.float32, device=input_ids.device)
        logits[..., 0] = 1.0
        logits[..., 2] = 0.5
        logits[..., 3] = 0.25
        return type("Output", (), {"logits": logits})


class CompletionTokenSumObservable:
    name = "completion_token_sum"

    def score_token_sequences(self, *, tokenizer, prompt, token_sequences, prompt_token_count):
        scores = []
        for ids in token_sequences:
            value = float(sum(ids[prompt_token_count:]))
            scores.append(ObservableScore(value=value, metadata={"raw_ari": value, "capped_ari": value}))
        return scores


def test_lambda_zero_keeps_constant_weights_and_ess_one() -> None:
    config = FullSequenceSMCConfig(
        num_particles=8,
        max_new_tokens=4,
        lambdas=[0.0, 0.0, 0.0],
        mcmc_steps_per_level=2,
        rare_event_threshold=4.0,
        rare_event_direction="ge",
        seed=7,
    )
    result = run_full_sequence_smc(TinyModel(), TinyTokenizer(), CompletionTokenSumObservable(), prompt="Once", config=config)
    assert np.isclose(result.summary["final_ess_over_n"], 1.0)
    assert np.isclose(result.summary["min_ess_over_n"], 1.0)
    assert all(sample["log_weight"] == 0.0 for sample in result.samples)
    assert all(np.isclose(sample["normalized_weight"], 1.0 / config.num_particles) for sample in result.samples)
    assert result.summary["mean_mcmc_acceptance_rate"] == 1.0


def test_nonzero_schedule_updates_weights_with_final_mutated_phi() -> None:
    config = FullSequenceSMCConfig(
        num_particles=6,
        max_new_tokens=3,
        lambdas=[0.0, 0.5],
        mcmc_steps_per_level=1,
        rare_event_threshold=5.0,
        rare_event_direction="ge",
        seed=11,
    )
    result = run_full_sequence_smc(TinyModel(), TinyTokenizer(), CompletionTokenSumObservable(), prompt="Once", config=config)
    for sample in result.samples:
        assert np.isclose(sample["log_weight"], -0.5 * sample["phi"])
    assert np.isclose(sum(sample["normalized_weight"] for sample in result.samples), 1.0)
    assert 0.0 < result.summary["final_ess_over_n"] <= 1.0


def test_helpers() -> None:
    log_weights = np.asarray([0.0, 0.0, 0.0])
    assert np.isclose(ess_over_n_from_log_weights(log_weights), 1.0)
    assert rare_event_indicator(1.0, 2.0, "le")
    assert not rare_event_indicator(1.0, 2.0, "ge")
    assert acceptance_from_log_ratio(10.0) == 1.0
    assert np.isclose(acceptance_from_log_ratio(math.log(0.25)), 0.25)


def test_cli_accepts_run_name_and_experiment_log() -> None:
    args = parse_args(["--run-name", "smc_unit", "--experiment-log", "results/EXPERIMENT_LOG.md"])
    assert args.run_name == "smc_unit"
    assert str(args.experiment_log).endswith("EXPERIMENT_LOG.md")
