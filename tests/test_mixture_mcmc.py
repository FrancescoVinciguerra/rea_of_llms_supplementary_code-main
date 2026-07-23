from __future__ import annotations

import math

import numpy as np

from mixture_mcmc_experiment.mixture_mcmc import (
    acceptance_from_log_ratio,
    component_log_acceptance_ratio,
    importance_summary,
    log_component_correction_sum,
    log_mixture_tilt,
    normalize_rhos,
    rare_event_indicator,
)
from mixture_mcmc_experiment.run_experiment import parse_args


def test_normalize_rhos() -> None:
    assert normalize_rhos([2.0, 1.0, 1.0]) == [0.5, 0.25, 0.25]


def test_log_mixture_tilt_matches_direct_sum() -> None:
    lambdas = [-1.0, 0.0, 0.5]
    rhos = normalize_rhos([1.0, 2.0, 1.0])
    phi = 3.2
    direct = sum(rho * math.exp(-lam * phi) for lam, rho in zip(lambdas, rhos))
    assert np.isclose(math.exp(log_mixture_tilt(phi, lambdas, rhos)), direct)


def test_component_correction_sum_matches_definition() -> None:
    lambdas = [-1.0, -0.5, 0.0]
    rhos = normalize_rhos([1.0, 1.0, 2.0])
    phi = 7.0
    k = 1
    direct = sum(rho * math.exp(-(lam - lambdas[k]) * phi) for lam, rho in zip(lambdas, rhos))
    assert np.isclose(math.exp(log_component_correction_sum(phi, k, lambdas, rhos)), direct)


def test_importance_summary_snis_event_probability() -> None:
    lambdas = [0.0]
    rhos = [1.0]
    summary = importance_summary([0.0, 1.0, 2.0, 3.0], lambdas=lambdas, rhos=rhos, threshold=2.0, direction="ge")
    assert summary["importance_ess_over_n"] == 1.0
    assert summary["mixture_rare_event_count"] == 2
    assert summary["snis_rare_event_probability"] == 0.5


def test_rare_event_direction_and_acceptance() -> None:
    assert rare_event_indicator(1.0, 2.0, "le")
    assert not rare_event_indicator(1.0, 2.0, "ge")
    assert acceptance_from_log_ratio(10.0) == 1.0
    assert np.isclose(acceptance_from_log_ratio(math.log(0.25)), 0.25)


def test_independence_component_acceptance_ratio() -> None:
    assert np.isclose(component_log_acceptance_ratio(-0.5, current_phi=2.0, proposal_phi=5.0), 1.5)
    assert np.isclose(component_log_acceptance_ratio(0.25, current_phi=2.0, proposal_phi=5.0), -0.75)


def test_mixture_does_not_run_direct_baseline_by_default() -> None:
    args = parse_args([])
    assert args.run_direct_baseline is False


def test_cli_accepts_run_name_and_experiment_log() -> None:
    args = parse_args(["--run-name", "mix_unit", "--experiment-log", "results/EXPERIMENT_LOG.md"])
    assert args.run_name == "mix_unit"
    assert str(args.experiment_log).endswith("EXPERIMENT_LOG.md")
