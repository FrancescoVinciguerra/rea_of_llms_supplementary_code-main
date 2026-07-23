from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class MixtureMCMCConfig:
    lambdas: list[float]
    rhos: list[float]
    num_iterations: int
    burn_in: int
    thinning: int
    seed: int
    rare_event_threshold: float
    rare_event_direction: str = "ge"
    max_new_tokens: int = 100

    def __post_init__(self) -> None:
        if not self.lambdas:
            raise ValueError("lambdas must be non-empty")
        if len(self.lambdas) != len(self.rhos):
            raise ValueError("lambdas and rhos must have the same length")
        if any(not math.isfinite(x) for x in self.lambdas):
            raise ValueError("all lambdas must be finite")
        if any((not math.isfinite(x)) or x < 0.0 for x in self.rhos):
            raise ValueError("all rhos must be finite and non-negative")
        if sum(self.rhos) <= 0.0:
            raise ValueError("at least one rho must be positive")
        if self.num_iterations < 0:
            raise ValueError("num_iterations must be >= 0")
        if self.burn_in < 0:
            raise ValueError("burn_in must be >= 0")
        if self.thinning <= 0:
            raise ValueError("thinning must be > 0")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if self.rare_event_direction not in {"ge", "le"}:
            raise ValueError("rare_event_direction must be one of {'ge', 'le'}")


@dataclass
class MixtureMCMCResult:
    samples: list[Any]
    summary: dict[str, Any]
    diagnostics: dict[str, Any]
    runtime: dict[str, float]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": [sample.to_dict() for sample in self.samples],
            "summary": self.summary,
            "diagnostics": self.diagnostics,
            "runtime": self.runtime,
            "config": self.config,
        }


@dataclass
class LMState:
    token_ids: list[int]
    completion: str
    phi: float
    raw_ari: float | None


@dataclass(frozen=True)
class MixtureTiltParameters:
    lambdas: np.ndarray
    rhos: np.ndarray
    log_rhos: np.ndarray


def parse_float_list(value: str) -> list[float]:
    try:
        parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Could not parse comma-separated floats from {value!r}") from exc
    if not parsed:
        raise ValueError("Expected at least one comma-separated float")
    return parsed


def normalize_rhos(rhos: Iterable[float]) -> list[float]:
    arr = np.asarray(list(rhos), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("rhos must be non-empty")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("rhos must be finite and non-negative")
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError("at least one rho must be positive")
    return (arr / total).astype(float).tolist()


def build_tilt_parameters(lambdas: Iterable[float], rhos: Iterable[float]) -> MixtureTiltParameters:
    lambda_array = np.asarray(list(lambdas), dtype=np.float64)
    rho_array = np.asarray(list(rhos), dtype=np.float64)
    if lambda_array.size == 0:
        raise ValueError("lambdas must be non-empty")
    if lambda_array.shape != rho_array.shape:
        raise ValueError("lambdas and rhos must have the same length")
    if np.any(~np.isfinite(lambda_array)):
        raise ValueError("all lambdas must be finite")
    if np.any(~np.isfinite(rho_array)) or np.any(rho_array < 0.0):
        raise ValueError("all rhos must be finite and non-negative")
    if float(np.sum(rho_array)) <= 0.0:
        raise ValueError("at least one rho must be positive")

    normalized_rhos = rho_array / float(np.sum(rho_array))
    log_rhos = np.full_like(normalized_rhos, -np.inf, dtype=np.float64)
    positive = normalized_rhos > 0.0
    log_rhos[positive] = np.log(normalized_rhos[positive])
    return MixtureTiltParameters(lambdas=lambda_array, rhos=normalized_rhos, log_rhos=log_rhos)


def log_mixture_tilt_from_parameters(phi: float, tilt: MixtureTiltParameters) -> float:
    return float(logsumexp(tilt.log_rhos - tilt.lambdas * float(phi)))


def log_mixture_tilt(phi: float, lambdas: list[float], rhos: list[float]) -> float:
    # log sum_k rho_k exp(-lambda_k phi). The p_M(x | c) factor cancels from
    # all mixture acceptance ratios and from the SNIS weights used below.
    return log_mixture_tilt_from_parameters(phi, build_tilt_parameters(lambdas, rhos))


def log_component_correction_sum_from_parameters(phi: float, component_index: int, tilt: MixtureTiltParameters) -> float:
    lambda_k = float(tilt.lambdas[component_index])
    return float(logsumexp(tilt.log_rhos - (tilt.lambdas - lambda_k) * float(phi)))


def log_component_correction_sum(phi: float, component_index: int, lambdas: list[float], rhos: list[float]) -> float:
    # log S_k(x), where S_k(x) = Gamma_mix(x) / gamma_k(x). This correction
    # turns a component-wise MH move into a valid move for the full mixture.
    return log_component_correction_sum_from_parameters(phi, component_index, build_tilt_parameters(lambdas, rhos))


def acceptance_from_log_ratio(log_ratio: float) -> float:
    if log_ratio >= 0.0:
        return 1.0
    if log_ratio < -745.0:
        return 0.0
    return float(math.exp(log_ratio))


def component_log_acceptance_ratio(lambda_value: float, current_phi: float, proposal_phi: float) -> float:
    return -lambda_value * (proposal_phi - current_phi)


def rare_event_indicator(phi: float, threshold: float, direction: str) -> bool:
    if direction == "ge":
        return phi >= threshold
    if direction == "le":
        return phi <= threshold
    raise ValueError("direction must be one of {'ge', 'le'}")


def mixture_log_weights(phi_values: list[float], tilt: MixtureTiltParameters) -> np.ndarray:
    phi_array = np.asarray(phi_values, dtype=np.float64)
    log_mixture_values = logsumexp(
        tilt.log_rhos[None, :] - phi_array[:, None] * tilt.lambdas[None, :],
        axis=1,
    )
    return -np.asarray(log_mixture_values, dtype=np.float64)


def importance_summary(
    phi_values: list[float],
    *,
    lambdas: list[float],
    rhos: list[float],
    threshold: float,
    direction: str,
    tilt: MixtureTiltParameters | None = None,
) -> dict[str, Any]:
    n = len(phi_values)
    if n == 0:
        return {
            "saved_samples": 0,
            "importance_ess": 0.0,
            "importance_ess_over_n": 0.0,
            "mixture_rare_event_count": 0,
            "snis_rare_event_probability": float("nan"),
        }
    # SNIS back to p_M: p_M / Gamma_mix = 1 / sum_k rho_k exp(-lambda_k phi).
    tilt_parameters = tilt if tilt is not None else build_tilt_parameters(lambdas, rhos)
    log_weights = mixture_log_weights(phi_values, tilt_parameters)
    log_norm = float(logsumexp(log_weights))
    normalized = np.exp(log_weights - log_norm)
    ess = float(1.0 / np.sum(normalized * normalized))
    events = np.asarray([rare_event_indicator(phi, threshold, direction) for phi in phi_values], dtype=bool)
    return {
        "saved_samples": int(n),
        "importance_ess": ess,
        "importance_ess_over_n": float(ess / n),
        "mixture_rare_event_count": int(np.sum(events)),
        "snis_rare_event_probability": float(np.sum(normalized[events])),
    }


def generate_independent_tokens(model, prompt_ids: list[int], max_length: int) -> list[int]:
    import torch

    # Independent proposal x* ~ p_M(. | c), used by every mixture component.
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_length=max_length,
            do_sample=True,
            top_k=None,
            temperature=1.0,
            eos_token_id=None,
            pad_token_id=None,
        )
    return output[0].tolist()


def state_from_tokens(tokenizer, observable, *, prompt: str, prompt_token_count: int, token_ids: list[int]) -> LMState:
    completion = tokenizer.decode(token_ids[prompt_token_count:], skip_special_tokens=True)
    score = observable.score_completion(prompt, completion)
    return LMState(
        token_ids=token_ids,
        completion=completion,
        phi=float(score.value),
        raw_ari=float(score.metadata["raw_ari"]) if "raw_ari" in score.metadata else None,
    )


def run_mixture_mcmc(
    model,
    tokenizer,
    observable,
    *,
    prompt: str,
    config: MixtureMCMCConfig,
    initial_token_ids: list[int] | None = None,
) -> MixtureMCMCResult:
    from rea_llms.models import set_seed
    from rea_llms.results import SampleRecord

    rhos = normalize_rhos(config.rhos)
    cfg = MixtureMCMCConfig(
        lambdas=[float(x) for x in config.lambdas],
        rhos=rhos,
        num_iterations=config.num_iterations,
        burn_in=config.burn_in,
        thinning=config.thinning,
        seed=config.seed,
        rare_event_threshold=config.rare_event_threshold,
        rare_event_direction=config.rare_event_direction,
        max_new_tokens=config.max_new_tokens,
    )
    tilt = build_tilt_parameters(cfg.lambdas, rhos)
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    prompt_token_count = len(prompt_ids)
    max_length = prompt_token_count + cfg.max_new_tokens

    start = time.perf_counter()
    initial_ids = initial_token_ids if initial_token_ids is not None else generate_independent_tokens(model, prompt_ids, max_length)
    current = state_from_tokens(tokenizer, observable, prompt=prompt, prompt_token_count=prompt_token_count, token_ids=initial_ids)

    samples: list[SampleRecord] = []
    accepted = 0
    phi_trace = [float(current.phi)]

    for step in range(1, cfg.num_iterations + 1):
        # Choose which tilted component gamma_k proposes this MH move.
        component_index = int(rng.choice(len(tilt.lambdas), p=tilt.rhos))
        lambda_k = float(tilt.lambdas[component_index])

        proposal_ids = generate_independent_tokens(model, prompt_ids, max_length)
        proposal = state_from_tokens(tokenizer, observable, prompt=prompt, prompt_token_count=prompt_token_count, token_ids=proposal_ids)

        # Component acceptance for gamma_k. Because the proposal is p_M, the
        # base-model probabilities cancel and only the observable difference
        # remains.
        log_a_ratio = component_log_acceptance_ratio(lambda_k, current.phi, proposal.phi)
        component_alpha = acceptance_from_log_ratio(log_a_ratio)

        # Mixture correction B_k = min(1, S_k(x*) / S_k(x)).
        log_b_ratio = log_component_correction_sum_from_parameters(proposal.phi, component_index, tilt) - log_component_correction_sum_from_parameters(
            current.phi,
            component_index,
            tilt,
        )
        mixture_alpha = acceptance_from_log_ratio(log_b_ratio)
        acceptance_probability = component_alpha * mixture_alpha
        did_accept = bool(rng.random() < acceptance_probability)
        if did_accept:
            current = proposal
            accepted += 1

        phi_trace.append(float(current.phi))
        should_save = step > cfg.burn_in and ((step - cfg.burn_in - 1) % cfg.thinning == 0)
        if should_save:
            # Store unnormalised SNIS weight for estimating expectations under
            # the original base language model p_M.
            log_weight = -log_mixture_tilt_from_parameters(float(current.phi), tilt)
            samples.append(
                SampleRecord(
                    text=prompt + current.completion,
                    completion=current.completion,
                    phi=float(current.phi),
                    raw_ari=current.raw_ari,
                    log_weight=float(log_weight),
                    normalized_weight=None,
                    token_ids=current.token_ids[prompt_token_count:],
                    metadata={
                        "step": step,
                        "component_index": component_index,
                        "lambda_value": lambda_k,
                        "component_acceptance_probability": float(component_alpha),
                        "mixture_acceptance_probability": float(mixture_alpha),
                        "acceptance_probability": float(acceptance_probability),
                        "accepted": did_accept,
                    },
                )
            )

    elapsed = time.perf_counter() - start
    phi_values = [sample.phi for sample in samples]
    summary = importance_summary(
        phi_values,
        lambdas=cfg.lambdas,
        rhos=rhos,
        threshold=cfg.rare_event_threshold,
        direction=cfg.rare_event_direction,
        tilt=tilt,
    )
    if samples:
        log_weights = np.asarray([sample.log_weight for sample in samples], dtype=np.float64)
        normalized = np.exp(log_weights - float(logsumexp(log_weights)))
        for sample, weight in zip(samples, normalized):
            sample.normalized_weight = float(weight)

    runtime = {
        "total_seconds": float(elapsed),
        "seconds_per_saved_sample": float(elapsed / len(samples)) if samples else float("nan"),
    }
    diagnostics = {
        "acceptance_rate": float(accepted / cfg.num_iterations) if cfg.num_iterations else 0.0,
        "phi_trace": phi_trace,
    }
    summary.update(
        {
            "total_runtime_seconds": runtime["total_seconds"],
            "seconds_per_saved_sample": runtime["seconds_per_saved_sample"],
            "acceptance_rate": diagnostics["acceptance_rate"],
        }
    )
    return MixtureMCMCResult(
        samples=samples,
        summary=summary,
        diagnostics=diagnostics,
        runtime=runtime,
        config=asdict(cfg),
    )


def run_direct_baseline(
    model,
    tokenizer,
    *,
    prompt: str,
    n_samples: int,
    max_new_tokens: int,
    seed: int,
    threshold: float,
    direction: str,
    ari_mode: str,
    ari_cap: float,
    batch_size: int,
) -> dict[str, Any]:
    from rea_llms.smc import direct_ancestral_llm

    result = direct_ancestral_llm(
        model,
        tokenizer,
        prompt=prompt,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        seed=seed,
        ari_cap=ari_cap,
        batch_size=batch_size,
    )
    phi_values = result["raw_ari"] if ari_mode == "raw" else result["capped_ari"]
    events = [rare_event_indicator(float(phi), threshold, direction) for phi in phi_values]
    summary = {
        "method": "direct",
        "n_samples": int(n_samples),
        "rare_event_count": int(sum(events)),
        "rare_event_proportion": float(sum(events) / n_samples) if n_samples else float("nan"),
        "runtime_seconds": float(result["runtime_seconds"]),
    }
    return {"summary": summary, "raw": result}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_report(mixture_summary: dict[str, Any], direct_summary: dict[str, Any] | None = None) -> str:
    lines = [
        "# Mixture MCMC Rare-Event Experiment",
        "",
        "## Mixture sampler",
        f"- Total runtime seconds: {mixture_summary['total_runtime_seconds']:.6g}",
        f"- Seconds per saved sample: {mixture_summary['seconds_per_saved_sample']:.6g}",
        f"- Importance ESS/N: {mixture_summary['importance_ess_over_n']:.6g}",
        f"- Raw rare events in mixture samples: {mixture_summary['mixture_rare_event_count']}",
        f"- SNIS rare-event probability under p_M: {mixture_summary['snis_rare_event_probability']:.12g}",
        f"- Acceptance rate: {mixture_summary['acceptance_rate']:.6g}",
    ]
    if direct_summary is not None:
        lines.extend(
            [
                "",
                "## Direct baseline",
                f"- Runtime seconds: {direct_summary['runtime_seconds']:.6g}",
                f"- Rare events observed: {direct_summary['rare_event_count']}",
                f"- Rare-event proportion: {direct_summary['rare_event_proportion']:.12g}",
            ]
        )
    tail = mixture_summary.get("ari_tail_summary")
    if isinstance(tail, dict):
        lines.extend(
            [
                "",
                "## ARI Tail Summary",
                f"- Mean raw ARI: {tail.get('mean_ari')}",
                f"- Min raw ARI: {tail.get('min_ari')}",
                f"- Max raw ARI: {tail.get('max_ari')}",
                "",
                "| threshold | count | probability |",
                "| ---: | ---: | ---: |",
            ]
        )
        for key, count in tail.get("tail_counts", {}).items():
            lines.append(f"| {key.replace('ari_ge_', 'ARI >= ')} | {count} | {tail.get('tail_probabilities', {}).get(key)} |")
    return "\n".join(lines) + "\n"
