from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rea_llms.models import set_seed


@dataclass(frozen=True)
class FullSequenceSMCConfig:
    num_particles: int
    max_new_tokens: int
    lambdas: list[float]
    mcmc_steps_per_level: int
    rare_event_threshold: float
    rare_event_direction: str = "ge"
    seed: int = 12345

    def __post_init__(self) -> None:
        if self.num_particles <= 0:
            raise ValueError("num_particles must be > 0")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if not self.lambdas:
            raise ValueError("lambdas must be non-empty")
        if any(not math.isfinite(x) for x in self.lambdas):
            raise ValueError("all lambdas must be finite")
        if self.mcmc_steps_per_level < 0:
            raise ValueError("mcmc_steps_per_level must be >= 0")
        if self.rare_event_direction not in {"ge", "le"}:
            raise ValueError("rare_event_direction must be one of {'ge', 'le'}")
        if not math.isfinite(self.rare_event_threshold):
            raise ValueError("rare_event_threshold must be finite")


@dataclass
class FullSequenceState:
    full_token_ids: list[int]
    completion_token_ids: list[int]
    completion: str
    text: str
    phi: float
    metadata: dict[str, Any]


@dataclass
class FullSequenceSMCResult:
    samples: list[dict[str, Any]]
    summary: dict[str, Any]
    diagnostics: dict[str, Any]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "summary": self.summary,
            "diagnostics": self.diagnostics,
            "config": self.config,
        }


def parse_float_list(value: str) -> list[float]:
    try:
        parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Could not parse comma-separated floats from {value!r}") from exc
    if not parsed:
        raise ValueError("Expected at least one comma-separated float")
    return parsed


def logsumexp(values: np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("-inf")
    max_value = float(np.max(finite))
    return max_value + float(np.log(np.sum(np.exp(finite - max_value))))


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    return np.asarray(log_weights, dtype=np.float64) - logsumexp(log_weights)


def ess_over_n_from_log_weights(log_weights: np.ndarray) -> float:
    normalized_log = normalize_log_weights(log_weights)
    ess = math.exp(-logsumexp(2.0 * normalized_log))
    return float(ess / len(log_weights)) if len(log_weights) else float("nan")


def acceptance_from_log_ratio(log_ratio: float) -> float:
    if log_ratio >= 0.0:
        return 1.0
    if log_ratio < -745.0:
        return 0.0
    return float(math.exp(log_ratio))


def rare_event_indicator(phi: float, threshold: float, direction: str) -> bool:
    if direction == "ge":
        return phi >= threshold
    if direction == "le":
        return phi <= threshold
    raise ValueError("direction must be one of {'ge', 'le'}")


def _prompt_ids(tokenizer, prompt: str) -> list[int]:
    encoded = tokenizer(prompt, return_tensors="pt")["input_ids"]
    if hasattr(encoded, "detach"):
        return encoded[0].detach().cpu().tolist()
    return list(encoded[0])


def _decode_completions(tokenizer, completion_ids: list[list[int]]) -> list[str]:
    if hasattr(tokenizer, "batch_decode"):
        return tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in completion_ids]


def generate_base_sequences(
    model,
    prompt_ids: list[int],
    *,
    num_sequences: int,
    max_new_tokens: int,
    generator: torch.Generator,
) -> list[list[int]]:
    # Draw complete continuations from the base language model p_M. These draws
    # are used both to initialise the particle population and as independent
    # Metropolis proposals inside the full-sequence SMC mutation kernel.
    device = next(model.parameters()).device
    batch_ids = [list(prompt_ids) for _ in range(num_sequences)]
    for _ in range(max_new_tokens):
        input_tensor = torch.tensor(batch_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=input_tensor).logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, 1, generator=generator).squeeze(1).detach().cpu().tolist()
        for ids, token_id in zip(batch_ids, next_tokens):
            ids.append(int(token_id))
    return batch_ids


def states_from_token_sequences(
    tokenizer,
    observable,
    *,
    prompt: str,
    prompt_token_count: int,
    token_sequences: list[list[int]],
) -> list[FullSequenceState]:
    # A particle state is a full completion plus its observable value phi(x).
    # The SMC algorithm below works on full completions, not on token prefixes.
    completion_ids = [ids[prompt_token_count:] for ids in token_sequences]
    completions = _decode_completions(tokenizer, completion_ids)
    score_batch = getattr(observable, "score_token_sequences", None)
    if callable(score_batch):
        scores = score_batch(
            tokenizer=tokenizer,
            prompt=prompt,
            token_sequences=token_sequences,
            prompt_token_count=prompt_token_count,
        )
    else:
        scores = [observable.score_completion(prompt, completion) for completion in completions]

    states: list[FullSequenceState] = []
    for full_ids, comp_ids, completion, score in zip(token_sequences, completion_ids, completions, scores):
        phi = float(score.value)
        if not math.isfinite(phi):
            raise ValueError("Observable returned a non-finite phi value")
        states.append(
            FullSequenceState(
                full_token_ids=list(full_ids),
                completion_token_ids=list(comp_ids),
                completion=completion,
                text=prompt + completion,
                phi=phi,
                metadata=dict(score.metadata),
            )
        )
    return states


def run_full_sequence_smc(
    model,
    tokenizer,
    observable,
    *,
    prompt: str,
    config: FullSequenceSMCConfig,
) -> FullSequenceSMCResult:
    set_seed(config.seed)
    device = next(model.parameters()).device
    torch_generator = torch.Generator(device=device).manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    prompt_ids = _prompt_ids(tokenizer, prompt)
    prompt_token_count = len(prompt_ids)
    start = time.perf_counter()

    # Level 0 is sampled from p_M. Therefore the usual schedule should start at
    # lambda=0; otherwise an initial importance correction would be needed.
    initial_tokens = generate_base_sequences(
        model,
        prompt_ids,
        num_sequences=config.num_particles,
        max_new_tokens=config.max_new_tokens,
        generator=torch_generator,
    )
    particles = states_from_token_sequences(
        tokenizer,
        observable,
        prompt=prompt,
        prompt_token_count=prompt_token_count,
        token_sequences=initial_tokens,
    )
    log_weights = np.zeros(config.num_particles, dtype=np.float64)
    ess_trajectory = [1.0]
    level_diagnostics: list[dict[str, Any]] = [
        {
            "level": 0,
            "lambda": float(config.lambdas[0]),
            "ess_over_n": 1.0,
            "accepted": 0,
            "proposals": 0,
            "acceptance_rate": None,
            "mean_phi": float(np.mean([p.phi for p in particles])),
        }
    ]

    total_accepted = 0
    total_proposals = 0
# Per ogni lambda eseuog (lambda_0 ... lambda_k):
    for level in range(1, len(config.lambdas)):
        previous_lambda = float(config.lambdas[level - 1])
        current_lambda = float(config.lambdas[level])
        level_accepted = 0
        level_proposals = 0

        
        # Mutation step: apply a short independent-proposal MH chain targeting
        # p_{k-1}(x) proportional to p_M(x | c) exp(-lambda_{k-1} phi(x)).
        # The p_M proposal terms cancel, leaving only the phi difference.
        for _ in range(config.mcmc_steps_per_level):

    #genero completions
            proposal_tokens = generate_base_sequences(
                model,
                prompt_ids,
                num_sequences=config.num_particles,
                max_new_tokens=config.max_new_tokens,
                generator=torch_generator,
            )
    # states_.. prende i token di completion e li separa dal blocco c, ottiene la frase completa e calcola l 'ari
            proposals = states_from_token_sequences(
                tokenizer,
                observable,
                prompt=prompt,
                prompt_token_count=prompt_token_count,
                token_sequences=proposal_tokens,
            )
    # per ogni particella ( proposal = x_k), calcolo la sua acceptance prob, e eventualmente setta la x_k come proposal
            for idx, proposal in enumerate(proposals):
                current = particles[idx]
                # log A = -lambda_{k-1} [phi(x*) - phi(x)].
                log_acceptance_ratio = -previous_lambda * (proposal.phi - current.phi)
                acceptance_probability = acceptance_from_log_ratio(log_acceptance_ratio)
                did_accept = bool(rng.random() < acceptance_probability)
                level_proposals += 1
                total_proposals += 1
                if did_accept:
                    particles[idx] = proposal
                    level_accepted += 1
                    total_accepted += 1

        phi_values = np.asarray([particle.phi for particle in particles], dtype=np.float64)
        # Reweight from gamma_{k-1} to gamma_k:
        # log w_k = log w_{k-1} - (lambda_k - lambda_{k-1}) phi(x).
        log_weights = log_weights - (current_lambda - previous_lambda) * phi_values
        ess_over_n = ess_over_n_from_log_weights(log_weights)
        ess_trajectory.append(ess_over_n)
        level_diagnostics.append(
            {
                "level": level,
                "lambda": current_lambda,
                "previous_lambda": previous_lambda,
                "delta_lambda": current_lambda - previous_lambda,
                "ess_over_n": ess_over_n,
                "accepted": int(level_accepted),
                "proposals": int(level_proposals),
                "acceptance_rate": float(level_accepted / level_proposals) if level_proposals else None,
                "mean_phi": float(np.mean(phi_values)),
            }
        )

    elapsed = time.perf_counter() - start
    # Normalised weights define the final empirical approximation of p_K.
    normalized_log_weights = normalize_log_weights(log_weights)
    normalized_weights = np.exp(normalized_log_weights)
    final_phi = np.asarray([particle.phi for particle in particles], dtype=np.float64)
    rare_events = np.asarray(
        [
            rare_event_indicator(float(phi), config.rare_event_threshold, config.rare_event_direction)
            for phi in final_phi
        ],
        dtype=bool,
    )

    samples: list[dict[str, Any]] = []
    for idx, (particle, log_weight, normalized_weight, is_event) in enumerate(
        zip(particles, log_weights, normalized_weights, rare_events)
    ):
        samples.append(
            {
                "particle_id": idx,
                "text": particle.text,
                "completion": particle.completion,
                "phi": float(particle.phi),
                "raw_ari": particle.metadata.get("raw_ari"),
                "capped_ari": particle.metadata.get("capped_ari"),
                "log_weight": float(log_weight),
                "normalized_weight": float(normalized_weight),
                "rare_event": bool(is_event),
                "completion_token_ids": particle.completion_token_ids,
                "full_token_ids": particle.full_token_ids,
                "metadata": particle.metadata,
            }
        )

    final_ess_over_n = ess_trajectory[-1]
    min_ess_over_n = float(np.min(np.asarray(ess_trajectory, dtype=np.float64)))
    raw_rare_event_count = int(np.sum(rare_events))
    weighted_rare_event_probability = float(np.sum(normalized_weights[rare_events]))
    acceptance_rate = float(total_accepted / total_proposals) if total_proposals else 0.0
    summary = {
        "method": "full_sequence_smc_no_resampling",
        "num_particles": int(config.num_particles),
        "max_new_tokens": int(config.max_new_tokens),
        "lambdas": [float(x) for x in config.lambdas],
        "target_lambda": float(config.lambdas[-1]),
        "mcmc_steps_per_level": int(config.mcmc_steps_per_level),
        "rare_event_threshold": float(config.rare_event_threshold),
        "rare_event_direction": config.rare_event_direction,
        "runtime_total_seconds": float(elapsed),
        "runtime_seconds_per_final_particle": float(elapsed / config.num_particles),
        "final_ess_over_n": float(final_ess_over_n),
        "min_ess_over_n": min_ess_over_n,
        "raw_rare_event_count_final_particles": raw_rare_event_count,
        "weighted_rare_event_probability_p_K": weighted_rare_event_probability,
        "mean_mcmc_acceptance_rate": acceptance_rate,
    }
    diagnostics = {
        "ess_over_n_trajectory": [float(x) for x in ess_trajectory],
        "level_diagnostics": level_diagnostics,
        "total_mcmc_accepted": int(total_accepted),
        "total_mcmc_proposals": int(total_proposals),
        "final_log_weight_min": float(np.min(log_weights)),
        "final_log_weight_max": float(np.max(log_weights)),
        "final_normalized_weight_max": float(np.max(normalized_weights)),
        "final_phi_mean": float(np.mean(final_phi)),
        "final_phi_min": float(np.min(final_phi)),
        "final_phi_max": float(np.max(final_phi)),
    }
    return FullSequenceSMCResult(
        samples=samples,
        summary=summary,
        diagnostics=diagnostics,
        config=asdict(config),
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_report(summary: dict[str, Any], diagnostics: dict[str, Any]) -> str:
    lines = [
        "# Full-Sequence SMC Sampler",
        "",
        "No resampling. No TPS. No MBAR. Particle state is the complete completion.",
        "",
        "## Summary",
        f"- Runtime total seconds: {summary['runtime_total_seconds']:.6g}",
        f"- Seconds per final particle: {summary['runtime_seconds_per_final_particle']:.6g}",
        f"- Final ESS/N: {summary['final_ess_over_n']:.6g}",
        f"- Minimum ESS/N: {summary['min_ess_over_n']:.6g}",
        f"- Raw rare events among final particles: {summary['raw_rare_event_count_final_particles']}",
        f"- Weighted rare-event probability under p_K: {summary['weighted_rare_event_probability_p_K']:.12g}",
        f"- Mean MCMC acceptance rate: {summary['mean_mcmc_acceptance_rate']:.6g}",
        "",
        "## Schedule",
        "",
        "| level | lambda | ESS/N | accepted | proposals | acceptance | mean phi |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in diagnostics["level_diagnostics"]:
        acceptance = row["acceptance_rate"]
        lines.append(
            "| {level} | {lam:.6g} | {ess:.6g} | {accepted} | {proposals} | {acceptance} | {mean_phi:.6g} |".format(
                level=row["level"],
                lam=row["lambda"],
                ess=row["ess_over_n"],
                accepted=row["accepted"],
                proposals=row["proposals"],
                acceptance="" if acceptance is None else f"{acceptance:.6g}",
                mean_phi=row["mean_phi"],
            )
        )
    tail = summary.get("ari_tail_summary")
    if isinstance(tail, dict):
        lines.extend(
            [
                "",
                "## ARI Tail Summary",
                "",
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
    if all(float(x) == 0.0 for x in summary["lambdas"]):
        lines.extend(
            [
                "",
                "## Lambda-Zero Sanity Check",
                "",
                "All lambda values are zero, so the incremental weights are constant and ESS/N should remain 1.",
            ]
        )
    return "\n".join(lines) + "\n"
