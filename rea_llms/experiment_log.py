from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TAIL_THRESHOLDS = (6, 7, 8, 10, 12, 15)


def _sample_to_dict(sample: Any) -> dict[str, Any]:
    if isinstance(sample, dict):
        return sample
    to_dict = getattr(sample, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    data = getattr(sample, "__dict__", None)
    if isinstance(data, dict):
        return data
    raise TypeError(f"Cannot convert sample of type {type(sample)!r} to dict")


def _sample_ari(sample: Any) -> float | None:
    row = _sample_to_dict(sample)
    for key in ("raw_ari", "ari", "phi"):
        value = row.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("raw_ari") is not None:
        parsed = float(metadata["raw_ari"])
        if math.isfinite(parsed):
            return parsed
    return None


def compute_ari_tail_summary(
    samples: Iterable[Any],
    thresholds: Iterable[float] = DEFAULT_TAIL_THRESHOLDS,
) -> dict[str, Any]:
    values = [value for sample in samples if (value := _sample_ari(sample)) is not None]
    n = len(values)
    summary: dict[str, Any] = {
        "n_samples": n,
        "mean_ari": float(sum(values) / n) if n else None,
        "min_ari": float(min(values)) if n else None,
        "max_ari": float(max(values)) if n else None,
        "tail_counts": {},
        "tail_probabilities": {},
    }
    for threshold in thresholds:
        label = f"ari_ge_{_format_threshold(threshold)}"
        count = sum(1 for value in values if value >= float(threshold))
        summary["tail_counts"][label] = int(count)
        summary["tail_probabilities"][label] = float(count / n) if n else None
    return summary


def render_experiment_log_entry(
    *,
    run_name: str,
    algorithm: str,
    summary: dict[str, Any],
    config: dict[str, Any],
    notes: str | None = None,
) -> str:
    tail = summary.get("ari_tail_summary", {})
    counts = tail.get("tail_counts", {}) if isinstance(tail, dict) else {}
    probs = tail.get("tail_probabilities", {}) if isinstance(tail, dict) else {}
    lines = [
        f"## {run_name}",
        "",
        f"- Timestamp UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        f"- Algorithm: `{algorithm}`",
        f"- Model: `{config.get('model_name', 'unknown')}`",
        f"- Tokenizer: `{config.get('tokenizer_name', 'unknown')}`",
        f"- Prompt: `{config.get('prompt', '')}`",
        f"- Max new tokens: `{config.get('max_new_tokens', summary.get('max_new_tokens', ''))}`",
        f"- Seed: `{config.get('seed', summary.get('seed', ''))}`",
        f"- Lambdas: `{config.get('lambdas', summary.get('lambdas', ''))}`",
    ]
    if "rhos" in config:
        lines.append(f"- Rhos: `{config.get('rhos')}`")
    if "num_particles" in summary:
        lines.append(f"- Particles: `{summary.get('num_particles')}`")
    if "num_iterations" in config:
        lines.append(f"- Iterations: `{config.get('num_iterations')}`")
        lines.append(f"- Burn-in/thinning: `{config.get('burn_in')}` / `{config.get('thinning')}`")
    if "mcmc_steps_per_level" in summary:
        lines.append(f"- MCMC steps per level: `{summary.get('mcmc_steps_per_level')}`")
    lines.extend(
        [
            f"- Runtime seconds: `{summary.get('runtime_total_seconds', summary.get('total_runtime_seconds', ''))}`",
            f"- ESS/N: `{summary.get('final_ess_over_n', summary.get('importance_ess_over_n', ''))}`",
            f"- Acceptance rate: `{summary.get('mean_mcmc_acceptance_rate', summary.get('acceptance_rate', ''))}`",
            f"- Mean/min/max raw ARI: `{tail.get('mean_ari')}` / `{tail.get('min_ari')}` / `{tail.get('max_ari')}`",
            "",
            "| Threshold | Count | Probability |",
            "| ---: | ---: | ---: |",
        ]
    )
    for key in counts:
        threshold = key.replace("ari_ge_", "ARI >= ").replace("p", ".")
        lines.append(f"| {threshold} | {counts[key]} | {probs.get(key)} |")
    if notes:
        lines.extend(["", f"Notes: {notes}"])
    return "\n".join(lines) + "\n"


def append_experiment_log(path: Path | str, entry: str) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("# Experiment Log\n\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as handle:
        if log_path.stat().st_size > 0:
            handle.write("\n")
        handle.write(entry.rstrip() + "\n")


def create_samples_preview(
    *,
    samples: Iterable[Any] | None = None,
    samples_jsonl: Path | str | None = None,
    output_path: Path | str,
    max_rows: int = 20,
) -> Path:
    if samples is None and samples_jsonl is None:
        raise ValueError("Provide samples or samples_jsonl")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if samples is not None:
        rows = [_sample_to_dict(sample) for sample in list(samples)[:max_rows]]
    else:
        with Path(samples_jsonl).open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(rows) >= max_rows:
                    break
                rows.append(json.loads(line))
    if out.suffix.lower() == ".csv":
        fields = sorted({key for row in rows for key in row})
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_cell(row.get(key)) for key in fields})
    else:
        with out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def publish_selected_result(
    *,
    output_dir: Path | str,
    publish_root: Path | str,
    run_name: str,
    samples: Iterable[Any] | None = None,
    preview_rows: int = 20,
) -> Path:
    out = Path(output_dir)
    destination = Path(publish_root) / run_name
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "report.md"):
        src = out / name
        if src.exists():
            shutil.copy2(src, destination / name)
    if samples is not None:
        create_samples_preview(samples=samples, output_path=destination / "samples_preview.jsonl", max_rows=preview_rows)
    elif (out / "samples.jsonl").exists():
        create_samples_preview(samples_jsonl=out / "samples.jsonl", output_path=destination / "samples_preview.jsonl", max_rows=preview_rows)
    elif (out / "mixture_samples.jsonl").exists():
        create_samples_preview(
            samples_jsonl=out / "mixture_samples.jsonl",
            output_path=destination / "samples_preview.jsonl",
            max_rows=preview_rows,
        )
    return destination


def _format_threshold(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace(".", "p").replace("-", "minus_")


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value
