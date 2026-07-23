from __future__ import annotations

import json

from rea_llms.experiment_log import (
    append_experiment_log,
    compute_ari_tail_summary,
    create_samples_preview,
    render_experiment_log_entry,
)


def test_compute_ari_tail_summary() -> None:
    samples = [{"raw_ari": 5.0}, {"raw_ari": 7.0}, {"phi": 12.0}]
    summary = compute_ari_tail_summary(samples, thresholds=(6, 10))
    assert summary["n_samples"] == 3
    assert summary["mean_ari"] == 8.0
    assert summary["min_ari"] == 5.0
    assert summary["max_ari"] == 12.0
    assert summary["tail_counts"]["ari_ge_6"] == 2
    assert summary["tail_probabilities"]["ari_ge_10"] == 1 / 3


def test_append_log_entry(tmp_path) -> None:
    entry = render_experiment_log_entry(
        run_name="unit_run",
        algorithm="mixture_mcmc",
        summary={"ari_tail_summary": compute_ari_tail_summary([{"raw_ari": 8.0}])},
        config={"model_name": "model", "tokenizer_name": "tok", "prompt": "prompt", "max_new_tokens": 2, "seed": 1},
    )
    path = tmp_path / "EXPERIMENT_LOG.md"
    append_experiment_log(path, entry)
    text = path.read_text(encoding="utf-8")
    assert "# Experiment Log" in text
    assert "## unit_run" in text
    assert "mixture_mcmc" in text


def test_create_samples_preview_from_jsonl(tmp_path) -> None:
    source = tmp_path / "samples.jsonl"
    source.write_text("\n".join(json.dumps({"raw_ari": i}) for i in range(3)) + "\n", encoding="utf-8")
    preview = create_samples_preview(samples_jsonl=source, output_path=tmp_path / "samples_preview.jsonl", max_rows=2)
    rows = preview.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2
    assert json.loads(rows[1])["raw_ari"] == 1
