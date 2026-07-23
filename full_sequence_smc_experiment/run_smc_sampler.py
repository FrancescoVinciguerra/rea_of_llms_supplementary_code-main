from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_sequence_smc_experiment.smc_sampler import (  # noqa: E402
    FullSequenceSMCConfig,
    parse_float_list,
    render_report,
    run_full_sequence_smc,
    write_json,
    write_jsonl,
)
from rea_llms.models import DEFAULT_MODEL_NAME, DEFAULT_PROMPT, DEFAULT_TOKENIZER_NAME, load_model_and_tokenizer  # noqa: E402
from rea_llms.experiment_log import (  # noqa: E402
    append_experiment_log,
    compute_ari_tail_summary,
    publish_selected_result,
    render_experiment_log_entry,
)
from rea_llms.observables import create_observable  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a full-sequence SMC sampler for p_lambda over complete LM completions, without resampling."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "full_sequence_smc_run")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--experiment-log", type=Path, default=None)
    parser.add_argument("--publish-result-dir", type=Path, default=None)

    parser.add_argument("--num-particles", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--lambdas", default="0,-0.25,-0.5,-0.75,-1.0")
    parser.add_argument("--mcmc-steps-per-level", type=int, default=1)
    parser.add_argument("--rare-event-threshold", type=float, default=15.0)
    parser.add_argument("--rare-event-direction", choices=["ge", "le"], default="ge")
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--ari-mode", choices=["raw", "capped"], default="capped")
    parser.add_argument("--ari-cap", type=float, default=15.0)
    parser.add_argument("--observable-backend", default="exact_cached")
    parser.add_argument("--disable-score-cache", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    lambdas = parse_float_list(args.lambdas)
    observable = create_observable(
        "ari",
        mode=args.ari_mode,
        cap=args.ari_cap,
        backend=args.observable_backend,
        cache_enabled=not args.disable_score_cache,
    )
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        args.tokenizer_name,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    config = FullSequenceSMCConfig(
        num_particles=args.num_particles,
        max_new_tokens=args.max_new_tokens,
        lambdas=lambdas,
        mcmc_steps_per_level=args.mcmc_steps_per_level,
        rare_event_threshold=args.rare_event_threshold,
        rare_event_direction=args.rare_event_direction,
        seed=args.seed,
    )
    result = run_full_sequence_smc(model, tokenizer, observable, prompt=args.prompt, config=config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or args.output_dir.name
    run_config = {
        **result.config,
        "run_name": run_name,
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "prompt": args.prompt,
        "device": args.device,
        "ari_mode": args.ari_mode,
        "ari_cap": args.ari_cap,
        "observable_backend": args.observable_backend,
    }
    enriched_summary = {
        **result.summary,
        "run_name": run_name,
        "algorithm": "full_sequence_smc",
        "ari_tail_summary": compute_ari_tail_summary(result.samples),
    }
    write_json(
        args.output_dir / "summary.json",
        {
            **enriched_summary,
            "diagnostics": result.diagnostics,
            "config": run_config,
        },
    )
    write_jsonl(args.output_dir / "samples.jsonl", result.samples)
    (args.output_dir / "report.md").write_text(render_report(enriched_summary, result.diagnostics), encoding="utf-8")
    if args.experiment_log is not None:
        append_experiment_log(
            args.experiment_log,
            render_experiment_log_entry(
                run_name=run_name,
                algorithm="full_sequence_smc",
                summary=enriched_summary,
                config=run_config,
            ),
        )
    if args.publish_result_dir is not None:
        publish_selected_result(
            output_dir=args.output_dir,
            publish_root=args.publish_result_dir,
            run_name=run_name,
            samples=result.samples,
        )
    print((args.output_dir / "summary.json").resolve())


if __name__ == "__main__":
    main()
