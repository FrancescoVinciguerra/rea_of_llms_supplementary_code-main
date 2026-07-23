from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mixture_mcmc_experiment.mixture_mcmc import (
    MixtureMCMCConfig,
    normalize_rhos,
    parse_float_list,
    render_report,
    run_direct_baseline,
    run_mixture_mcmc,
    write_csv,
    write_json,
    write_jsonl,
)
from rea_llms.experiment_log import (
    append_experiment_log,
    compute_ari_tail_summary,
    publish_selected_result,
    render_experiment_log_entry,
)

DEFAULT_PROMPT = "Once upon a time, in a big forest, there lived a rhinoc"
DEFAULT_MODEL_NAME = "roneneldan/TinyStories-8M"
DEFAULT_TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mixture-MCMC sampling from unnormalized biased LM components, with SNIS to p_M.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "mixture_mcmc_run")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--experiment-log", type=Path, default=None)
    parser.add_argument("--publish-result-dir", type=Path, default=None)

    parser.add_argument("--lambdas", default="-1.0,-0.7,-0.4,-0.2,0.0")
    parser.add_argument("--rhos", default=None, help="Comma-separated component weights. Defaults to uniform.")
    parser.add_argument("--num-iterations", type=int, default=500)
    parser.add_argument("--burn-in", type=int, default=100)
    parser.add_argument("--thinning", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-new-tokens", type=int, default=100)

    parser.add_argument("--rare-event-threshold", type=float, default=15.0)
    parser.add_argument("--rare-event-direction", choices=["ge", "le"], default="ge")
    parser.add_argument("--ari-mode", choices=["raw", "capped"], default="capped")
    parser.add_argument("--ari-cap", type=float, default=15.0)
    parser.add_argument("--observable-backend", default="exact_cached")
    parser.add_argument("--disable-score-cache", action="store_true")

    parser.add_argument("--run-direct-baseline", action="store_true")
    parser.add_argument("--direct-batch-size", type=int, default=32)
    parser.add_argument("--direct-seed-offset", type=int, default=1_000_000)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    from rea_llms.models import load_model_and_tokenizer
    from rea_llms.observables import create_observable

    lambdas = parse_float_list(args.lambdas)
    rhos = normalize_rhos(parse_float_list(args.rhos) if args.rhos is not None else [1.0] * len(lambdas))
    if len(lambdas) != len(rhos):
        raise ValueError("--lambdas and --rhos must contain the same number of values")

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
    config = MixtureMCMCConfig(
        lambdas=lambdas,
        rhos=rhos,
        num_iterations=args.num_iterations,
        burn_in=args.burn_in,
        thinning=args.thinning,
        seed=args.seed,
        rare_event_threshold=args.rare_event_threshold,
        rare_event_direction=args.rare_event_direction,
        max_new_tokens=args.max_new_tokens,
    )
    result = run_mixture_mcmc(model, tokenizer, observable, prompt=args.prompt, config=config)

    direct = None
    if args.run_direct_baseline:
        direct = run_direct_baseline(
            model,
            tokenizer,
            prompt=args.prompt,
            n_samples=len(result.samples),
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + args.direct_seed_offset,
            threshold=args.rare_event_threshold,
            direction=args.rare_event_direction,
            ari_mode=args.ari_mode,
            ari_cap=args.ari_cap,
            batch_size=args.direct_batch_size,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or args.output_dir.name
    run_config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "run_name": run_name,
        "lambdas": lambdas,
        "rhos": rhos,
        "output_dir": str(args.output_dir),
    }
    enriched_summary = {
        **result.summary,
        "run_name": run_name,
        "algorithm": "mixture_mcmc",
        "ari_tail_summary": compute_ari_tail_summary(result.samples),
    }
    write_json(args.output_dir / "config.json", run_config)
    write_json(args.output_dir / "mixture_result.json", result.to_dict())
    write_json(args.output_dir / "mixture_summary.json", enriched_summary)
    write_json(args.output_dir / "summary.json", {"summary": enriched_summary, "config": run_config, "diagnostics": result.diagnostics})
    write_csv(args.output_dir / "summary.csv", [enriched_summary])
    write_jsonl(args.output_dir / "mixture_samples.jsonl", [sample.to_dict() for sample in result.samples])
    pd.DataFrame([sample.to_dict() for sample in result.samples]).to_csv(args.output_dir / "mixture_samples.csv", index=False)

    direct_summary = None
    if direct is not None:
        direct_summary = direct["summary"]
        write_json(args.output_dir / "direct_baseline.json", direct)
        write_csv(args.output_dir / "direct_baseline_summary.csv", [direct_summary])

    report = render_report(enriched_summary, direct_summary)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    if args.experiment_log is not None:
        append_experiment_log(
            args.experiment_log,
            render_experiment_log_entry(
                run_name=run_name,
                algorithm="mixture_mcmc",
                summary=enriched_summary,
                config=run_config,
            ),
        )
    if args.publish_result_dir is not None:
        publish_selected_result(
            output_dir=args.output_dir,
            publish_root=args.publish_result_dir,
            run_name=run_name,
            samples=[sample.to_dict() for sample in result.samples],
        )
    print(report)


if __name__ == "__main__":
    main()
