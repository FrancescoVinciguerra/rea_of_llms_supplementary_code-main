# Full-Sequence SMC Experiment

This folder implements a complete-sequence SMC-style sampler for tilted completion distributions.

The particle state is the full generated completion. The sampler:

- initializes particles from ancestral model samples;
- mutates complete sequences with independence proposals from the base model;
- updates importance weights across a lambda schedule;
- reports ESS/N, acceptance rate, ARI tails, and selected sample previews.

Smoke run:

```bash
python full_sequence_smc_experiment/run_smc_sampler.py \
  --device auto \
  --num-particles 2 \
  --max-new-tokens 2 \
  --lambdas 0,0 \
  --mcmc-steps-per-level 1 \
  --output-dir outputs/full_sequence_smoke
```

Selected-result publication:

```bash
python full_sequence_smc_experiment/run_smc_sampler.py \
  --run-name smc_smoke \
  --experiment-log results/EXPERIMENT_LOG.md \
  --publish-result-dir results/selected_runs
```
