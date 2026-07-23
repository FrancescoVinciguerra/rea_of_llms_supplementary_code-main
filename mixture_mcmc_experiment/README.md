# Mixture MCMC rare-event experiment

This folder contains a small, reproducible experiment for sampling from

```text
Gamma_mix(x) = sum_k rho_k p_M(x | c) exp(-lambda_k phi(x))
```

without estimating the component normalizers. The component-wise proposal is an independence Metropolis-Hastings move:

```text
x* ~ p_M(. | c)
a_k = min(1, exp(-lambda_k (phi(x*) - phi(x))))
```

It then applies the mixture correction

```text
b_k = min(1, S_k(x*) / S_k(x)),   S_k(x) = sum_j rho_j exp(-(lambda_j - lambda_k) phi(x)).
```

The final acceptance probability is

```text
A = a_k b_k.
```

Run a smoke experiment:

```powershell
python mixture_mcmc_experiment/run_experiment.py --num-iterations 20 --burn-in 5 --thinning 3 --max-new-tokens 20 --lambdas=-1.0,-0.5,0.0 --rare-event-threshold 15 --rare-event-direction ge
```

Default outputs are written under `mixture_mcmc_experiment/results/mixture_mcmc_run`:

- `mixture_samples.jsonl` and `mixture_samples.csv`
- `mixture_summary.json` and `summary.csv`
- `direct_baseline.json`, only when `--run-direct-baseline` is set
- `report.md`

The report intentionally includes only runtime, time per saved sample, importance ESS/N, raw rare-event count in mixture samples, the self-normalized rare-event estimate under `p_M`, acceptance rate, and the direct baseline when explicitly enabled.

Selected-result publication:

```bash
python mixture_mcmc_experiment/run_experiment.py \
  --run-name mixture_smoke \
  --experiment-log results/EXPERIMENT_LOG.md \
  --publish-result-dir results/selected_runs
```
