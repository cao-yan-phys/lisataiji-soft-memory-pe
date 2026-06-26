# Companion code: LISA-Taiji PE example

This repository accompanies the paper **"Probing soft signals of gravitational-wave memory with space-based interferometers"** ([arXiv:2603.28689](https://arxiv.org/abs/2603.28689)).

The example uses a soft displacement-memory signal with $H=10^{-19}$, $\lambda=\pi/4$, $\beta=\pi/3$, $\Psi=\pi/5$, $\tau_*=0.3$, $T=10^{-3}\,\mathrm{yr}$, and $f_{\max}=0.04\,\mathrm{Hz}$.

## Install

```bash
conda env create -f environment.yml
conda activate lisataiji_soft_memory_pe
```

## Run

```bash
python scripts/run_lisataiji_pe.py
```

This writes a deterministic diagnostic file for the configured paper example.

MLE and FIM diagnostics are available with:

```bash
python scripts/run_lisataiji_pe.py --with-mle --with-fim
```

```bash
python scripts/plot_lisataiji_pe.py
```

This redraws the posterior plot from the bundled thinned sample.

To rerun the MCMC chain:

```bash
python scripts/run_lisataiji_pe.py --run-mcmc
```

## Files

- `configs/lisataiji_pe.yaml`: parameters for the paper example.
- `data/lisataiji_samples_thinned.npz`: thinned posterior sample for quick plotting.
- `src/lisataiji_soft_pe/`: response, likelihood, MLE, FIM, and MCMC code.
- `scripts/`: scripts for running diagnostics, plotting the posterior, and rerunning the chain.
