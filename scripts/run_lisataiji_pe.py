from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lisataiji_soft_pe.core import (  # noqa: E402
    configure_context,
    diagnostics,
    event_from_config,
    json_clean,
    run_mcmc,
    save_mcmc_npz,
    write_history_csv,
    write_json,
)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LISA-Taiji soft-memory PE example.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "lisataiji_pe.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--run-mcmc", action="store_true", help="Run an emcee chain after the diagnostics.")
    parser.add_argument("--with-mle", action="store_true", help="Include matched-filter maximum-likelihood diagnostics.")
    parser.add_argument("--with-fim", action="store_true", help="Include FIM diagnostics at the injected parameters.")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None, help="Override mcmc.max_steps from the config.")
    parser.add_argument("--n-burn", type=int, default=None, help="Override mcmc.n_burn from the config.")
    parser.add_argument("--n-walkers", type=int, default=None, help="Override mcmc.n_walkers from the config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    event = event_from_config(config)
    ctx = configure_context(config)
    run = config.get("run", {})
    diag_cfg = config.get("diagnostics", {})
    inject_noise = bool(run.get("inject_noise", True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diag = diagnostics(
        ctx,
        event,
        inject_noise=inject_noise,
        maxiter=int(diag_cfg.get("mf_maxiter", 160)),
        rel_step=float(diag_cfg.get("fim_rel_step", 1.0e-5)),
        compute_mle=bool(args.with_mle),
        compute_fim=bool(args.with_fim),
    )
    diag_path = write_json(args.output_dir / "lisataiji_pe_diagnostics.json", diag)
    print(f"diagnostics: {diag_path}")
    print("SNR:", json_clean(diag["snr"]))
    if diag["matched_filter"] is not None:
        print("MLE:", json_clean(diag["matched_filter"]["x_mle"]))

    if not args.run_mcmc:
        return

    mcmc_cfg = dict(config.get("mcmc", {}))
    if args.max_steps is not None:
        mcmc_cfg["max_steps"] = args.max_steps
    if args.n_burn is not None:
        mcmc_cfg["n_burn"] = args.n_burn
    if args.n_walkers is not None:
        mcmc_cfg["n_walkers"] = args.n_walkers

    result = run_mcmc(
        ctx,
        event,
        inject_noise=inject_noise,
        start=str(mcmc_cfg.get("start", "mle")),
        n_walkers=int(mcmc_cfg.get("n_walkers", 48)),
        n_burn=int(mcmc_cfg.get("n_burn", 300)),
        max_steps=int(mcmc_cfg.get("max_steps", 12000)),
        chunk_size=int(mcmc_cfg.get("chunk_size", 500)),
        seed=int(mcmc_cfg.get("seed", 0)),
        scatter=float(mcmc_cfg.get("scatter", 1.0e-3)),
        moves=str(mcmc_cfg.get("moves", "robust")),
        mf_maxiter=int(diag_cfg.get("mf_maxiter", 160)),
        progress=args.progress,
    )
    npz_path = save_mcmc_npz(args.output_dir / "lisataiji_pe_chain.npz", result)
    hist_path = write_history_csv(args.output_dir / "lisataiji_pe_history.csv", result["history"])
    summary = {
        "output_npz": npz_path,
        "history_csv": hist_path,
        "truths": result["truths"],
        "labels": result["labels"],
        "snr": result["snr"],
        "tau": result["tau"],
        "mean_acceptance_fraction": result["acceptance_fraction"].mean(),
        "chain_shape": result["chain"].shape,
        "mcmc": {
            "n_burn": result["n_burn"],
            "steps_done": result["steps_done"],
            "n_walkers": result["n_walkers"],
            "seed": result["seed"],
            "moves": result["moves"],
            "elapsed_sec": result["elapsed_sec"],
            "last_history": result["history"][-1] if result["history"] else None,
        },
        "start": result["start"],
    }
    summary_path = write_json(args.output_dir / "lisataiji_pe_chain_summary.json", summary)
    print(f"chain: {npz_path}")
    print(f"history: {hist_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
