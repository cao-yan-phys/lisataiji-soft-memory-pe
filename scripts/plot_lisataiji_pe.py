from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lisataiji_soft_pe.plotting import corner_plot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the LISA-Taiji posterior samples.")
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data" / "lisataiji_samples_thinned.npz",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "lisataiji_pe_corner.pdf")
    parser.add_argument("--discard", type=int, default=None)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--title", type=str, default="LISA-Taiji")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = corner_plot(args.samples, args.output, discard=args.discard, thin=args.thin, title=args.title)
    print(f"saved {summary['output']}")
    print(f"samples {summary['n_samples']}")


if __name__ == "__main__":
    main()
