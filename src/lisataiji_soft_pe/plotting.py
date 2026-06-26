from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .core import LABELS, PLOT_LABELS, flatten_chain, json_clean, parameter_stats


def load_samples(path: str | Path, *, discard: int | None = None, thin: int = 1) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "samples" in data.files:
            samples = np.asarray(data["samples"], dtype=float)
        elif "chain" in data.files:
            chain = np.asarray(data["chain"], dtype=float)
            if discard is None:
                discard = chain.shape[0] // 2
            samples = flatten_chain(chain, discard=discard, thin=thin)
        else:
            raise ValueError(f"{path} contains neither 'samples' nor 'chain'")
        truths = np.asarray(data["truths"], dtype=float) if "truths" in data.files else None
        labels = [str(x) for x in data["labels"]] if "labels" in data.files else LABELS
        log_prob = np.asarray(data["log_prob"], dtype=float) if "log_prob" in data.files else None
    return {"samples": samples, "truths": truths, "labels": labels, "log_prob": log_prob, "path": path}


def corner_plot(
    sample_path: str | Path,
    output: str | Path,
    *,
    discard: int | None = None,
    thin: int = 1,
    title: str | None = None,
) -> dict[str, Any]:
    import corner

    loaded = load_samples(sample_path, discard=discard, thin=thin)
    samples = loaded["samples"]
    truths = loaded["truths"]
    stats = parameter_stats(samples)
    fig = corner.corner(
        samples,
        labels=PLOT_LABELS,
        truths=truths,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".4g",
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 11},
        color="#315f9d",
        truth_color="black",
        smooth=1.0,
        plot_datapoints=False,
        fill_contours=True,
        levels=(0.393, 0.865),
        bins=40,
    )
    if title:
        fig.suptitle(title, fontsize=13, y=0.995)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    summary = {
        "input": str(sample_path),
        "output": str(output),
        "n_samples": int(samples.shape[0]),
        "labels": loaded["labels"],
        "truths": None if truths is None else truths,
        "q16": stats["q16"],
        "median": stats["median"],
        "q84": stats["q84"],
        "err_minus": stats["err_minus"],
        "err_plus": stats["err_plus"],
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(json_clean(summary), indent=2), encoding="utf-8")
    return summary
