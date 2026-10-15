# pyright: reportMissingTypeArgument=false, reportArgumentType=false
"""Thirty-run robustness and FFD-vs-random experiment.

The script uses the same production implementation as the main experiment and
records both the randomized initial solution and the final Tabu solution.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from cohesive_bin_packing import contract_instance, load_instance, lower_bound, tabu_search

ROOT = Path(__file__).resolve().parent
INSTANCES_DIR = ROOT / "instances"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
N_RUNS = 30

GROUPS = (
    ("medium", 1000, 12),
    ("hard_medium", 1200, 14),
)


def main() -> None:
    rows: list[dict] = []
    summaries: list[dict] = []

    for group, max_iter, tenure in GROUPS:
        paths = sorted((INSTANCES_DIR / group).glob("*.txt"), key=lambda p: load_instance(p)["n"])
        for path in paths:
            instance = load_instance(path)
            contracted = contract_instance(instance)
            bound = lower_bound(contracted.weights, contracted.capacity)
            instance_rows: list[dict] = []
            for seed in range(1, N_RUNS + 1):
                result = tabu_search(
                    instance,
                    max_iter=max_iter,
                    tabu_tenure=tenure,
                    seed=seed,
                    initial="random",
                    target=bound,
                )
                if result.num_bins is None or result.initial_bins is None or not result.valid:
                    raise RuntimeError(f"Invalid result for {path}, seed={seed}")
                row = {
                    "instance": path.name,
                    "group": group,
                    "n": instance["n"],
                    "seed": seed,
                    "lower_bound": bound,
                    "random_initial_bins": result.initial_bins,
                    "tabu_bins": result.num_bins,
                    "improvement_bins": result.initial_bins - result.num_bins,
                    "time_sec": result.solve_time,
                    "iterations": result.iterations,
                    "valid": result.valid,
                }
                rows.append(row)
                instance_rows.append(row)

            initial = np.array([row["random_initial_bins"] for row in instance_rows], dtype=float)
            final = np.array([row["tabu_bins"] for row in instance_rows], dtype=float)
            times = np.array([row["time_sec"] for row in instance_rows], dtype=float)
            summaries.append(
                {
                    "instance": path.name,
                    "group": group,
                    "n": instance["n"],
                    "runs": N_RUNS,
                    "lower_bound": bound,
                    "initial_mean": round(float(initial.mean()), 3),
                    "initial_std": round(float(initial.std(ddof=1)), 3),
                    "final_mean": round(float(final.mean()), 3),
                    "final_std": round(float(final.std(ddof=1)), 3),
                    "best": int(final.min()),
                    "worst": int(final.max()),
                    "mean_improvement_bins": round(float((initial - final).mean()), 3),
                    "mean_time_sec": round(float(times.mean()), 6),
                    "std_time_sec": round(float(times.std(ddof=1)), 6),
                }
            )
            summary = summaries[-1]
            print(
                f"{group:11s} n={instance['n']:3d}: random={summary['initial_mean']:.2f}±{summary['initial_std']:.2f} "
                f"-> tabu={summary['final_mean']:.2f}±{summary['final_std']:.2f}, "
                f"best={summary['best']}, mean time={summary['mean_time_sec']:.4f}s"
            )

    with (RESULTS_DIR / "robustness.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (RESULTS_DIR / "robustness_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
