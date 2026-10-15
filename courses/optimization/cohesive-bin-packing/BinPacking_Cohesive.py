#!/usr/bin/env python3
"""Educational entry point for the cohesive-bin-packing project.

The production implementation lives in :mod:`cohesive_bin_packing`.  Keeping
this file as a thin entry point prevents the notebook, scripts, and report from
silently using divergent algorithm versions.
"""

from pathlib import Path

from cohesive_bin_packing import (
    contract_instance,
    first_fit_decreasing,
    load_instance,
    lower_bound,
    solve_exact_ilp,
    tabu_search,
)


def demo() -> None:
    root = Path(__file__).resolve().parent
    path = root / "instances" / "small" / "n=30_C=100_pairs=10.txt"
    instance = load_instance(path)
    contracted = contract_instance(instance)
    bound = lower_bound(contracted.weights, contracted.capacity)
    ffd = first_fit_decreasing(contracted.weights, contracted.capacity)
    exact = solve_exact_ilp(instance, time_limit=120)
    result = tabu_search(instance, max_iter=500, tabu_tenure=7, target=exact["num_bins"] or bound)

    print(f"Instance: {path.name}")
    print(f"Original items: {instance['n']} | contracted components: {len(contracted.components)}")
    print(f"Lower bound: {bound} | FFD: {len(ffd)}")
    print(f"Exact ILP: {exact['num_bins']} ({exact['status']})")
    print(f"Tabu: {result.num_bins} | valid={result.valid} | time={result.solve_time:.6f}s")


if __name__ == "__main__":
    demo()
