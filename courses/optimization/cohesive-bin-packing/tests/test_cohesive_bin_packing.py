import unittest
from pathlib import Path
from typing import override

from cohesive_bin_packing import (
    apply_move,
    bin_loads,
    calculate_delta,
    contract_instance,
    expand_solution,
    first_fit_decreasing,
    generate_neighborhood,
    load_instance,
    Instance,
    lower_bound,
    solve_exact_ilp,
    tabu_search,
    validate_component_solution,
    validate_item_solution,
)


ROOT = Path(__file__).resolve().parents[1]


class CohesiveBinPackingTests(unittest.TestCase):
    instance: Instance = {"n": 0, "C": 1, "weights": [], "pairs": []}

    @override
    def setUp(self) -> None:
        self.instance = {
            "n": 6,
            "C": 10,
            "weights": [2, 3, 4, 1, 5, 5],
            "pairs": [(0, 1), (2, 3)],
        }

    def test_contraction_is_exact_and_transitive(self):
        contracted = contract_instance(self.instance)
        self.assertEqual(contracted.components, ((0, 1), (2, 3), (4,), (5,)))
        self.assertEqual(contracted.weights, (5, 5, 5, 5))

    def test_lower_bound_and_ffd(self):
        contracted = contract_instance(self.instance)
        self.assertEqual(lower_bound(contracted.weights, contracted.capacity), 2)
        bins = first_fit_decreasing(contracted.weights, contracted.capacity)
        self.assertEqual(len(bins), 2)
        self.assertTrue(validate_component_solution(bins, contracted.weights, contracted.capacity)[0])
        self.assertTrue(validate_item_solution(self.instance, expand_solution(bins, contracted.components))[0])

    def test_delta_matches_applied_move(self):
        weights = (4, 3, 3, 2)
        capacity = 7
        bins = [[0, 1], [2, 3]]
        loads = bin_loads(bins, weights)
        for move in generate_neighborhood(bins, weights, capacity, include_eject=True):
            before_count = len(bins)
            before_sq = sum(load * load for load in loads)
            after = apply_move(bins, move)
            after_loads = bin_loads(after, weights)
            expected = (len(after) - before_count, sum(load * load for load in after_loads) - before_sq)
            self.assertEqual(calculate_delta(bins, loads, weights, move), expected, move)

    def test_infeasible_component_is_rejected(self):
        instance: Instance = {"n": 2, "C": 5, "weights": [3, 3], "pairs": [(0, 1)]}
        result = tabu_search(instance)
        self.assertIsNone(result.num_bins)
        self.assertFalse(result.valid)

    def test_small_real_instance_matches_exact_optimum(self):
        instance = load_instance(ROOT / "instances/small/n=30_C=100_pairs=10.txt")
        exact = solve_exact_ilp(instance, time_limit=30)
        heuristic = tabu_search(instance, max_iter=500, tabu_tenure=7, target=6)
        self.assertTrue(exact["optimal"])
        self.assertEqual(exact["num_bins"], 6)
        self.assertEqual(heuristic.num_bins, exact["num_bins"])
        self.assertTrue(heuristic.valid)


if __name__ == "__main__":
    _ = unittest.main()
