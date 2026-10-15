"""Core algorithms for cohesive one-dimensional bin packing.

Hard must-link constraints are contracted into connected components.  The
resulting instance is exactly a classical one-dimensional bin-packing problem
on component weights; every function in this module works on that contracted
representation and maps solutions back to original items when needed.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import networkx as nx
from pulp import (
    LpBinary,
    LpMinimize,
    LpProblem,
    LpStatusOptimal,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
    value,
)

Move = tuple[Any, ...]


class Instance(TypedDict):
    n: int
    C: int
    weights: list[int]
    pairs: list[tuple[int, int]]


@dataclass(frozen=True)
class ContractedInstance:
    capacity: int
    components: tuple[tuple[int, ...], ...]
    weights: tuple[int, ...]


@dataclass
class SearchResult:
    num_bins: int | None
    bins: list[list[int]] | None
    components: tuple[tuple[int, ...], ...]
    component_weights: tuple[int, ...]
    history: list[int]
    solve_time: float
    initial_bins: int | None
    iterations: int
    seed: int | None
    valid: bool


def load_instance(filepath: str | Path) -> Instance:
    data: dict[str, Any] = {}
    with Path(filepath).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            key, raw_value = line.split("=", 1)
            if key in {"n", "C"}:
                data[key] = int(raw_value)
            elif key == "weights":
                data[key] = [int(v) for v in raw_value.split(",") if v]
            elif key == "pairs":
                data[key] = [tuple(map(int, pair.split("-"))) for pair in raw_value.split(",") if pair]

    required = {"n", "C", "weights", "pairs"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Missing fields in {filepath}: {sorted(missing)}")
    if data["n"] != len(data["weights"]):
        raise ValueError(f"Filename/data instance has n={data['n']} but {len(data['weights'])} weights")
    if data["C"] <= 0 or any(weight <= 0 for weight in data["weights"]):
        raise ValueError("Capacity and item weights must be positive")
    for i, j in data["pairs"]:
        if not (0 <= i < data["n"] and 0 <= j < data["n"] and i != j):
            raise ValueError(f"Invalid cohesion edge {(i, j)}")
    return cast(Instance, cast(object, data))


def contract_instance(instance: Instance) -> ContractedInstance:
    graph = nx.Graph()
    graph.add_nodes_from(range(instance["n"]))
    graph.add_edges_from(instance["pairs"])
    components = tuple(
        sorted((tuple(sorted(component)) for component in nx.connected_components(graph)), key=lambda c: c[0])
    )
    weights = tuple(sum(instance["weights"][item] for item in component) for component in components)
    return ContractedInstance(instance["C"], components, weights)


def is_feasible_contracted(contracted: ContractedInstance) -> bool:
    return all(weight <= contracted.capacity for weight in contracted.weights)


def lower_bound(component_weights: Sequence[int], capacity: int) -> int:
    """A valid lower bound combining volume and items heavier than C/2."""
    if capacity <= 0:
        raise ValueError("Capacity must be positive")
    volume = math.ceil(sum(component_weights) / capacity)
    large_items = sum(weight > capacity / 2 for weight in component_weights)
    return max(volume, large_items)


def first_fit_decreasing(component_weights: Sequence[int], capacity: int) -> list[list[int]]:
    order = sorted(range(len(component_weights)), key=lambda cid: (-component_weights[cid], cid))
    bins: list[list[int]] = []
    loads: list[int] = []
    for cid in order:
        weight = component_weights[cid]
        for bin_index, load in enumerate(loads):
            if load + weight <= capacity:
                bins[bin_index].append(cid)
                loads[bin_index] += weight
                break
        else:
            bins.append([cid])
            loads.append(weight)
    return bins


def randomized_first_fit(
    component_weights: Sequence[int], capacity: int, seed: int
) -> list[list[int]]:
    rng = random.Random(seed)
    order = list(range(len(component_weights)))
    rng.shuffle(order)
    bins: list[list[int]] = []
    loads: list[int] = []
    for cid in order:
        weight = component_weights[cid]
        feasible_bins = [index for index, load in enumerate(loads) if load + weight <= capacity]
        if feasible_bins:
            destination = rng.choice(feasible_bins)
            bins[destination].append(cid)
            loads[destination] += weight
        else:
            bins.append([cid])
            loads.append(weight)
    return bins


def bin_loads(bins: Sequence[Sequence[int]], component_weights: Sequence[int]) -> list[int]:
    return [sum(component_weights[cid] for cid in bin_) for bin_ in bins]


def validate_component_solution(
    bins: Sequence[Sequence[int]], component_weights: Sequence[int], capacity: int
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    flattened = [cid for bin_ in bins for cid in bin_]
    expected = list(range(len(component_weights)))
    if sorted(flattened) != expected:
        errors.append("Every component must occur exactly once")
    for index, load in enumerate(bin_loads(bins, component_weights)):
        if load > capacity:
            errors.append(f"Bin {index} exceeds capacity: {load}>{capacity}")
    return not errors, errors


def expand_solution(
    bins: Sequence[Sequence[int]], components: Sequence[Sequence[int]]
) -> list[list[int]]:
    return [[item for cid in bin_ for item in components[cid]] for bin_ in bins]


def validate_item_solution(instance: Instance, item_bins: Sequence[Sequence[int]]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    item_to_bin: dict[int, int] = {}
    for bin_index, bin_ in enumerate(item_bins):
        load = sum(instance["weights"][item] for item in bin_)
        if load > instance["C"]:
            errors.append(f"Bin {bin_index} exceeds capacity: {load}>{instance['C']}")
        for item in bin_:
            if item in item_to_bin:
                errors.append(f"Item {item} occurs more than once")
            item_to_bin[item] = bin_index
    if set(item_to_bin) != set(range(instance["n"])):
        errors.append("Every original item must occur exactly once")
    for i, j in instance["pairs"]:
        if item_to_bin.get(i) != item_to_bin.get(j):
            errors.append(f"Cohesion edge {(i, j)} is violated")
    return not errors, errors


def generate_neighborhood(
    bins: Sequence[Sequence[int]], component_weights: Sequence[int], capacity: int, include_eject: bool = True
) -> list[Move]:
    moves: list[Move] = []
    loads = bin_loads(bins, component_weights)
    number_of_bins = len(bins)

    for source, source_bin in enumerate(bins):
        for cid in source_bin:
            weight = component_weights[cid]
            for destination in range(number_of_bins):
                if source != destination and loads[destination] + weight <= capacity:
                    moves.append(("move", cid, source, destination))
            if include_eject and len(source_bin) > 1:
                moves.append(("eject", cid, source))

    for first_bin in range(number_of_bins):
        for second_bin in range(first_bin + 1, number_of_bins):
            for first_cid in bins[first_bin]:
                first_weight = component_weights[first_cid]
                for second_cid in bins[second_bin]:
                    second_weight = component_weights[second_cid]
                    if (
                        loads[first_bin] - first_weight + second_weight <= capacity
                        and loads[second_bin] - second_weight + first_weight <= capacity
                    ):
                        moves.append(("swap", first_cid, first_bin, second_cid, second_bin))
            if loads[first_bin] + loads[second_bin] <= capacity:
                moves.append(("merge", first_bin, second_bin))
    return moves


def move_components(move: Move, bins: Sequence[Sequence[int]]) -> tuple[int, ...]:
    if move[0] in {"move", "eject"}:
        return (move[1],)
    if move[0] == "swap":
        return (move[1], move[3])
    if move[0] == "merge":
        return tuple(bins[move[2]])
    raise ValueError(f"Unknown move: {move}")


def calculate_delta(
    bins: Sequence[Sequence[int]], loads: Sequence[int], component_weights: Sequence[int], move: Move
) -> tuple[int, int]:
    """Return (change in bin count, change in sum of squared loads) in O(1)."""
    kind = move[0]
    if kind == "move":
        _, cid, source, destination = move
        weight = component_weights[cid]
        delta_bins = -1 if len(bins[source]) == 1 else 0
        delta_sq = (
            (loads[source] - weight) ** 2
            + (loads[destination] + weight) ** 2
            - loads[source] ** 2
            - loads[destination] ** 2
        )
        return delta_bins, delta_sq
    if kind == "eject":
        _, cid, source = move
        weight = component_weights[cid]
        return 1, (loads[source] - weight) ** 2 + weight**2 - loads[source] ** 2
    if kind == "swap":
        _, first_cid, first_bin, second_cid, second_bin = move
        first_weight = component_weights[first_cid]
        second_weight = component_weights[second_cid]
        new_first = loads[first_bin] - first_weight + second_weight
        new_second = loads[second_bin] - second_weight + first_weight
        return 0, new_first**2 + new_second**2 - loads[first_bin] ** 2 - loads[second_bin] ** 2
    if kind == "merge":
        _, first_bin, second_bin = move
        return -1, (loads[first_bin] + loads[second_bin]) ** 2 - loads[first_bin] ** 2 - loads[second_bin] ** 2
    raise ValueError(f"Unknown move: {move}")


def apply_move(bins: Sequence[Sequence[int]], move: Move) -> list[list[int]]:
    new_bins = [list(bin_) for bin_ in bins]
    kind = move[0]
    if kind == "move":
        _, cid, source, destination = move
        new_bins[source].remove(cid)
        new_bins[destination].append(cid)
    elif kind == "eject":
        _, cid, source = move
        new_bins[source].remove(cid)
        new_bins.append([cid])
    elif kind == "swap":
        _, first_cid, first_bin, second_cid, second_bin = move
        new_bins[first_bin].remove(first_cid)
        new_bins[second_bin].remove(second_cid)
        new_bins[first_bin].append(second_cid)
        new_bins[second_bin].append(first_cid)
    elif kind == "merge":
        _, first_bin, second_bin = move
        new_bins[first_bin].extend(new_bins[second_bin])
        new_bins[second_bin].clear()
    else:
        raise ValueError(f"Unknown move: {move}")
    return [bin_ for bin_ in new_bins if bin_]


def tabu_search(
    instance: Instance,
    *,
    max_iter: int = 1000,
    tabu_tenure: int = 12,
    seed: int = 42,
    initial: str = "ffd",
    stagnation_limit: int | None = None,
    target: int | None = None,
) -> SearchResult:
    contracted = contract_instance(instance)
    started = time.perf_counter()
    if not is_feasible_contracted(contracted):
        return SearchResult(None, None, contracted.components, contracted.weights, [], 0.0, None, 0, seed, False)

    if initial == "ffd":
        bins = first_fit_decreasing(contracted.weights, contracted.capacity)
    elif initial == "random":
        bins = randomized_first_fit(contracted.weights, contracted.capacity, seed)
    else:
        raise ValueError("initial must be 'ffd' or 'random'")

    rng = random.Random(seed)
    best_bins = [list(bin_) for bin_ in bins]
    best_cost = len(best_bins)
    initial_cost = best_cost
    history = [best_cost]
    tabu_until: dict[int, int] = {}
    no_improvement = 0
    stagnation_limit = stagnation_limit or max(75, max_iter // 8)
    iterations = 0

    if target is not None and best_cost <= target:
        valid, _ = validate_component_solution(best_bins, contracted.weights, contracted.capacity)
        return SearchResult(
            best_cost,
            best_bins,
            contracted.components,
            contracted.weights,
            history,
            round(time.perf_counter() - started, 6),
            initial_cost,
            0,
            seed,
            valid,
        )

    for iteration in range(max_iter):
        iterations = iteration + 1
        loads = bin_loads(bins, contracted.weights)
        neighborhood = generate_neighborhood(bins, contracted.weights, contracted.capacity, include_eject=True)
        if not neighborhood:
            break

        current_cost = len(bins)
        admissible: list[tuple[tuple[int, int, float], Move]] = []
        eject_candidates: list[tuple[tuple[int, int, float], Move]] = []
        for move in neighborhood:
            delta_bins, delta_sq = calculate_delta(bins, loads, contracted.weights, move)
            moved = move_components(move, bins)
            is_tabu = any(tabu_until.get(cid, -1) >= iteration for cid in moved)
            aspiration = current_cost + delta_bins < best_cost
            if is_tabu and not aspiration:
                continue
            tie_break = rng.random()
            candidate = ((delta_bins, -delta_sq, tie_break), move)
            if move[0] == "eject":
                eject_candidates.append(candidate)
            else:
                admissible.append(candidate)

        # Intensify normally.  After sustained stagnation, force one admissible
        # EJECT move (at most best+1 bins) to diversify before continuing.
        force_eject = no_improvement > 0 and no_improvement % max(20, stagnation_limit // 3) == 0
        pool = eject_candidates if force_eject and eject_candidates and current_cost <= best_cost else admissible
        if not pool:
            pool = eject_candidates
        if not pool:
            break

        _, selected = min(pool, key=lambda candidate: candidate[0])
        moved = move_components(selected, bins)
        bins = apply_move(bins, selected)
        tenure = rng.randint(max(3, tabu_tenure - 2), tabu_tenure + 2)
        for cid in moved:
            tabu_until[cid] = iteration + tenure

        current_cost = len(bins)
        history.append(current_cost)
        if current_cost < best_cost:
            best_cost = current_cost
            best_bins = [list(bin_) for bin_ in bins]
            no_improvement = 0
            if target is not None and best_cost <= target:
                break
        else:
            no_improvement += 1
        if no_improvement >= stagnation_limit:
            break

    valid, _ = validate_component_solution(best_bins, contracted.weights, contracted.capacity)
    item_valid, _ = validate_item_solution(instance, expand_solution(best_bins, contracted.components))
    return SearchResult(
        best_cost,
        best_bins,
        contracted.components,
        contracted.weights,
        history,
        round(time.perf_counter() - started, 6),
        initial_cost,
        iterations,
        seed,
        valid and item_valid,
    )


def multi_start_tabu_search(
    instance: Instance,
    *,
    seeds: Iterable[int],
    max_iter: int = 1000,
    tabu_tenure: int = 12,
    target: int | None = None,
) -> tuple[SearchResult, list[SearchResult]]:
    results = [
        tabu_search(
            instance,
            max_iter=max_iter,
            tabu_tenure=tabu_tenure,
            seed=seed,
            initial="random",
            target=target,
        )
        for seed in seeds
    ]
    feasible_results = [result for result in results if result.num_bins is not None and result.valid]
    if not feasible_results:
        raise ValueError("No feasible result was produced")
    best = min(feasible_results, key=lambda result: (result.num_bins, result.solve_time))
    return best, results


def solve_exact_ilp(instance: Instance, *, time_limit: int = 120, msg: bool = False) -> dict[str, Any]:
    """Solve the exact component-level formulation and accept only optimal CBC output."""
    contracted = contract_instance(instance)
    if not is_feasible_contracted(contracted):
        return {"status": "Infeasible", "status_code": -1, "optimal": False, "num_bins": None, "solve_time": 0.0}

    number_of_components = len(contracted.weights)
    candidate_bins = range(number_of_components)
    problem = LpProblem("CohesiveBinPackingContracted", LpMinimize)
    x = {
        (cid, bin_index): LpVariable(f"x_{cid}_{bin_index}", cat=LpBinary)
        for cid in range(number_of_components)
        for bin_index in candidate_bins
    }
    y = {bin_index: LpVariable(f"y_{bin_index}", cat=LpBinary) for bin_index in candidate_bins}
    problem += lpSum(y[bin_index] for bin_index in candidate_bins)
    for cid in range(number_of_components):
        problem += lpSum(x[cid, bin_index] for bin_index in candidate_bins) == 1
    for bin_index in candidate_bins:
        problem += (
            lpSum(contracted.weights[cid] * x[cid, bin_index] for cid in range(number_of_components))
            <= contracted.capacity * y[bin_index]
        )
    for bin_index in range(number_of_components - 1):
        problem += y[bin_index] >= y[bin_index + 1]

    started = time.perf_counter()
    status_code = problem.solve(PULP_CBC_CMD(msg=msg, timeLimit=time_limit, gapRel=0.0))
    elapsed = round(time.perf_counter() - started, 6)
    optimal = status_code == LpStatusOptimal
    objective = cast(float | None, value(problem.objective)) if optimal else None
    return {
        "status": "Optimal" if optimal else "Not optimal",
        "status_code": status_code,
        "optimal": optimal,
        "num_bins": int(round(objective)) if objective is not None else None,
        "solve_time": elapsed,
    }
