# pyright: reportMissingTypeArgument=false, reportArgumentType=false, reportOperatorIssue=false
"""Reproducible experiment pipeline for the final report.

Every reported number is generated from ``cohesive_bin_packing.py``.  The
script evaluates all standard and synthetic-hard text instances, validates
every heuristic solution, solves n<=30 instances exactly, and writes both CSV
and LaTeX tables.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from cohesive_bin_packing import (
    contract_instance,
    first_fit_decreasing,
    load_instance,
    lower_bound,
    solve_exact_ilp,
    tabu_search,
)

ROOT = Path(__file__).resolve().parent
INSTANCES_DIR = ROOT / "instances"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GROUPS = (
    ("small", "standard", 500, 7),
    ("medium", "standard", 1000, 12),
    ("large", "standard", 1500, 15),
    ("hard_small", "synthetic-hard", 800, 10),
    ("hard_medium", "synthetic-hard", 1200, 14),
    ("hard_large", "synthetic-hard", 1500, 16),
)

FIELDNAMES = [
    "instance",
    "group",
    "instance_type",
    "n",
    "capacity",
    "cohesion_edges",
    "components",
    "lower_bound",
    "exact_bins",
    "exact_status",
    "exact_time_sec",
    "reference_value",
    "reference_type",
    "ffd_bins",
    "tabu_bins",
    "improvement_bins",
    "gap_to_reference_pct",
    "tabu_time_sec",
    "iterations",
    "valid",
]


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def fmt(value: object, digits: int = 3) -> str:
    if value in {None, ""}:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def run_all() -> tuple[list[dict], dict[str, list[int]]]:
    rows: list[dict] = []
    histories: dict[str, list[int]] = {}

    for group, instance_type, max_iter, tenure in GROUPS:
        for path in sorted((INSTANCES_DIR / group).glob("*.txt"), key=lambda p: load_instance(p)["n"]):
            instance = load_instance(path)
            contracted = contract_instance(instance)
            bound = lower_bound(contracted.weights, contracted.capacity)
            ffd_bins = len(first_fit_decreasing(contracted.weights, contracted.capacity))

            exact = {
                "num_bins": None,
                "status": "Not run (n>30)",
                "solve_time": None,
                "optimal": False,
            }
            if instance["n"] <= 30:
                exact = solve_exact_ilp(instance, time_limit=120)

            reference_value = exact["num_bins"] if exact["optimal"] else bound
            reference_type = "ILP optimum" if exact["optimal"] else "lower bound"
            search = tabu_search(
                instance,
                max_iter=max_iter,
                tabu_tenure=tenure,
                seed=42,
                initial="ffd",
                target=reference_value,
            )
            if search.num_bins is None:
                raise RuntimeError(f"No feasible solution for {path}")
            gap = 100.0 * (search.num_bins - reference_value) / max(reference_value, 1)
            row = {
                "instance": path.name,
                "group": group,
                "instance_type": instance_type,
                "n": instance["n"],
                "capacity": instance["C"],
                "cohesion_edges": len(instance["pairs"]),
                "components": len(contracted.components),
                "lower_bound": bound,
                "exact_bins": exact["num_bins"] if exact["optimal"] else "",
                "exact_status": exact["status"],
                "exact_time_sec": exact["solve_time"] if exact["optimal"] else "",
                "reference_value": reference_value,
                "reference_type": reference_type,
                "ffd_bins": ffd_bins,
                "tabu_bins": search.num_bins,
                "improvement_bins": ffd_bins - search.num_bins,
                "gap_to_reference_pct": round(gap, 2),
                "tabu_time_sec": search.solve_time,
                "iterations": search.iterations,
                "valid": search.valid,
            }
            if not row["valid"]:
                raise RuntimeError(f"Invalid heuristic solution for {path}")
            rows.append(row)
            histories[path.name] = search.history
            print(
                f"{group:12s} n={instance['n']:4d} LB={bound:3d} FFD={ffd_bins:3d} "
                f"Tabu={search.num_bins:3d} ref={reference_value:3d} "
                f"gap={gap:5.2f}% time={search.solve_time:.4f}s"
            )
    return rows, histories


def write_csv(rows: list[dict]) -> None:
    with (RESULTS_DIR / "results_final.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(rows: list[dict], *, hard: bool) -> None:
    selected = [row for row in rows if (row["instance_type"] == "synthetic-hard") == hard]
    selected.sort(key=lambda row: row["n"])
    output = RESULTS_DIR / ("hard_results_table.tex" if hard else "results_table.tex")
    caption = (
        "نتایج نمونه‌های مصنوعی دشوار؛ نوع مقدار مرجع در ستون «مرجع» مشخص شده است."
        if hard
        else "نتایج نمونه‌های استاندارد؛ نوع مقدار مرجع در ستون «مرجع» مشخص شده است."
    )
    label = "tab:hard-results" if hard else "tab:standard-results"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{r l r r r r l r r}",
        r"\toprule",
        r"ردیف & نمونه & $n$ & \lr{FFD} & \lr{Tabu} & مقدار مرجع & مرجع & شکاف (\%) & زمان (s) \\",
        r"\midrule",
    ]
    for index, row in enumerate(selected, 1):
        reference_type = r"\lr{ILP}" if row["reference_type"] == "ILP optimum" else "کران پایین"
        lines.append(
            f"{index} & \\lr{{{latex_escape(row['instance'])}}} & {row['n']} & {row['ffd_bins']} & "
            f"{row['tabu_bins']} & {row['reference_value']} & {reference_type} & "
            f"{row['gap_to_reference_pct']:.2f} & {row['tabu_time_sec']:.4f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows, histories = run_all()
    write_csv(rows)
    write_latex_table(rows, hard=False)
    write_latex_table(rows, hard=True)
    (RESULTS_DIR / "histories.json").write_text(
        json.dumps(histories, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {len(rows)} validated rows to {RESULTS_DIR / 'results_final.csv'}")


if __name__ == "__main__":
    main()
