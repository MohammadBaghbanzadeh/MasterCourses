# pyright: reportMissingTypeArgument=false, reportArgumentType=false, reportAttributeAccessIssue=false
"""Generate every report figure and the robustness LaTeX table from CSV data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cohesive_bin_packing import (
    bin_loads,
    contract_instance,
    first_fit_decreasing,
    load_instance,
    lower_bound,
    tabu_search,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"
SLIDE_FIGURES = ROOT / "PresentationSlides" / "Final_Presentation" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
SLIDE_FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 300, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25})


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, name: str) -> None:
    for directory in (FIGURES, SLIDE_FIGURES):
        fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(directory / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def best_so_far(history: list[int]) -> list[int]:
    best = history[0]
    result = []
    for value in history:
        best = min(best, value)
        result.append(best)
    return result


def convergence_figure() -> None:
    cases = [
        ("Standard n=150", ROOT / "instances/medium/n=150_C=200_pairs=45.txt", 7, 27),
        ("Synthetic-hard n=100", ROOT / "instances/hard_medium/hard_n=100_C=200_pairs=20.txt", 8, 33),
        ("Synthetic-hard n=150", ROOT / "instances/hard_medium/hard_n=150_C=200_pairs=30.txt", 3, 49),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, path, seed, target in cases:
        result = tabu_search(load_instance(path), max_iter=1200, tabu_tenure=14, seed=seed, initial="random", target=target)
        ax.step(range(len(result.history)), best_so_far(result.history), where="post", linewidth=2, label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best number of bins")
    ax.set_title("Tabu-search convergence from randomized starts")
    ax.legend()
    save(fig, "fig1_convergence")


def time_figure(rows: list[dict]) -> None:
    small = sorted((row for row in rows if row["group"] == "small"), key=lambda row: int(row["n"]))
    x = np.arange(len(small))
    ilp = [max(float(row["exact_time_sec"]), 1e-6) for row in small]
    tabu = [max(float(row["tabu_time_sec"]), 1e-6) for row in small]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    ax.bar(x - width / 2, ilp, width, label="Exact ILP")
    ax.bar(x + width / 2, tabu, width, label="FFD + Tabu")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"n={row['n']}" for row in small])
    ax.set_ylabel("Wall-clock time (s, log scale)")
    ax.set_title("Exact ILP and heuristic running times")
    ax.legend()
    save(fig, "fig2_time_comparison")


def gap_figure(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for kind, marker, label in (("standard", "o", "Standard"), ("synthetic-hard", "s", "Synthetic-hard")):
        subset = sorted((row for row in rows if row["instance_type"] == kind), key=lambda row: int(row["n"]))
        ax.plot(
            [int(row["n"]) for row in subset],
            [float(row["gap_to_reference_pct"]) for row in subset],
            marker=marker,
            linewidth=2,
            label=label,
        )
    ax.set_xlabel("Number of original items (n)")
    ax.set_ylabel("Gap to stated reference (%)")
    ax.set_title("Solution gap by instance size")
    ax.legend()
    save(fig, "fig3_gap_vs_size")


def boxplot_figure(robust_rows: list[dict]) -> None:
    names = []
    values = []
    for group in ("medium", "hard_medium"):
        ns = sorted({int(row["n"]) for row in robust_rows if row["group"] == group})
        for n in ns:
            subset = [int(row["tabu_bins"]) for row in robust_rows if row["group"] == group and int(row["n"]) == n]
            names.append(("Std" if group == "medium" else "Hard") + f"\nn={n}")
            values.append(subset)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    plot = ax.boxplot(values, tick_labels=names, patch_artist=True, showmeans=True)
    for index, box in enumerate(plot["boxes"]):
        box.set_facecolor("#4C72B0" if index < 3 else "#C44E52")
        box.set_alpha(0.7)
    ax.set_ylabel("Final number of bins")
    ax.set_title("Distribution over 30 independent randomized starts")
    save(fig, "fig4_boxplot")


def packing_figure() -> None:
    instance = load_instance(ROOT / "instances/small/n=30_C=100_pairs=10.txt")
    contracted = contract_instance(instance)
    bins = first_fit_decreasing(contracted.weights, contracted.capacity)
    loads = bin_loads(bins, contracted.weights)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottoms = np.zeros(len(bins))
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(contracted.weights), 2)))
    for bin_index, bin_ in enumerate(bins):
        for cid in bin_:
            weight = contracted.weights[cid]
            ax.bar(bin_index, weight, bottom=bottoms[bin_index], color=palette[cid], edgecolor="white")
            bottoms[bin_index] += weight
    ax.axhline(contracted.capacity, color="black", linestyle="--", linewidth=1.2, label="Capacity")
    ax.set_xticks(range(len(bins)), [f"B{i + 1}\nload={load}" for i, load in enumerate(loads)])
    ax.set_ylabel("Load")
    ax.set_title("Validated component packing for the n=30 instance")
    ax.legend()
    save(fig, "fig5_bin_visualization")


def hard_improvement_figure() -> None:
    cases = [
        ("Hard n=150", ROOT / "instances/hard_medium/hard_n=150_C=200_pairs=30.txt", 42),
        ("Hard n=300", ROOT / "instances/hard_large/hard_n=300_C=500_pairs=50.txt", 42),
        ("Hard n=500", ROOT / "instances/hard_large/hard_n=500_C=500_pairs=80.txt", 42),
    ]
    labels, ffd_values, tabu_values, lbs = [], [], [], []
    for label, path, seed in cases:
        instance = load_instance(path)
        contracted = contract_instance(instance)
        labels.append(label)
        ffd_values.append(len(first_fit_decreasing(contracted.weights, contracted.capacity)))
        lbs.append(lower_bound(contracted.weights, contracted.capacity))
        tabu_values.append(tabu_search(instance, max_iter=1500, tabu_tenure=16, seed=seed, target=lbs[-1]).num_bins)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    width = 0.25
    ax.bar(x - width, ffd_values, width, label="FFD")
    ax.bar(x, tabu_values, width, label="FFD + Tabu")
    ax.bar(x + width, lbs, width, label="Lower bound")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Number of bins")
    ax.set_title("Improvement on synthetic-hard instances")
    ax.legend()
    save(fig, "fig6_pathological")


def random_comparison_figures(summary_rows: list[dict]) -> None:
    labels = [("Std" if row["group"] == "medium" else "Hard") + f" n={row['n']}" for row in summary_rows]
    initial = np.array([float(row["initial_mean"]) for row in summary_rows])
    final = np.array([float(row["final_mean"]) for row in summary_rows])
    lb = np.array([float(row["lower_bound"]) for row in summary_rows])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.35
    ax.bar(x - width / 2, 100 * (initial - lb) / lb, width, label="Random initial")
    ax.bar(x + width / 2, 100 * (final - lb) / lb, width, label="After Tabu")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("Gap to lower bound (%)")
    ax.set_title("Random-start gap before and after Tabu search (30 runs)")
    ax.legend()
    save(fig, "gap_comparison_random_vs_ffd")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x, initial - final, color=["#4C72B0" if row["group"] == "medium" else "#C44E52" for row in summary_rows])
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("Mean bins removed")
    ax.set_title("Mean improvement produced by Tabu search")
    save(fig, "tabu_improvement_on_random")

    convergence_figure()
    # Keep the historical filename used by the report, with real convergence data.
    source = FIGURES / "fig1_convergence.png"
    for directory in (FIGURES, SLIDE_FIGURES):
        (directory / "random_vs_ffd_convergence.png").write_bytes(source.read_bytes())
    source_pdf = FIGURES / "fig1_convergence.pdf"
    for directory in (FIGURES, SLIDE_FIGURES):
        (directory / "random_vs_ffd_convergence.pdf").write_bytes(source_pdf.read_bytes())


def write_robustness_table(summary_rows: list[dict]) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"نمونه & اجرا & میانگین اولیه & میانگین نهایی & انحراف معیار نهایی & بهترین & زمان میانگین (s) \\",
        r"\midrule",
    ]
    for row in summary_rows:
        short = ("Std" if row["group"] == "medium" else "Hard") + f" n={row['n']}"
        lines.append(
            f"\\lr{{{short}}} & {row['runs']} & {float(row['initial_mean']):.2f} & "
            f"{float(row['final_mean']):.2f} & {float(row['final_std']):.2f} & {row['best']} & "
            f"{float(row['mean_time_sec']):.4f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{تحلیل پایداری در ۳۰ شروع تصادفی مستقل؛ اعداد مستقیماً از فایل \lr{robustness.csv} تولید شده‌اند.}",
            r"\label{tab:robustness}",
            r"\end{table}",
            "",
        ]
    )
    (RESULTS / "robustness_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = read_csv(RESULTS / "results_final.csv")
    robust_rows = read_csv(RESULTS / "robustness.csv")
    summary_rows = read_csv(RESULTS / "robustness_summary.csv")
    time_figure(rows)
    gap_figure(rows)
    boxplot_figure(robust_rows)
    packing_figure()
    hard_improvement_figure()
    random_comparison_figures(summary_rows)
    write_robustness_table(summary_rows)
    print("Generated report and slide artifacts from validated CSV files.")


if __name__ == "__main__":
    main()
