# Cohesive Bin Packing

Final project for Combinatorial Optimization: exact and heuristic solution of one-dimensional bin packing with hard must-link constraints.

## Reproducible workflow

Python 3.12 is recommended.

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 run_experiments_final.py
python3 run_robustness.py
python3 generate_report_artifacts.py
python3 build_notebook.py
jupyter nbconvert --to notebook --execute --inplace BinPacking_Cohesive.ipynb
latexmk -cd -xelatex report/main.tex
latexmk -cd -xelatex PresentationSlides/Final_Presentation/main_presentation.tex
```

## Single source of truth

- `cohesive_bin_packing.py`: contraction, exact ILP, FFD, randomized initialization, Tabu Search, and validation.
- `run_experiments_final.py`: all 17 main experiments and generated LaTeX tables.
- `run_robustness.py`: 30 independent randomized starts per medium instance.
- `generate_report_artifacts.py`: all report/presentation charts from CSV data.
- `tests/test_cohesive_bin_packing.py`: automated correctness tests.

## Interpretation of reported gaps

For instances with at most 30 original items, the reference is a CBC-proven ILP optimum. For larger instances, the reference is a valid lower bound. Reaching that lower bound proves optimality; a positive gap to the lower bound is not necessarily the true optimality gap.

The `hard_*` instances are synthetic difficult mixtures inspired by bin-packing literature; they are not official Falkenauer benchmark instances.
