# Exported notebook code; may contain IPython commands.
# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

from cohesive_bin_packing import (
    load_instance, contract_instance, lower_bound,
    first_fit_decreasing, solve_exact_ilp, tabu_search,
    expand_solution, validate_item_solution,
)

ROOT = Path.cwd()
assert (ROOT / 'cohesive_bin_packing.py').exists(), 'Notebook must run from the project root'
print('Project root:', ROOT)

# %%
instance = load_instance(ROOT / 'instances/small/n=30_C=100_pairs=10.txt')
contracted = contract_instance(instance)
print('n =', instance['n'], '| capacity =', instance['C'], '| edges =', len(instance['pairs']))
print('connected components =', len(contracted.components))
print('component weights =', contracted.weights)
print('lower bound =', lower_bound(contracted.weights, contracted.capacity))

# %%
exact = solve_exact_ilp(instance, time_limit=120)
exact

# %%
bound = lower_bound(contracted.weights, contracted.capacity)
ffd_bins = first_fit_decreasing(contracted.weights, contracted.capacity)
result = tabu_search(instance, max_iter=500, tabu_tenure=7, seed=42,
                     initial='ffd', target=exact['num_bins'] or bound)
item_bins = expand_solution(result.bins, result.components)
valid, errors = validate_item_solution(instance, item_bins)
print('FFD bins:', len(ffd_bins))
print('Tabu bins:', result.num_bins, '| exact:', exact['num_bins'])
print('valid:', valid, '| validation errors:', errors)
print('iterations:', result.iterations, '| time:', result.solve_time)

# %%
hard = load_instance(ROOT / 'instances/hard_large/hard_n=300_C=500_pairs=50.txt')
hard_contracted = contract_instance(hard)
hard_lb = lower_bound(hard_contracted.weights, hard_contracted.capacity)
hard_ffd = first_fit_decreasing(hard_contracted.weights, hard_contracted.capacity)
hard_result = tabu_search(hard, max_iter=1500, tabu_tenure=16, seed=42, target=hard_lb)
print('LB:', hard_lb, '| FFD:', len(hard_ffd), '| Tabu:', hard_result.num_bins)
print('improvement:', len(hard_ffd) - hard_result.num_bins, 'bin | valid:', hard_result.valid)
plt.step(range(len(hard_result.history)), hard_result.history, where='post')
plt.axhline(hard_lb, color='red', linestyle='--', label='Lower bound')
plt.xlabel('Iteration'); plt.ylabel('Current bins'); plt.legend(); plt.show()

# %%
results = pd.read_csv(ROOT / 'results/results_final.csv')
columns = ['instance', 'n', 'instance_type', 'ffd_bins', 'tabu_bins',
           'reference_value', 'reference_type', 'gap_to_reference_pct', 'valid']
display(results[columns])
assert len(results) == 17 and results['valid'].all()

# %%
robustness = pd.read_csv(ROOT / 'results/robustness_summary.csv')
display(robustness[['instance', 'runs', 'initial_mean', 'initial_std',
                    'final_mean', 'final_std', 'best', 'mean_time_sec']])
assert (robustness['runs'] == 30).all()

# %%
display(Image(filename=str(ROOT / 'report/figures/fig1_convergence.png'), width=800))
display(Image(filename=str(ROOT / 'report/figures/fig4_boxplot.png'), width=800))

# %%
import subprocess, sys
completed = subprocess.run(
    [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
    cwd=ROOT, text=True, capture_output=True, check=True,
)
print(completed.stderr)
print('All automated tests passed.')
