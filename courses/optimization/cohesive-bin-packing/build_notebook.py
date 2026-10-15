"""Build the submitted notebook from the production API and execute it separately."""

from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}

cells = []
cells.append(nbf.v4.new_markdown_cell("""# مسئله بسته‌بندی با اقلام چسبنده
## نوت‌بوک بازتولیدپذیر پروژه نهایی

این نوت‌بوک یک رابط آموزشی روی پیاده‌سازی واحد `cohesive_bin_packing.py` است. تمام اعداد گزارش و اسلایدها توسط اسکریپت‌های `run_experiments_final.py`، `run_robustness.py` و `generate_report_artifacts.py` از همین API تولید می‌شوند.

**مراحل:** خواندن داده ← انقباض مؤلفه‌ها ← ILP دقیق ← FFD و Tabu ← اعتبارسنجی ← نتایج آماری."""))

cells.append(nbf.v4.new_code_cell("""from pathlib import Path
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
print('Project root:', ROOT)"""))

cells.append(nbf.v4.new_markdown_cell("""## ۱. داده و انقباض گراف

هر مؤلفه همبند گراف چسبندگی به یک اَبَرقلم تبدیل می‌شود. این تبدیل دقیق است: جواب‌های مسئله منقبض‌شده و مسئله اصلی تناظر یک‌به‌یک دارند."""))

cells.append(nbf.v4.new_code_cell("""instance = load_instance(ROOT / 'instances/small/n=30_C=100_pairs=10.txt')
contracted = contract_instance(instance)
print('n =', instance['n'], '| capacity =', instance['C'], '| edges =', len(instance['pairs']))
print('connected components =', len(contracted.components))
print('component weights =', contracted.weights)
print('lower bound =', lower_bound(contracted.weights, contracted.capacity))"""))

cells.append(nbf.v4.new_markdown_cell(r"""## ۲. مدل دقیق component-level

پس از contraction، متغیر $x_{cj}$ قرارگیری مؤلفه $c$ در سطل $j$ و متغیر $y_j$ استفاده از سطل را نشان می‌دهد:

$$\min \sum_j y_j$$
$$\sum_j x_{cj}=1\quad\forall c$$
$$\sum_c W_c x_{cj}\le C y_j\quad\forall j$$

فقط خروجی CBC با وضعیت `Optimal` به‌عنوان optimum پذیرفته می‌شود."""))

cells.append(nbf.v4.new_code_cell("""exact = solve_exact_ilp(instance, time_limit=120)
exact"""))

cells.append(nbf.v4.new_markdown_cell("""## ۳. الگوریتم پیشنهادی

جواب اولیه با FFD یا Randomized First Fit ساخته می‌شود. جستجوی تابو از `MOVE`، `SWAP`، `MERGE` و `EJECT`، delta ثابت، tabu مبتنی بر شناسه مؤلفه، aspiration و tenure پویا استفاده می‌کند."""))

cells.append(nbf.v4.new_code_cell("""bound = lower_bound(contracted.weights, contracted.capacity)
ffd_bins = first_fit_decreasing(contracted.weights, contracted.capacity)
result = tabu_search(instance, max_iter=500, tabu_tenure=7, seed=42,
                     initial='ffd', target=exact['num_bins'] or bound)
item_bins = expand_solution(result.bins, result.components)
valid, errors = validate_item_solution(instance, item_bins)
print('FFD bins:', len(ffd_bins))
print('Tabu bins:', result.num_bins, '| exact:', exact['num_bins'])
print('valid:', valid, '| validation errors:', errors)
print('iterations:', result.iterations, '| time:', result.solve_time)"""))

cells.append(nbf.v4.new_markdown_cell("""## ۴. نمونه‌ای که Tabu واقعاً FFD را بهبود می‌دهد

نمونه‌های دشوار مصنوعی benchmark رسمی Falkenauer نیستند؛ آن‌ها فقط از مخلوط وزن‌های بزرگ، متوسط و کوچک الهام گرفته‌اند."""))

cells.append(nbf.v4.new_code_cell("""hard = load_instance(ROOT / 'instances/hard_large/hard_n=300_C=500_pairs=50.txt')
hard_contracted = contract_instance(hard)
hard_lb = lower_bound(hard_contracted.weights, hard_contracted.capacity)
hard_ffd = first_fit_decreasing(hard_contracted.weights, hard_contracted.capacity)
hard_result = tabu_search(hard, max_iter=1500, tabu_tenure=16, seed=42, target=hard_lb)
print('LB:', hard_lb, '| FFD:', len(hard_ffd), '| Tabu:', hard_result.num_bins)
print('improvement:', len(hard_ffd) - hard_result.num_bins, 'bin | valid:', hard_result.valid)
plt.step(range(len(hard_result.history)), hard_result.history, where='post')
plt.axhline(hard_lb, color='red', linestyle='--', label='Lower bound')
plt.xlabel('Iteration'); plt.ylabel('Current bins'); plt.legend(); plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""## ۵. جدول کامل ۱۷ نمونه

ستون `reference_type` مشخص می‌کند مقدار مرجع optimum مدل دقیق است یا فقط lower bound."""))

cells.append(nbf.v4.new_code_cell("""results = pd.read_csv(ROOT / 'results/results_final.csv')
columns = ['instance', 'n', 'instance_type', 'ffd_bins', 'tabu_bins',
           'reference_value', 'reference_type', 'gap_to_reference_pct', 'valid']
display(results[columns])
assert len(results) == 17 and results['valid'].all()"""))

cells.append(nbf.v4.new_markdown_cell("""## ۶. تحلیل پایداری با ۳۰ شروع تصادفی

برای شش نمونه متوسط، جواب اولیه و جواب نهایی در ۳۰ seed مستقل ثبت شده‌اند."""))

cells.append(nbf.v4.new_code_cell("""robustness = pd.read_csv(ROOT / 'results/robustness_summary.csv')
display(robustness[['instance', 'runs', 'initial_mean', 'initial_std',
                    'final_mean', 'final_std', 'best', 'mean_time_sec']])
assert (robustness['runs'] == 30).all()"""))

cells.append(nbf.v4.new_markdown_cell("""## ۷. شکل‌های تولیدشده از داده واقعی"""))
cells.append(nbf.v4.new_code_cell("""display(Image(filename=str(ROOT / 'report/figures/fig1_convergence.png'), width=800))
display(Image(filename=str(ROOT / 'report/figures/fig4_boxplot.png'), width=800))"""))

cells.append(nbf.v4.new_markdown_cell("""## ۸. آزمون‌های خودکار و بازتولید

فرمان‌های زیر بدون تغییر فایل‌های منبع اجرا می‌شوند. برای بازتولید کامل:

```bash
python3 run_experiments_final.py
python3 run_robustness.py
python3 generate_report_artifacts.py
python3 -m unittest discover -s tests -v
```"""))

cells.append(nbf.v4.new_code_cell("""import subprocess, sys
completed = subprocess.run(
    [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
    cwd=ROOT, text=True, capture_output=True, check=True,
)
print(completed.stderr)
print('All automated tests passed.')"""))

nb["cells"] = cells
nbf.write(nb, root / "BinPacking_Cohesive.ipynb")
print("Notebook source rebuilt.")
