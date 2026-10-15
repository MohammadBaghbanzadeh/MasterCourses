# Exported notebook code; may contain IPython commands.
# %%
!pip install -i https://mirror-pypi.runflare.com/simple ortools

# %%
# وارد کردن کتابخانه‌های مورد نیاز
import time
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

# %%
def solve_assignment(n, integer=True):
    """
    Solve the assignment problem with n jobs and n machines.
    
    Parameters:
        n (int): Number of jobs and machines.
        integer (bool): True for IP model, False for LP model.
    
    Returns:
        obj_value (float): Optimal objective value.
        elapsed (float): Solution time in seconds.
        costs (list): The cost matrix used.
    """
    # Generate random cost matrix (integer values between 1 and 100)
    # Use a fixed seed for reproducibility
    np.random.seed(42 + n)
    costs = np.random.randint(1, 101, size=(n, n)).tolist()

    # Create OR-Tools solver (using SCIP for both IP and LP)
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception('SCIP solver is not available.')

    # Define decision variables
    x = {}
    for i in range(n):
        for j in range(n):
            if integer:
                x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')
            else:
                x[i, j] = solver.NumVar(0, 1, f'x_{i}_{j}')

    # Constraints: each job must be assigned to exactly one machine
    for i in range(n):
        solver.Add(solver.Sum([x[i, j] for j in range(n)]) == 1)

    # Each machine must perform exactly one job
    for j in range(n):
        solver.Add(solver.Sum([x[i, j] for i in range(n)]) == 1)

    # Objective function: minimize total cost
    objective = solver.Objective()
    for i in range(n):
        for j in range(n):
            objective.SetCoefficient(x[i, j], costs[i][j])
    objective.SetMinimization()

    # Solve the model and measure execution time
    start_time = time.time()
    status = solver.Solve()
    elapsed = time.time() - start_time

    if status == pywraplp.Solver.OPTIMAL:
        obj_value = objective.Value()
        return obj_value, elapsed, costs
    else:
        return None, elapsed, costs

# %%
sizes = [5, 20, 50, 100]
results = []

for n in sizes:
    # Integer programming (IP) case
    obj_int, t_int, _ = solve_assignment(n, integer=True)
    results.append({
        'n': n,
        'Mode': 'Discrete (IP)',
        'Objective': obj_int,
        'Time (s)': t_int
    })
    
    # Linear programming (LP) case
    obj_lp, t_lp, _ = solve_assignment(n, integer=False)
    results.append({
        'n': n,
        'Mode': 'Continuous (LP)',
        'Objective': obj_lp,
        'Time (s)': t_lp
    })

# Create a DataFrame
df_results = pd.DataFrame(results)
df_results

# %%
# Configure float display format
pd.options.display.float_format = '{:,.4f}'.format

# Display the results table
print("نتایج حل مسئله تخصیص با OR-Tools")
print("-" * 60)
print(df_results.to_string(index=False))
print("-" * 60)

# Save to CSV file
df_results.to_csv('assignment_results.csv', index=False, encoding='utf-8-sig')
print("\nنتایج در فایل assignment_results.csv ذخیره شد.")
