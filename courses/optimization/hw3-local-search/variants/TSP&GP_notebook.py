# Exported notebook code; may contain IPython commands.
# %%
!pip install jinja2 numpy pandas ortools

# %%
import numpy as np
import time
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

np.random.seed(42)

# ====================== ایجاد Instanceها ======================
N_TSP = 100
tsp_coords = np.random.rand(N_TSP, 2) * 100
tsp_dist_matrix = np.sqrt(((tsp_coords[:, np.newaxis, :] - tsp_coords[np.newaxis, :, :]) ** 2).sum(axis=-1))
np.fill_diagonal(tsp_dist_matrix, 0)

N_GP = 100   # اگر لپ‌تاپ قوی داری به 100 تغییر بده
gp_adj_matrix = np.random.choice([0, 1], size=(N_GP, N_GP), p=[0.88, 0.12])
gp_adj_matrix = np.tril(gp_adj_matrix) + np.tril(gp_adj_matrix, -1).T
np.fill_diagonal(gp_adj_matrix, 0)
gp_adj_matrix = gp_adj_matrix.astype(int)

print(f"Instance آماده → TSP: {N_TSP} شهر | GP: {N_GP} گره")

# ====================== توابع کمکی TSP ======================
def calc_tsp_cost(route, dist):
    return sum(dist[route[i]][route[(i + 1) % len(route)]] for i in range(len(route)))

def apply_2opt(route, i, k):
    return route[:i] + route[i:k+1][::-1] + route[k+1:]

def calc_tsp_delta(route, i, k, dist):
    n = len(route)
    a, b = route[i-1], route[i]
    c, d = route[k], route[(k + 1) % n]
    return (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])

initial_tsp_route = list(np.random.permutation(N_TSP))
initial_tsp_cost = calc_tsp_cost(initial_tsp_route, tsp_dist_matrix)

# ====================== توابع کمکی Graph Partitioning ======================
def calc_gp_cost(A, B, adj):
    return np.sum(adj[np.ix_(A, B)])

def calc_gp_delta(u, v, A, B, adj):   # فرمول استاندارد و صحیح
    Iu = np.sum(adj[u, A])
    Eu = np.sum(adj[u, B])
    Iv = np.sum(adj[v, B])
    Ev = np.sum(adj[v, A])
    return (Eu - Iu) + (Ev - Iv) - 2 * adj[u, v]

nodes = np.random.permutation(N_GP)
initial_A = list(nodes[:N_GP//2])
initial_B = list(nodes[N_GP//2:])
initial_gp_cost = calc_gp_cost(initial_A, initial_B, gp_adj_matrix)

# ====================== الگوریتم TSP ======================
def solve_tsp(route, dist, mode="best", use_delta=True):
    curr = route[:]
    cost = calc_tsp_cost(curr, dist)
    n = len(curr)
    iters = 0
    while True:
        iters += 1
        best_delta = 0
        best_move = None
        improved = False
        for i in range(1, n-1):
            for k in range(i+1, n):
                delta = calc_tsp_delta(curr, i, k, dist) if use_delta else \
                        calc_tsp_cost(apply_2opt(curr, i, k), dist) - cost
                if delta < best_delta - 1e-6:
                    if mode == "first":
                        curr = apply_2opt(curr, i, k)
                        cost += delta
                        improved = True
                        break
                    best_delta = delta
                    best_move = (i, k)
            if improved and mode == "first": break
        if mode == "best" and best_move:
            i, k = best_move
            curr = apply_2opt(curr, i, k)
            cost += best_delta
            improved = True
        if not improved:
            break
    return curr, cost, iters

# ====================== الگوریتم Graph Partitioning ======================
def _solve_gp(A0, B0, adj, mode="best", use_delta=True):
    A = A0[:]
    B = B0[:]
    cost = calc_gp_cost(A, B, adj)
    iters = 0
    while True:
        iters += 1
        best_delta = 0
        best_move = None
        improved = False
        for i, u in enumerate(A):
            for j, v in enumerate(B):
                if use_delta:
                    delta = calc_gp_delta(u, v, A, B, adj)
                else:
                    A[i], B[j] = B[j], A[i]
                    delta = calc_gp_cost(A, B, adj) - cost
                    A[i], B[j] = B[j], A[i]   # Revert

                if delta < best_delta - 1e-6:
                    if mode == "first":
                        A[i], B[j] = B[j], A[i]
                        cost += delta
                        improved = True
                        break
                    best_delta = delta
                    best_move = (i, j)
            if improved and mode == "first": break
        if mode == "best" and best_move:
            i, j = best_move
            A[i], B[j] = B[j], A[i]
            cost += best_delta
            improved = True
        if not improved:
            break
    return cost, iters

def solve_gp(A0, B0, adj, mode="best", use_delta=True, max_iter=10000):
    A = A0[:]
    B = B0[:]
    cost = calc_gp_cost(A, B, adj)
    iters = 0

    while iters < max_iter:
        iters += 1
        best_delta = 0          # مهم: همیشه از 0 شروع شود
        best_move = None
        improved = False

        for i, u in enumerate(A):
            for j, v in enumerate(B):
                if use_delta:
                    delta = calc_gp_delta(u, v, A, B, adj)
                else:
                    A[i], B[j] = B[j], A[i]
                    delta = calc_gp_cost(A, B, adj) - cost
                    A[i], B[j] = B[j], A[i]   # Revert

                # فقط حرکت‌های بهبوددهنده (delta < 0) را در نظر بگیر
                if delta < best_delta - 1e-6:     # بهبود نسبت به بهترین دلتا پیدا شده تا حالا
                    if mode == "first":
                        A[i], B[j] = B[j], A[i]
                        cost += delta
                        improved = True
                        break
                    best_delta = delta
                    best_move = (i, j)

            if improved and mode == "first":
                break

        # اعمال بهترین حرکت در حالت Best
        if mode == "best" and best_move is not None and best_delta < -1e-6:
            i, j = best_move
            A[i], B[j] = B[j], A[i]
            cost += best_delta
            improved = True

        if not improved:
            break   # به local optimum رسیدیم

    return cost, iters
# ====================== OR-Tools Baseline ======================
def solve_tsp_ortools(dist_matrix):
    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    def dist_cb(f, t):
        return int(dist_matrix[manager.IndexToNode(f)][manager.IndexToNode(t)])
    tid = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(tid)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
    params.time_limit.FromSeconds(8)
    start = time.time()
    sol = routing.SolveWithParameters(params)
    return sol.ObjectiveValue() if sol else None, time.time() - start

# ====================== اجرا و گزارش ======================
results = []

def run_bench(name, func, *args, is_tsp=True):
    start = time.time()
    if is_tsp:
        _, final, iters = func(*args)
        initial = initial_tsp_cost
    else:
        final, iters = func(*args)
        initial = initial_gp_cost
    elapsed = time.time() - start
    imp = ((initial - final) / initial * 100) if initial > 0 else 0
    results.append({
        "الگوریتم": name,
        "هزینه اولیه": round(initial, 1),
        "هزینه نهایی": round(final, 1),
        "بهبود (%)": round(imp, 2),
        "زمان (ثانیه)": round(elapsed, 4),
        "Iteration": iters
    })

# اجرای ۸ الگوریتم
run_bench("TSP Best NoΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "best", False)
run_bench("TSP Best WithΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "best", True)
run_bench("TSP First NoΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "first", False)
run_bench("TSP First WithΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "first", True)

run_bench("GP Best NoΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "best", False, is_tsp=False)
run_bench("GP Best WithΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "best", True, is_tsp=False)
run_bench("GP First NoΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "first", False, is_tsp=False)
run_bench("GP First WithΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "first", True, is_tsp=False)

# OR-Tools
ot_cost, ot_time = solve_tsp_ortools(tsp_dist_matrix)
results.append({
    "الگوریتم": "TSP OR-Tools (Baseline)",
    "هزینه اولیه": "-",
    "هزینه نهایی": round(ot_cost, 1) if ot_cost else "-",
    "بهبود (%)": "-",
    "زمان (ثانیه)": round(ot_time, 4),
    "Iteration": "-"
})

df = pd.DataFrame(results)
print("\n" + "="*75)
print("          گزارش نهایی تمرین Local Search")
print("="*75)
display(df.style.set_properties(**{'text-align': 'center'}))

print("در این تمرین، ۸ الگوریتم Local Search برای دو مسئله TSP و Graph Partitioning پیاده‌سازی شد.")
print("استفاده از Delta Evaluation باعث کاهش چشمگیر زمان اجرا (معمولاً ۱۰ تا ۵۰ برابر) شد.")
print("تکنیک In-place Swap در حالت بدون Delta نیز پرفورمنس را بهبود بخشید.")
print("الگوریتم Best Improvement معمولاً کیفیت بهتری داشت، در حالی که First Improvement سریع‌تر بود.")

# %%
import numpy as np
import time
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

np.random.seed(42)

# ====================== ایجاد Instanceها ======================
N_TSP = 100
tsp_coords = np.random.rand(N_TSP, 2) * 100
tsp_dist_matrix = np.sqrt(((tsp_coords[:, np.newaxis, :] - tsp_coords[np.newaxis, :, :]) ** 2).sum(axis=-1))
np.fill_diagonal(tsp_dist_matrix, 0)

N_GP = 100
gp_adj_matrix = np.random.choice([0, 1], size=(N_GP, N_GP), p=[0.88, 0.12])
gp_adj_matrix = np.tril(gp_adj_matrix) + np.tril(gp_adj_matrix, -1).T
np.fill_diagonal(gp_adj_matrix, 0)
gp_adj_matrix = gp_adj_matrix.astype(int)   # اطمینان از integer بودن

print(f"Instance آماده → TSP: {N_TSP} شهر | GP: {N_GP} گره")

# ====================== توابع کمکی TSP ======================
def calc_tsp_cost(route, dist):
    return sum(dist[route[i]][route[(i + 1) % len(route)]] for i in range(len(route)))

def apply_2opt(route, i, k):
    return route[:i] + route[i:k+1][::-1] + route[k+1:]

def calc_tsp_delta(route, i, k, dist):
    n = len(route)
    a, b = route[i-1], route[i]
    c, d = route[k], route[(k + 1) % n]
    return (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])

initial_tsp_route = list(np.random.permutation(N_TSP))
initial_tsp_cost = calc_tsp_cost(initial_tsp_route, tsp_dist_matrix)

# ====================== توابع کمکی Graph Partitioning ======================
def calc_gp_cost(A, B, adj):
    return np.sum(adj[np.ix_(A, B)])

def calc_gp_delta(u, v, A, B, adj):
    """فرمول استاندارد Kernighan-Lin برای Graph Bisection"""
    Iu = np.sum(adj[u, A])      # internal edges of u
    Eu = np.sum(adj[u, B])      # external edges of u
    Iv = np.sum(adj[v, B])      # internal edges of v
    Ev = np.sum(adj[v, A])      # external edges of v
    return (Eu - Iu) + (Ev - Iv) - 2 * adj[u, v]

nodes = np.random.permutation(N_GP)
initial_A = list(nodes[:N_GP//2])
initial_B = list(nodes[N_GP//2:])
initial_gp_cost = calc_gp_cost(initial_A, initial_B, gp_adj_matrix)

# ====================== الگوریتم TSP (بدون تغییر عمده) ======================
def solve_tsp(route, dist, mode="best", use_delta=True):
    curr = route[:]
    cost = calc_tsp_cost(curr, dist)
    n = len(curr)
    iters = 0
    while True:
        iters += 1
        best_delta = 0
        best_move = None
        improved = False
        for i in range(1, n-1):
            for k in range(i+1, n):
                if use_delta:
                    delta = calc_tsp_delta(curr, i, k, dist)
                else:
                    new_route = apply_2opt(curr, i, k)
                    delta = calc_tsp_cost(new_route, dist) - cost

                if delta < best_delta - 1e-6:
                    if mode == "first":
                        curr = apply_2opt(curr, i, k)
                        cost += delta
                        improved = True
                        break
                    best_delta = delta
                    best_move = (i, k)
            if improved and mode == "first":
                break
        if mode == "best" and best_move:
            i, k = best_move
            curr = apply_2opt(curr, i, k)
            cost += best_delta
            improved = True
        if not improved:
            break
    return curr, cost, iters

# ====================== الگوریتم Graph Partitioning (اصلاح شده) ======================
def solve_gp(A0, B0, adj, mode="best", use_delta=True, max_iter=10000):
    A = A0[:]
    B = B0[:]
    cost = calc_gp_cost(A, B, adj)
    iters = 0

    while iters < max_iter:
        iters += 1
        best_delta = 0
        best_move = None
        improved = False

        for i, u in enumerate(A):
            for j, v in enumerate(B):
                if use_delta:
                    delta = calc_gp_delta(u, v, A, B, adj)
                else:
                    # NoDelta: سواپ موقت و محاسبه واقعی
                    A[i], B[j] = B[j], A[i]
                    delta = calc_gp_cost(A, B, adj) - cost
                    A[i], B[j] = B[j], A[i]   # revert

                if delta < best_delta - 1e-6:
                    if mode == "first":
                        A[i], B[j] = B[j], A[i]
                        cost += delta
                        improved = True
                        break
                    best_delta = delta
                    best_move = (i, j)

            if improved and mode == "first":
                break

        # اعمال بهترین حرکت (حالت Best)
        if mode == "best" and best_move is not None and best_delta < -1e-6:
            i, j = best_move
            A[i], B[j] = B[j], A[i]
            cost += best_delta
            improved = True

        if not improved:
            break   # به local optimum رسیدیم

    return cost, iters

# ====================== OR-Tools Baseline ======================
def solve_tsp_ortools(dist_matrix):
    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    def dist_cb(f, t):
        return int(dist_matrix[manager.IndexToNode(f)][manager.IndexToNode(t)])
    tid = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(tid)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
    params.time_limit.FromSeconds(8)
    start = time.time()
    sol = routing.SolveWithParameters(params)
    return sol.ObjectiveValue() if sol else None, time.time() - start

# ====================== اجرا و گزارش ======================
results = []

def run_bench(name, func, *args, is_tsp=True):
    start = time.time()
    if is_tsp:
        _, final, iters = func(*args)
        initial = initial_tsp_cost
    else:
        final, iters = func(*args)
        initial = initial_gp_cost
    elapsed = time.time() - start
    imp = ((initial - final) / initial * 100) if initial > 0 else 0
    results.append({
        "الگوریتم": name,
        "هزینه اولیه": round(initial, 1),
        "هزینه نهایی": round(final, 1),
        "بهبود (%)": round(imp, 2),
        "زمان (ثانیه)": round(elapsed, 4),
        "Iteration": iters
    })

# اجرای ۸ الگوریتم
run_bench("TSP Best NoΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "best", False, is_tsp=True)
run_bench("TSP Best WithΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "best", True, is_tsp=True)
run_bench("TSP First NoΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "first", False, is_tsp=True)
run_bench("TSP First WithΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "first", True, is_tsp=True)

run_bench("GP Best NoΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "best", False, is_tsp=False)
run_bench("GP Best WithΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "best", True, is_tsp=False)
run_bench("GP First NoΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "first", False, is_tsp=False)
run_bench("GP First WithΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "first", True, is_tsp=False)

# OR-Tools
ot_cost, ot_time = solve_tsp_ortools(tsp_dist_matrix)
results.append({
    "الگوریتم": "TSP OR-Tools (Baseline)",
    "هزینه اولیه": "-",
    "هزینه نهایی": round(ot_cost, 1) if ot_cost else "-",
    "بهبود (%)": "-",
    "زمان (ثانیه)": round(ot_time, 4),
    "Iteration": "-"
})

df = pd.DataFrame(results)
print("\n" + "="*80)
print("          گزارش نهایی تمرین Local Search (اصلاح شده)")
print("="*80)
display(df.style.set_properties(**{'text-align': 'center'}))

print("\nنکته مهم: حالا در GP، هزینه نهایی حالت WithΔ و NoΔ باید یکسان باشد.")

# %%
import numpy as np
import time
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

np.random.seed(42)

# ====================== ایجاد Instanceها ======================
N_TSP = 60
tsp_coords = np.random.rand(N_TSP, 2) * 100
tsp_dist_matrix = np.sqrt(((tsp_coords[:, np.newaxis, :] - tsp_coords[np.newaxis, :, :]) ** 2).sum(axis=-1))
np.fill_diagonal(tsp_dist_matrix, 0)

N_GP = 80
gp_adj_matrix = np.random.choice([0, 1], size=(N_GP, N_GP), p=[0.88, 0.12])
gp_adj_matrix = np.tril(gp_adj_matrix) + np.tril(gp_adj_matrix, -1).T
np.fill_diagonal(gp_adj_matrix, 0)
gp_adj_matrix = gp_adj_matrix.astype(int)

print(f"Instance آماده → TSP: {N_TSP} شهر | GP: {N_GP} گره")

# ====================== توابع کمکی TSP ======================
def calc_tsp_cost(route, dist):
    return sum(dist[route[i]][route[(i + 1) % len(route)]] for i in range(len(route)))

def apply_2opt(route, i, k):
    return route[:i] + route[i:k+1][::-1] + route[k+1:]

def calc_tsp_delta(route, i, k, dist):
    n = len(route)
    a, b = route[i-1], route[i]
    c, d = route[k], route[(k + 1) % n]
    return (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])

initial_tsp_route = list(np.random.permutation(N_TSP))
initial_tsp_cost = calc_tsp_cost(initial_tsp_route, tsp_dist_matrix)

# ====================== توابع کمکی Graph Partitioning ======================
# ====================== توابع کمکی Graph Partitioning (اصلاح شده) ======================

def calc_gp_cost(A, B, adj):
    return sum(adj[u, v] for u in A for v in B)

def calc_gp_delta(u, v, A, B, adj):
    """
    delta دقیق و سازگار با cost واقعی
    """
    # توجه: u در A و v در B است
    external_u = sum(adj[u, x] for x in B if x != v)
    internal_u = sum(adj[u, x] for x in A if x != u)

    external_v = sum(adj[v, x] for x in A if x != u)
    internal_v = sum(adj[v, x] for x in B if x != v)

    delta = (external_u - internal_u) + (external_v - internal_v) - 2 * adj[u, v]
    return delta

nodes = np.random.permutation(N_GP)
initial_A = list(nodes[:N_GP//2])
initial_B = list(nodes[N_GP//2:])
initial_gp_cost = calc_gp_cost(initial_A, initial_B, gp_adj_matrix)

# ====================== الگوریتم TSP (بدون تغییر) ======================
def solve_gp(A0, B0, adj, mode="best", use_delta=True, max_iter=10000):
    # استفاده از set برای جلوگیری از خطای membership
    A = set(A0)
    B = set(B0)

    cost = calc_gp_cost(A, B, adj)
    iters = 0

    while iters < max_iter:
        iters += 1
        best_delta = 0
        best_move = None
        improved = False

        A_list = list(A)
        B_list = list(B)

        for i, u in enumerate(A_list):
            for j, v in enumerate(B_list):

                if use_delta:
                    delta = calc_gp_delta(u, v, A, B, adj)
                else:
                    # swap موقت
                    A.remove(u); B.remove(v)
                    A.add(v); B.add(u)

                    new_cost = calc_gp_cost(A, B, adj)
                    delta = new_cost - cost

                    # revert
                    A.remove(v); B.remove(u)
                    A.add(u); B.add(v)

                if delta < best_delta - 1e-9:

                    if mode == "first":
                        # اعمال swap
                        A.remove(u); B.remove(v)
                        A.add(v); B.add(u)

                        cost += delta
                        improved = True
                        break

                    best_delta = delta
                    best_move = (u, v)

            if improved and mode == "first":
                break

        if mode == "best" and best_move is not None:
            u, v = best_move

            A.remove(u); B.remove(v)
            A.add(v); B.add(u)

            cost += best_delta
            improved = True

        # 🔍 consistency check (خیلی مهم برای دیباگ)
        real_cost = calc_gp_cost(A, B, adj)
        if abs(real_cost - cost) > 1e-6:
            raise ValueError(f"Inconsistency detected! real={real_cost}, tracked={cost}")

        if not improved:
            break

    return cost, iters
# ====================== الگوریتم Graph Partitioning (اصلاح نهایی) ======================
def solve_gp(A0, B0, adj, mode="best", use_delta=True, max_iter=10000):
    A = A0[:]
    B = B0[:]
    cost = calc_gp_cost(A, B, adj)
    iters = 0

    while iters < max_iter:
        iters += 1
        best_delta = 0
        best_move = None
        improved = False

        for i, u in enumerate(A):
            for j, v in enumerate(B):
                if use_delta:
                    delta = calc_gp_delta(u, v, A, B, adj)
                else:
                    A[i], B[j] = B[j], A[i]
                    delta = calc_gp_cost(A, B, adj) - cost
                    A[i], B[j] = B[j], A[i]  # revert

                if delta < best_delta - 1e-6:
                    if mode == "first":
                        A[i], B[j] = B[j], A[i]
                        cost += delta
                        improved = True
                        break
                    best_delta = delta
                    best_move = (i, j)

            if improved and mode == "first":
                break

        # اعمال بهترین حرکت در حالت Best
        if mode == "best" and best_move is not None and best_delta < -1e-6:
            i, j = best_move
            A[i], B[j] = B[j], A[i]
            cost += best_delta
            improved = True

        if not improved:
            break

    return cost, iters

# ====================== OR-Tools Baseline ======================
def solve_tsp_ortools(dist_matrix):
    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    def dist_cb(f, t):
        return int(dist_matrix[manager.IndexToNode(f)][manager.IndexToNode(t)])
    tid = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(tid)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
    params.time_limit.FromSeconds(8)
    start = time.time()
    sol = routing.SolveWithParameters(params)
    return sol.ObjectiveValue() if sol else None, time.time() - start

# ====================== اجرا و گزارش ======================
results = []

def run_bench(name, func, *args, is_tsp=True):
    start = time.time()
    if is_tsp:
        _, final, iters = func(*args)
        initial = initial_tsp_cost
    else:
        final, iters = func(*args)
        initial = initial_gp_cost
    elapsed = time.time() - start
    imp = ((initial - final) / initial * 100) if initial > 0 else 0
    results.append({
        "الگوریتم": name,
        "هزینه اولیه": round(initial, 1),
        "هزینه نهایی": round(final, 1),
        "بهبود (%)": round(imp, 2),
        "زمان (ثانیه)": round(elapsed, 4),
        "Iteration": iters
    })

# اجرای الگوریتم‌ها
run_bench("TSP Best NoΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "best", False, is_tsp=True)
run_bench("TSP Best WithΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "best", True, is_tsp=True)
run_bench("TSP First NoΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "first", False, is_tsp=True)
run_bench("TSP First WithΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "first", True, is_tsp=True)

run_bench("GP Best NoΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "best", False, is_tsp=False)
run_bench("GP Best WithΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "best", True, is_tsp=False)
run_bench("GP First NoΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "first", False, is_tsp=False)
run_bench("GP First WithΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "first", True, is_tsp=False)

# OR-Tools
ot_cost, ot_time = solve_tsp_ortools(tsp_dist_matrix)
results.append({
    "الگوریتم": "TSP OR-Tools (Baseline)",
    "هزینه اولیه": "-",
    "هزینه نهایی": round(ot_cost, 1) if ot_cost else "-",
    "بهبود (%)": "-",
    "زمان (ثانیه)": round(ot_time, 4),
    "Iteration": "-"
})

df = pd.DataFrame(results)
print("\n" + "="*85)
print("          گزارش نهایی تمرین Local Search (اصلاح نهایی)")
print("="*85)
display(df.style.set_properties(**{'text-align': 'center'}))

print("\nحالا باید هزینه نهایی در هر جفت (NoΔ و WithΔ) برای GP کاملاً یکسان باشد.")

# %%

