# ============================================================================
# تمرین Local Search - کاملاً با اعداد صحیح
# ============================================================================

import numpy as np
import time
import pandas as pd

np.random.seed(42)

# ============================================================================
# ایجاد Instance های تست - کاملاً INTEGER
# ============================================================================
N_TSP = 50
N_GP = 50

# TSP: فقط اعداد صحیح
tsp_coords = np.random.randint(0, 100, size=(N_TSP, 2))  # اعداد صحیح 0-99
tsp_dist_matrix = np.zeros((N_TSP, N_TSP), dtype=int)

# محاسبه فاصله منهتن (Manhattan distance) - کاملاً صحیح
for i in range(N_TSP):
    for j in range(N_TSP):
        if i != j:
            tsp_dist_matrix[i][j] = abs(tsp_coords[i][0] - tsp_coords[j][0]) + abs(tsp_coords[i][1] - tsp_coords[j][1])

# GP: فقط 0 و 1
gp_adj_matrix = np.random.choice([0, 1], size=(N_GP, N_GP), p=[0.88, 0.12])
gp_adj_matrix = np.tril(gp_adj_matrix) + np.tril(gp_adj_matrix, -1).T
np.fill_diagonal(gp_adj_matrix, 0)
gp_adj_matrix = gp_adj_matrix.astype(int)

print(f"Instance آماده → TSP: {N_TSP} شهر | GP: {N_GP} گره")
print("همه داده‌ها کاملاً INTEGER هستند")

# ============================================================================
# توابع کمکی TSP - کاملاً INTEGER
# ============================================================================
def calc_tsp_cost(route, dist):
    """محاسبه هزینه TSP - خروجی INTEGER"""
    total = 0
    n = len(route)
    for i in range(n):
        total += dist[route[i]][route[(i + 1) % n]]
    return int(total)

def apply_2opt(route, i, k):
    """اعمال 2-opt"""
    return route[:i] + route[i:k+1][::-1] + route[k+1:]

def calc_tsp_delta(route, i, k, dist):
    """محاسبه دلتا TSP - خروجی INTEGER"""
    n = len(route)
    a, b = route[i-1], route[i]
    c, d = route[k], route[(k + 1) % n]
    delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
    return int(delta)

# ============================================================================
# توابع کمکی Graph Partitioning - کاملاً INTEGER
# ============================================================================
def calc_gp_cost(A, B, adj):
    """محاسبه هزینه GP - خروجی INTEGER"""
    total = 0
    for u in A:
        for v in B:
            total += adj[u, v]
    return int(total)

def calc_gp_delta_SIMPLE(u, v, A, B, adj):
    """
    محاسبه دلتا GP - روش ساده و مطمئن
    فقط محاسبه مستقیم - بدون فرمول پیچیده
    """
    # هزینه فعلی
    old_cost = calc_gp_cost(A, B, adj)
    
    # کپی کردن partition ها
    A_new = A[:]
    B_new = B[:]
    
    # پیدا کردن جای u و v
    u_pos = A_new.index(u)
    v_pos = B_new.index(v)
    
    # swap کردن
    A_new[u_pos] = v
    B_new[v_pos] = u
    
    # هزینه جدید
    new_cost = calc_gp_cost(A_new, B_new, adj)
    
    # دلتا
    delta = new_cost - old_cost
    return int(delta)

# ============================================================================
# نقاط شروع یکسان
# ============================================================================
nodes = list(range(N_GP))
np.random.shuffle(nodes)
initial_A = nodes[:N_GP//2]
initial_B = nodes[N_GP//2:]
initial_gp_cost = calc_gp_cost(initial_A, initial_B, gp_adj_matrix)

initial_tsp_route = list(range(N_TSP))
np.random.shuffle(initial_tsp_route)
initial_tsp_cost = calc_tsp_cost(initial_tsp_route, tsp_dist_matrix)

print(f"نقاط شروع → TSP: {initial_tsp_cost}, GP: {initial_gp_cost}")

# ============================================================================
# الگوریتم TSP - کاملاً INTEGER
# ============================================================================
def solve_tsp(route, dist, mode="best", use_delta=True):
    curr = route[:]
    cost = calc_tsp_cost(curr, dist)
    n = len(curr)
    iters = 0
    start_time = time.time()
    TIME_LIMIT = 120  # 2 دقیقه
    
    while True:
        if time.time() - start_time > TIME_LIMIT:
            print(f"⏰ TSP timeout - {iters} تکرار")
            break
            
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
                    new_cost = calc_tsp_cost(new_route, dist)
                    delta = new_cost - cost

                if delta < best_delta:
                    if mode == "first":
                        curr = apply_2opt(curr, i, k)
                        cost += delta
                        improved = True
                        break
                    best_delta = delta
                    best_move = (i, k)
            
            if improved and mode == "first":
                break
        
        if mode == "best" and best_move and best_delta < 0:
            i, k = best_move
            curr = apply_2opt(curr, i, k)
            cost += best_delta
            improved = True
        
        if not improved:
            break
    
    return curr, int(cost), iters

# ============================================================================
# الگوریتم GP - کاملاً INTEGER
# ============================================================================
def solve_gp(A0, B0, adj, mode="best", use_delta=True):
    A = A0[:]
    B = B0[:]
    cost = calc_gp_cost(A, B, adj)
    iters = 0
    start_time = time.time()
    TIME_LIMIT = 120  # 2 دقیقه

    while True:
        if time.time() - start_time > TIME_LIMIT:
            print(f"⏰ GP timeout - {iters} تکرار")
            break
            
        iters += 1
        best_delta = 0
        best_move = None
        improved = False

        for i in range(len(A)):
            for j in range(len(B)):
                u, v = A[i], B[j]
                
                if use_delta:
                    delta = calc_gp_delta_SIMPLE(u, v, A, B, adj)
                else:
                    # محاسبه مستقیم
                    A_temp = A[:]
                    B_temp = B[:]
                    A_temp[i] = v
                    B_temp[j] = u
                    new_cost = calc_gp_cost(A_temp, B_temp, adj)
                    delta = new_cost - cost

                if delta < best_delta:
                    if mode == "first":
                        A[i] = v
                        B[j] = u
                        cost += delta
                        improved = True
                        break
                    else:
                        best_delta = delta
                        best_move = (i, j, u, v)

            if improved and mode == "first":
                break

        if mode == "best" and best_move is not None and best_delta < 0:
            i, j, u, v = best_move
            A[i] = v
            B[j] = u
            cost += best_delta
            improved = True

        if not improved:
            break

    return int(cost), iters

# ============================================================================
# تست سریع برای اطمینان
# ============================================================================
print("\n🧪 تست سریع تطابق GP:")

# تست با نقاط شروع یکسان
test_A = initial_A[:]
test_B = initial_B[:]

print(f"نقطه شروع: A={test_A[:5]}..., B={test_B[:5]}...")
print(f"هزینه اولیه: {calc_gp_cost(test_A, test_B, gp_adj_matrix)}")

# اجرای کوتاه برای تست
start = time.time()
cost1, iter1 = solve_gp(test_A, test_B, gp_adj_matrix, "first", False)
time1 = time.time() - start

start = time.time()
cost2, iter2 = solve_gp(test_A, test_B, gp_adj_matrix, "first", True)
time2 = time.time() - start

print(f"GP First NoΔ:  {cost1} ({iter1} iter, {time1:.3f}s)")
print(f"GP First WithΔ: {cost2} ({iter2} iter, {time2:.3f}s)")

if cost1 == cost2:
    print("✅ تست موفق - نتایج یکسان!")
else:
    print(f"❌ تست ناموفق - اختلاف: {abs(cost1 - cost2)}")

# ============================================================================
# اجرای کامل
# ============================================================================
def run_full_benchmark():
    results = []

    def run_bench(name, func, *args, is_tsp=True):
        print(f"🔄 {name}...")
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
            "هزینه اولیه": initial,
            "هزینه نهایی": final,
            "بهبود (%)": round(imp, 2),
            "زمان (ثانیه)": round(elapsed, 4),
            "Iteration": iters
        })
        
        print(f"✅ {initial} → {final} ({elapsed:.3f}s)")

    print("\n🚀 اجرای کامل benchmark...")
    
    run_bench("TSP Best NoΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "best", False, is_tsp=True)
    run_bench("TSP Best WithΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "best", True, is_tsp=True)
    run_bench("TSP First NoΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "first", False, is_tsp=True)
    run_bench("TSP First WithΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "first", True, is_tsp=True)

    run_bench("GP Best NoΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "best", False, is_tsp=False)
    run_bench("GP Best WithΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "best", True, is_tsp=False)
    run_bench("GP First NoΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "first", False, is_tsp=False)
    run_bench("GP First WithΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "first", True, is_tsp=False)

    return results

if __name__ == "__main__":
    results = run_full_benchmark()

    # نمایش نتایج
    df = pd.DataFrame(results)
    print("\n" + "="*85)
    print("          گزارش نهایی - کاملاً INTEGER")
    print("="*85)
    print(df.to_string(index=False))

    # بررسی تطابق GP
    gp_no_delta = next(r["هزینه نهایی"] for r in results if "GP First NoΔ" in r["الگوریتم"])
    gp_with_delta = next(r["هزینه نهایی"] for r in results if "GP First WithΔ" in r["الگوریتم"])

    print(f"\n🔍 بررسی نهایی:")
    print(f"GP First NoΔ: {gp_no_delta}")
    print(f"GP First WithΔ: {gp_with_delta}")

    if gp_no_delta == gp_with_delta:
        print("🎉 مشکل حل شد! نتایج کاملاً یکسان!")
    else:
        print(f"❌ هنوز اختلاف دارد: {abs(gp_no_delta - gp_with_delta)}")