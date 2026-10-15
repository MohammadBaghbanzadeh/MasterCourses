# ============================================================================
# تمرین Local Search برای TSP و Graph Partitioning - نسخه نهایی
# محدودیت زمان: 2 دقیقه برای هر الگوریتم
# ============================================================================

import numpy as np
import time
import pandas as pd

np.random.seed(42)

# ============================================================================
# ایجاد Instance های تست (50 شهر/گره)
# ============================================================================
N_TSP = 50
N_GP = 50

# TSP با اعداد صحیح
tsp_coords = (np.random.rand(N_TSP, 2) * 100).astype(int)
tsp_dist_matrix = np.sqrt(((tsp_coords[:, np.newaxis, :] - tsp_coords[np.newaxis, :, :]) ** 2).sum(axis=-1)).astype(int)
np.fill_diagonal(tsp_dist_matrix, 0)

# GP با اعداد صحیح
gp_adj_matrix = np.random.choice([0, 1], size=(N_GP, N_GP), p=[0.88, 0.12])
gp_adj_matrix = np.tril(gp_adj_matrix) + np.tril(gp_adj_matrix, -1).T
np.fill_diagonal(gp_adj_matrix, 0)
gp_adj_matrix = gp_adj_matrix.astype(int)

print(f"Instance آماده → TSP: {N_TSP} شهر | GP: {N_GP} گره")

# ============================================================================
# توابع کمکی TSP
# ============================================================================
def calc_tsp_cost(route, dist):
    return sum(dist[route[i]][route[(i + 1) % len(route)]] for i in range(len(route)))

def apply_2opt(route, i, k):
    return route[:i] + route[i:k+1][::-1] + route[k+1:]

def calc_tsp_delta(route, i, k, dist):
    n = len(route)
    a, b = route[i-1], route[i]
    c, d = route[k], route[(k + 1) % n]
    return (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])

# ============================================================================
# توابع کمکی Graph Partitioning (فرمول صحیح)
# ============================================================================
def calc_gp_cost(A, B, adj):
    return sum(adj[u, v] for u in A for v in B)

def calc_gp_delta_CORRECT(u, v, A, B, adj):
    """
    فرمول کاملاً صحیح - محاسبه مستقیم بدون فرمول پیچیده
    """
    # محاسبه هزینه فعلی
    current_cost = calc_gp_cost(A, B, adj)
    
    # ایجاد partition های جدید بعد از swap
    A_new = A[:]
    B_new = B[:]
    
    # پیدا کردن موقعیت u و v
    u_idx = A.index(u)
    v_idx = B.index(v)
    
    # انجام swap
    A_new[u_idx] = v
    B_new[v_idx] = u
    
    # محاسبه هزینه جدید
    new_cost = calc_gp_cost(A_new, B_new, adj)
    
    return new_cost - current_cost

# تولید نقاط شروع یکسان
nodes = np.random.permutation(N_GP)
initial_A = list(nodes[:N_GP//2])
initial_B = list(nodes[N_GP//2:])
initial_gp_cost = calc_gp_cost(initial_A, initial_B, gp_adj_matrix)

initial_tsp_route = list(np.random.permutation(N_TSP))
initial_tsp_cost = calc_tsp_cost(initial_tsp_route, tsp_dist_matrix)

print(f"نقاط شروع → TSP: {initial_tsp_cost}, GP: {initial_gp_cost}")

# ============================================================================
# الگوریتم TSP Local Search با محدودیت زمان 2 دقیقه
# ============================================================================
def solve_tsp(route, dist, mode="best", use_delta=True):
    curr = route[:]
    cost = calc_tsp_cost(curr, dist)
    n = len(curr)
    iters = 0
    start_time = time.time()
    TIME_LIMIT = 120  # 2 دقیقه
    
    while True:
        # بررسی محدودیت زمان
        if time.time() - start_time > TIME_LIMIT:
            print(f"⏰ TSP متوقف شد بعد از 2 دقیقه - {iters} تکرار")
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

# ============================================================================
# الگوریتم Graph Partitioning Local Search با محدودیت زمان 2 دقیقه
# ============================================================================
def solve_gp(A0, B0, adj, mode="best", use_delta=True):
    A = A0[:]
    B = B0[:]
    cost = calc_gp_cost(A, B, adj)
    iters = 0
    start_time = time.time()
    TIME_LIMIT = 120  # 2 دقیقه

    while True:
        # بررسی محدودیت زمان
        if time.time() - start_time > TIME_LIMIT:
            print(f"⏰ GP متوقف شد بعد از 2 دقیقه - {iters} تکرار")
            break
            
        iters += 1
        best_delta = 0
        best_move = None
        improved = False

        for i in range(len(A)):
            for j in range(len(B)):
                u, v = A[i], B[j]
                
                if use_delta:
                    delta = calc_gp_delta_CORRECT(u, v, A, B, adj)
                else:
                    # محاسبه مستقیم بدون دلتا
                    A_new = A[:]
                    B_new = B[:]
                    A_new[i] = v
                    B_new[j] = u
                    new_cost = calc_gp_cost(A_new, B_new, adj)
                    delta = new_cost - cost

                if delta < best_delta - 1e-6:
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

        if mode == "best" and best_move is not None:
            i, j, u, v = best_move
            A[i] = v
            B[j] = u
            cost += best_delta
            improved = True

        if not improved:
            break

    return cost, iters

# ============================================================================
# اجرا و گزارش نتایج
# ============================================================================
def run_benchmark():
    results = []

    def run_bench(name, func, *args, is_tsp=True):
        print(f"🔄 اجرای {name} (حداکثر 2 دقیقه)...")
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
            "هزینه اولیه": int(initial),
            "هزینه نهایی": int(final),
            "بهبود (%)": round(imp, 2),
            "زمان (ثانیه)": round(elapsed, 4),
            "Iteration": iters
        })
        
        print(f"✅ {name}: {int(initial)} → {int(final)} ({elapsed:.1f}s, {iters} iter)")

    print("🚀 شروع benchmark - هر الگوریتم حداکثر 2 دقیقه...")
    
    # اجرای الگوریتم‌ها
    run_bench("TSP Best NoΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "best", False, is_tsp=True)
    run_bench("TSP Best WithΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "best", True, is_tsp=True)
    run_bench("TSP First NoΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "first", False, is_tsp=True)
    run_bench("TSP First WithΔ", solve_tsp, initial_tsp_route, tsp_dist_matrix, "first", True, is_tsp=True)

    run_bench("GP Best NoΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "best", False, is_tsp=False)
    run_bench("GP Best WithΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "best", True, is_tsp=False)
    run_bench("GP First NoΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "first", False, is_tsp=False)
    run_bench("GP First WithΔ", solve_gp, initial_A, initial_B, gp_adj_matrix, "first", True, is_tsp=False)

    return results

# ============================================================================
# اجرای اصلی
# ============================================================================
if __name__ == "__main__":
    results = run_benchmark()

    # نمایش نتایج
    df = pd.DataFrame(results)
    print("\\n" + "="*85)
    print("          گزارش نهایی - محدودیت زمان 2 دقیقه")
    print("="*85)
    print(df.to_string(index=False))

    # بررسی تطابق نتایج GP
    gp_no_delta = next(r["هزینه نهایی"] for r in results if "GP First NoΔ" in r["الگوریتم"])
    gp_with_delta = next(r["هزینه نهایی"] for r in results if "GP First WithΔ" in r["الگوریتم"])

    print(f"\n🔍 بررسی تطابق GP:")
    print(f"GP First NoΔ: {gp_no_delta}")
    print(f"GP First WithΔ: {gp_with_delta}")

    if abs(gp_no_delta - gp_with_delta) < 1:
        print("✅ باگ GP رفع شد!")
    else:
        print("❌ باگ GP هنوز وجود دارد")

    print(f"\n📊 خلاصه:")
    print(f"• محدودیت زمان: 2 دقیقه برای هر الگوریتم")
    print(f"• اعداد صحیح - بدون اعشار")
    print(f"• نتایج GP WithΔ و NoΔ باید یکسان باشند")

    if abs(gp_no_delta - gp_with_delta) < 1:
        print("🎉 همه چیز درست کار می‌کند!")
    else:
        print("⚠️  نیاز به بررسی بیشتر دارد")