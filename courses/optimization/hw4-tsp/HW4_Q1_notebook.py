# Exported notebook code; may contain IPython commands.
# %%
#!/usr/bin/env python3
# ============================================================================
# مقایسه کامل 2-opt و 3-opt برای مسئله فروشنده دوره‌گرد (TSP)
# ============================================================================

import numpy as np
import time
import pandas as pd
from itertools import combinations

np.random.seed(42)

# ============================================================================
# تولید Instance های تست
# ============================================================================
def generate_tsp_instance(n, coord_range=100):
    """تولید یک instance تصادفی TSP"""
    coords = np.random.randint(0, coord_range, size=(n, 2))
    dist_matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = abs(coords[i][0] - coords[j][0]) + \
                                   abs(coords[i][1] - coords[j][1])
    
    return coords, dist_matrix

# ============================================================================
# توابع کمکی TSP
# ============================================================================
def calc_tsp_cost(route, dist):
    """محاسبه هزینه کل مسیر"""
    total = 0
    n = len(route)
    for i in range(n):
        total += dist[route[i]][route[(i + 1) % n]]
    return int(total)

def apply_2opt(route, i, k):
    """اعمال حرکت 2-opt: معکوس کردن بخش [i:k+1]"""
    return route[:i] + route[i:k+1][::-1] + route[k+1:]

def calc_2opt_delta(route, i, k, dist):
    """محاسبه دلتای 2-opt بدون محاسبه کامل هزینه"""
    n = len(route)
    a, b = route[i-1], route[i]
    c, d = route[k], route[(k + 1) % n]
    return int((dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d]))

def apply_3opt(route, i, j, k, case=0):
    """
    اعمال حرکت 3-opt با 7 حالت مختلف
    
    بخش‌ها:
    - A: route[0:i]
    - B: route[i:j]
    - C: route[j:k+1]
    - D: route[k+1:]
    
    حالت‌های مختلف:
    0: A + B + C + D (بدون تغییر)
    1: A + B' + C + D (معکوس B)
    2: A + B + C' + D (معکوس C)
    3: A + C + B + D (جابجایی B و C)
    4: A + B' + C' + D (معکوس B و C)
    5: A + C' + B + D (معکوس C و جابجایی)
    6: A + C + B' + D (جابجایی و معکوس B)
    """
    A = route[:i]
    B = route[i:j]
    C = route[j:k+1]
    D = route[k+1:]
    
    if case == 0:
        return route[:]
    elif case == 1:
        return A + B[::-1] + C + D
    elif case == 2:
        return A + B + C[::-1] + D
    elif case == 3:
        return A + C + B + D
    elif case == 4:
        return A + B[::-1] + C[::-1] + D
    elif case == 5:
        return A + C[::-1] + B + D
    elif case == 6:
        return A + C + B[::-1] + D
    else:
        return route[:]

def calc_3opt_delta(route, i, j, k, case, dist):
    """محاسبه دلتای 3-opt برای یک حالت خاص"""
    current_cost = calc_tsp_cost(route, dist)
    new_route = apply_3opt(route, i, j, k, case)
    new_cost = calc_tsp_cost(new_route, dist)
    return int(new_cost - current_cost), new_route

# ============================================================================
# الگوریتم 2-opt Local Search
# ============================================================================
def solve_tsp_2opt(route, dist, mode="best", use_delta=True, max_time=300):
    """
    الگوریتم Local Search با 2-opt
    
    Args:
        route: مسیر اولیه
        dist: ماتریس فاصله
        mode: "best" یا "first" improvement
        use_delta: استفاده از محاسبه دلتا
        max_time: حداکثر زمان اجرا (ثانیه)
    """
    curr = route[:]
    cost = calc_tsp_cost(curr, dist)
    initial_cost = cost
    n = len(curr)
    iters = 0
    start_time = time.time()
    
    while True:
        if time.time() - start_time > max_time:
            print(f"⏰ 2-opt timeout بعد از {max_time}s")
            break
            
        iters += 1
        best_delta = 0
        best_move = None
        improved = False
        
        for i in range(1, n-1):
            for k in range(i+1, n):
                if use_delta:
                    delta = calc_2opt_delta(curr, i, k, dist)
                else:
                    new_route = apply_2opt(curr, i, k)
                    delta = calc_tsp_cost(new_route, dist) - cost

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
    
    elapsed = time.time() - start_time
    improvement = initial_cost - cost
    
    return {
        'route': curr,
        'cost': int(cost),
        'initial_cost': int(initial_cost),
        'improvement': int(improvement),
        'improvement_pct': (improvement / initial_cost * 100) if initial_cost > 0 else 0,
        'iterations': iters,
        'time': elapsed
    }

# ============================================================================
# الگوریتم 3-opt Local Search
# ============================================================================
def solve_tsp_3opt(route, dist, mode="best", use_delta=True, max_time=300):
    """
    الگوریتم Local Search با 3-opt
    
    Args:
        route: مسیر اولیه
        dist: ماتریس فاصله
        mode: "best" یا "first" improvement
        use_delta: استفاده از محاسبه دلتا
        max_time: حداکثر زمان اجرا (ثانیه)
    """
    curr = route[:]
    cost = calc_tsp_cost(curr, dist)
    initial_cost = cost
    n = len(curr)
    iters = 0
    start_time = time.time()
    
    while True:
        if time.time() - start_time > max_time:
            print(f"⏰ 3-opt timeout بعد از {max_time}s")
            break
            
        iters += 1
        best_delta = 0
        best_move = None
        improved = False
        
        # بررسی تمام ترکیبات ممکن i, j, k
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    # بررسی 7 حالت مختلف 3-opt (حالت 0 تغییری ندارد)
                    for case in range(1, 7):
                        if use_delta:
                            delta, new_route = calc_3opt_delta(curr, i, j, k, case, dist)
                        else:
                            new_route = apply_3opt(curr, i, j, k, case)
                            delta = calc_tsp_cost(new_route, dist) - cost
                        
                        if delta < best_delta:
                            if mode == "first":
                                curr = new_route
                                cost += delta
                                improved = True
                                break
                            best_delta = delta
                            best_move = (i, j, k, case, new_route)
                    
                    if improved and mode == "first":
                        break
                
                if improved and mode == "first":
                    break
            
            if improved and mode == "first":
                break
        
        if mode == "best" and best_move and best_delta < 0:
            i, j, k, case, new_route = best_move
            curr = new_route
            cost += best_delta
            improved = True
        
        if not improved:
            break
    
    elapsed = time.time() - start_time
    improvement = initial_cost - cost
    
    return {
        'route': curr,
        'cost': int(cost),
        'initial_cost': int(initial_cost),
        'improvement': int(improvement),
        'improvement_pct': (improvement / initial_cost * 100) if initial_cost > 0 else 0,
        'iterations': iters,
        'time': elapsed
    }

# ============================================================================
# اجرای آزمایش‌ها
# ============================================================================
def run_experiments():
    """اجرای آزمایش‌های مقایسه‌ای"""
    
    # اندازه‌های مختلف برای تست
    sizes = {
        'کوچک': 15,
        'متوسط': 25,
        'بزرگ': 40
    }
    
    results = []
    
    for size_name, n in sizes.items():
        print(f"\n{'='*80}")
        print(f"آزمایش برای اندازه {size_name} (n={n})")
        print('='*80)
        
        # تولید instance
        coords, dist_matrix = generate_tsp_instance(n)
        initial_route = list(range(n))
        np.random.shuffle(initial_route)
        initial_cost = calc_tsp_cost(initial_route, dist_matrix)
        
        print(f"هزینه اولیه: {initial_cost}")
        
        # تست 2-opt Best
        print("\n🔄 اجرای 2-opt Best Improvement...")
        result_2opt_best = solve_tsp_2opt(initial_route, dist_matrix, "best", True, 300)
        
        # تست 2-opt First
        print("🔄 اجرای 2-opt First Improvement...")
        result_2opt_first = solve_tsp_2opt(initial_route, dist_matrix, "first", True, 300)
        
        # تست 3-opt Best
        print("🔄 اجرای 3-opt Best Improvement...")
        result_3opt_best = solve_tsp_3opt(initial_route, dist_matrix, "best", True, 300)
        
        # تست 3-opt First
        print("🔄 اجرای 3-opt First Improvement...")
        result_3opt_first = solve_tsp_3opt(initial_route, dist_matrix, "first", True, 300)
        
        # ذخیره نتایج
        results.append({
            'اندازه': size_name,
            'n': n,
            'هزینه اولیه': initial_cost,
            '2-opt Best - هزینه': result_2opt_best['cost'],
            '2-opt Best - بهبود%': f"{result_2opt_best['improvement_pct']:.2f}",
            '2-opt Best - زمان': f"{result_2opt_best['time']:.3f}",
            '2-opt Best - تکرار': result_2opt_best['iterations'],
            '2-opt First - هزینه': result_2opt_first['cost'],
            '2-opt First - بهبود%': f"{result_2opt_first['improvement_pct']:.2f}",
            '2-opt First - زمان': f"{result_2opt_first['time']:.3f}",
            '2-opt First - تکرار': result_2opt_first['iterations'],
            '3-opt Best - هزینه': result_3opt_best['cost'],
            '3-opt Best - بهبود%': f"{result_3opt_best['improvement_pct']:.2f}",
            '3-opt Best - زمان': f"{result_3opt_best['time']:.3f}",
            '3-opt Best - تکرار': result_3opt_best['iterations'],
            '3-opt First - هزینه': result_3opt_first['cost'],
            '3-opt First - بهبود%': f"{result_3opt_first['improvement_pct']:.2f}",
            '3-opt First - زمان': f"{result_3opt_first['time']:.3f}",
            '3-opt First - تکرار': result_3opt_first['iterations'],
        })
        
        # نمایش خلاصه
        print(f"\n📊 خلاصه نتایج برای n={n}:")
        print(f"2-opt Best:  {initial_cost} → {result_2opt_best['cost']} "
              f"({result_2opt_best['improvement_pct']:.2f}% بهبود، "
              f"{result_2opt_best['time']:.3f}s)")
        print(f"2-opt First: {initial_cost} → {result_2opt_first['cost']} "
              f"({result_2opt_first['improvement_pct']:.2f}% بهبود، "
              f"{result_2opt_first['time']:.3f}s)")
        print(f"3-opt Best:  {initial_cost} → {result_3opt_best['cost']} "
              f"({result_3opt_best['improvement_pct']:.2f}% بهبود، "
              f"{result_3opt_best['time']:.3f}s)")
        print(f"3-opt First: {initial_cost} → {result_3opt_first['cost']} "
              f"({result_3opt_first['improvement_pct']:.2f}% بهبود، "
              f"{result_3opt_first['time']:.3f}s)")
    
    return results

# ============================================================================
# تولید گزارش نهایی
# ============================================================================
def generate_report(results):
    """تولید گزارش کامل از نتایج"""
    
    print("\n" + "="*100)
    print("گزارش نهایی مقایسه 2-opt و 3-opt برای مسئله TSP")
    print("="*100)
    
    # جدول خلاصه
    df = pd.DataFrame(results)
    print("\n📊 جدول خلاصه نتایج:")
    print(df.to_string(index=False))
    
    # تحلیل نتایج
    print("\n" + "="*100)
    print("تحلیل نتایج")
    print("="*100)
    
    for result in results:
        size = result['اندازه']
        n = result['n']
        
        print(f"\n🔍 تحلیل برای اندازه {size} (n={n}):")
        
        # مقایسه کیفیت
        cost_2opt_best = result['2-opt Best - هزینه']
        cost_3opt_best = result['3-opt Best - هزینه']
        
        if cost_3opt_best < cost_2opt_best:
            improvement = ((cost_2opt_best - cost_3opt_best) / cost_2opt_best * 100)
            print(f"   ✅ 3-opt بهتر: {improvement:.2f}% بهبود نسبت به 2-opt")
        elif cost_3opt_best == cost_2opt_best:
            print(f"   ⚖️  کیفیت یکسان: هر دو به همان جواب رسیدند")
        else:
            print(f"   ⚡ 2-opt بهتر (نادر)")
        
        # مقایسه زمان
        time_2opt = float(result['2-opt Best - زمان'])
        time_3opt = float(result['3-opt Best - زمان'])
        slowdown = time_3opt / time_2opt if time_2opt > 0 else 1
        
        print(f"   ⏱️  زمان اجرا: 3-opt {slowdown:.1f}x کندتر از 2-opt")
        print(f"   🔄 تکرارها: 2-opt={result['2-opt Best - تکرار']}, "
              f"3-opt={result['3-opt Best - تکرار']}")
    
    # نتیجه‌گیری کلی
    print("\n" + "="*100)
    print("نتیجه‌گیری کلی")
    print("="*100)
    print("""
1. کیفیت جواب:
   • 3-opt معمولاً جواب‌های بهتری نسبت به 2-opt پیدا می‌کند
   • برای مسائل کوچک، تفاوت قابل توجه است
   • برای مسائل بزرگ، تفاوت کمتر می‌شود

2. زمان اجرا:
   • 3-opt به طور قابل توجهی کندتر از 2-opt است
   • پیچیدگی زمانی: 2-opt O(n²) vs 3-opt O(n³)
   • برای n>50، 3-opt ممکن است غیرعملی باشد

3. توصیه‌ها:
   • برای مسائل کوچک (n<30): استفاده از 3-opt
   • برای مسائل متوسط (30<n<50): ترکیب 2-opt و 3-opt
   • برای مسائل بزرگ (n>50): استفاده از 2-opt یا روش‌های پیشرفته‌تر

4. استراتژی ترکیبی:
   • ابتدا 2-opt را اجرا کنید (سریع)
   • سپس 3-opt را روی نتیجه اعمال کنید (بهبود نهایی)
   • این رویکرد تعادل خوبی بین سرعت و کیفیت ایجاد می‌کند
    """)
    
    return df

# ============================================================================
# اجرای اصلی
# ============================================================================
if __name__ == "__main__":
    print("🚀 شروع آزمایش‌های مقایسه‌ای 2-opt و 3-opt")
    print("="*100)
    
    # اجرای آزمایش‌ها
    results = run_experiments()
    
    # تولید گزارش
    df = generate_report(results)
    
    # ذخیره نتایج در فایل CSV
    output_file = "TSP_2opt_vs_3opt_Results.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 نتایج در فایل {output_file} ذخیره شد")
    
    print("\n✅ اجرای برنامه با موفقیت به پایان رسید!")

