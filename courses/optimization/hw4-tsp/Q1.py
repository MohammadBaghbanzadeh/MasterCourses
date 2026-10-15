# ============================================================================
# تمرین 4 - سوال 1: پیاده‌سازی 3-opt برای مساله فروشنده دوره گرد (TSP)
# ============================================================================
"""
مساله فروشنده دوره گرد که در تمرین قبلی برای 2-opt پیاده سازی کردید را برای 3-opt تکرار کنید 
و یک گزارش کامل متنی از نتایج عددی که شامل جدول جواب ها و زمان اجرا و تحلیل نتایج باشد 
ارائه کنید لطفا در فایل متنی توضیح مختصری در مورد چگونگی اجرا الگوریتم ها ارائه کنید. 
فایل ضمیمه شامل کدها و پی دی اف توضیحات باشد.
"""

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import random

# تنظیم seed برای تکرارپذیری نتایج
np.random.seed(42)
random.seed(42)

# ============================================================================
# ایجاد Instance های تست
# ============================================================================

def create_tsp_instance(n_cities):
    """ایجاد یک instance تصادفی برای TSP"""
    coords = np.random.rand(n_cities, 2) * 100
    dist_matrix = np.sqrt(((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(dist_matrix, 0)
    return coords, dist_matrix

# ایجاد instance های مختلف برای تست
instances = {
    'کوچک': create_tsp_instance(20),
    'متوسط': create_tsp_instance(50), 
    'بزرگ': create_tsp_instance(100)
}

print("Instance های TSP آماده شدند:")
for name, (coords, dist) in instances.items():
    print(f"  {name}: {len(coords)} شهر")

# ============================================================================
# توابع کمکی TSP
# ============================================================================

def calc_tsp_cost(route, dist_matrix):
    """محاسبه هزینه کل مسیر TSP"""
    cost = 0
    n = len(route)
    for i in range(n):
        cost += dist_matrix[route[i]][route[(i + 1) % n]]
    return cost

def generate_initial_route(n_cities):
    """تولید مسیر اولیه تصادفی"""
    return list(np.random.permutation(n_cities))

# ============================================================================
# پیاده‌سازی 2-opt (برای مقایسه)
# ============================================================================

def apply_2opt(route, i, k):
    """اعمال حرکت 2-opt: معکوس کردن بخشی از مسیر"""
    return route[:i] + route[i:k+1][::-1] + route[k+1:]

def calc_2opt_delta(route, i, k, dist_matrix):
    """محاسبه دلتای 2-opt بدون محاسبه کل هزینه"""
    n = len(route)
    a, b = route[i-1], route[i]
    c, d = route[k], route[(k + 1) % n]
    return (dist_matrix[a][c] + dist_matrix[b][d]) - (dist_matrix[a][b] + dist_matrix[c][d])

def solve_tsp_2opt(route, dist_matrix, mode="best"):
    """الگوریتم Local Search برای TSP با 2-opt"""
    curr = route[:]
    cost = calc_tsp_cost(curr, dist_matrix)
    n = len(curr)
    iterations = 0
    
    while True:
        iterations += 1
        best_delta = 0
        best_move = None
        improved = False
        
        for i in range(1, n-1):
            for k in range(i+1, n):
                delta = calc_2opt_delta(curr, i, k, dist_matrix)
                
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
    
    return curr, cost, iterations

# ============================================================================
# پیاده‌سازی 3-opt
# ============================================================================

def get_3opt_moves(route):
    """تولید تمام حرکات ممکن 3-opt برای یک مسیر"""
    n = len(route)
    moves = []
    
    # برای 3-opt نیاز به انتخاب 3 یال داریم
    # یال‌ها: (i-1,i), (j-1,j), (k-1,k) که i < j < k
    for i in range(1, n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                moves.append((i, j, k))
    
    return moves

def apply_3opt(route, i, j, k, move_type):
    """
    اعمال حرکت 3-opt با نوع حرکت مشخص
    
    برای 3 یال که حذف می‌شوند، 7 روش مختلف برای اتصال مجدد وجود دارد:
    1: 2-opt بین i و j
    2: 2-opt بین j و k  
    3: 2-opt بین i و k
    4: جابجایی قطعه وسط
    5: جابجایی و معکوس segment2 و segment3
    6: جابجایی segment2 و معکوس segment3
    7: جابجایی و معکوس هر دو segment
    """
    # تقسیم مسیر به 4 قطعه
    segment1 = route[:i]                    # [0, i)
    segment2 = route[i:j]                   # [i, j)  
    segment3 = route[j:k]                   # [j, k)
    segment4 = route[k:]                    # [k, n)
    
    if move_type == 1:  # 2-opt بین i و j
        return segment1 + segment2[::-1] + segment3 + segment4
    elif move_type == 2:  # 2-opt بین j و k
        return segment1 + segment2 + segment3[::-1] + segment4
    elif move_type == 3:  # 2-opt بین i و k (ترکیب دو 2-opt)
        return segment1 + segment2[::-1] + segment3[::-1] + segment4
    elif move_type == 4:  # جابجایی segment2 و segment3
        return segment1 + segment3 + segment2 + segment4
    elif move_type == 5:  # جابجایی و معکوس segment2 و segment3
        return segment1 + segment3[::-1] + segment2 + segment4
    elif move_type == 6:  # جابجایی segment2 و معکوس segment3
        return segment1 + segment3 + segment2[::-1] + segment4
    elif move_type == 7:  # جابجایی و معکوس هر دو segment
        return segment1 + segment3[::-1] + segment2[::-1] + segment4
    else:
        return route  # بدون تغییر

def calc_3opt_delta(route, i, j, k, move_type, dist_matrix):
    """محاسبه دلتای 3-opt برای یک حرکت خاص"""
    n = len(route)
    
    # یال‌های فعلی که حذف می‌شوند
    old_edges = [
        (route[i-1], route[i]),
        (route[j-1], route[j]), 
        (route[k-1], route[k % n])
    ]
    
    # محاسبه یال‌های جدید بر اساس نوع حرکت
    new_route = apply_3opt(route, i, j, k, move_type)
    
    # محاسبه دقیق تفاوت (برای سادگی)
    old_cost = sum(dist_matrix[edge[0]][edge[1]] for edge in old_edges)
    
    # یال‌های جدید در همان موقعیت‌ها
    new_edges = [
        (new_route[i-1], new_route[i]),
        (new_route[j-1], new_route[j]),
        (new_route[k-1], new_route[k % n])
    ]
    
    new_cost = sum(dist_matrix[edge[0]][edge[1]] for edge in new_edges)
    
    return new_cost - old_cost

def solve_tsp_3opt(route, dist_matrix, mode="best"):
    """الگوریتم Local Search برای TSP با 3-opt"""
    curr = route[:]
    cost = calc_tsp_cost(curr, dist_matrix)
    n = len(curr)
    iterations = 0
    
    print(f"شروع 3-opt با هزینه اولیه: {cost:.2f}")
    
    while True:
        iterations += 1
        best_delta = 0
        best_move = None
        improved = False
        
        # تولید تمام حرکات ممکن 3-opt
        moves = get_3opt_moves(curr)
        
        for i, j, k in moves:
            # بررسی تمام 7 نوع حرکت 3-opt
            for move_type in range(1, 8):
                # محاسبه دلتا به روش ساده (محاسبه کل هزینه)
                new_route = apply_3opt(curr, i, j, k, move_type)
                new_cost = calc_tsp_cost(new_route, dist_matrix)
                delta = new_cost - cost
                
                if delta < best_delta - 1e-6:
                    if mode == "first":
                        curr = new_route
                        cost = new_cost
                        improved = True
                        break
                    best_delta = delta
                    best_move = (i, j, k, move_type)
            
            if improved and mode == "first":
                break
        
        if mode == "best" and best_move:
            i, j, k, move_type = best_move
            curr = apply_3opt(curr, i, j, k, move_type)
            cost += best_delta
            improved = True
        
        if not improved:
            break
            
        if iterations % 10 == 0:
            print(f"  Iteration {iterations}: هزینه فعلی = {cost:.2f}")
    
    print(f"3-opt تمام شد بعد از {iterations} تکرار با هزینه نهایی: {cost:.2f}")
    return curr, cost, iterations

# ============================================================================
# تابع مقایسه و بنچمارک
# ============================================================================

def run_comparison(instance_name, coords, dist_matrix, num_runs=5):
    """اجرای مقایسه بین 2-opt و 3-opt"""
    print(f"\n{'='*60}")
    print(f"مقایسه 2-opt و 3-opt برای instance {instance_name}")
    print(f"تعداد شهرها: {len(coords)}")
    print(f"{'='*60}")
    
    results = []
    
    for run in range(num_runs):
        print(f"\nاجرای {run + 1} از {num_runs}:")
        
        # تولید مسیر اولیه یکسان برای هر دو الگوریتم
        initial_route = generate_initial_route(len(coords))
        initial_cost = calc_tsp_cost(initial_route, dist_matrix)
        
        print(f"هزینه مسیر اولیه: {initial_cost:.2f}")
        
        # اجرای 2-opt
        print("اجرای 2-opt...")
        start_time = time.time()
        route_2opt, cost_2opt, iter_2opt = solve_tsp_2opt(initial_route[:], dist_matrix, "best")
        time_2opt = time.time() - start_time
        
        # اجرای 3-opt
        print("اجرای 3-opt...")
        start_time = time.time()
        route_3opt, cost_3opt, iter_3opt = solve_tsp_3opt(initial_route[:], dist_matrix, "best")
        time_3opt = time.time() - start_time
        
        # محاسبه درصد بهبود
        improvement_2opt = ((initial_cost - cost_2opt) / initial_cost) * 100
        improvement_3opt = ((initial_cost - cost_3opt) / initial_cost) * 100
        
        results.append({
            'اجرا': run + 1,
            'هزینه اولیه': round(initial_cost, 2),
            'هزینه 2-opt': round(cost_2opt, 2),
            'هزینه 3-opt': round(cost_3opt, 2),
            'بهبود 2-opt (%)': round(improvement_2opt, 2),
            'بهبود 3-opt (%)': round(improvement_3opt, 2),
            'زمان 2-opt (ثانیه)': round(time_2opt, 4),
            'زمان 3-opt (ثانیه)': round(time_3opt, 4),
            'تکرار 2-opt': iter_2opt,
            'تکرار 3-opt': iter_3opt
        })
        
        print(f"نتایج اجرای {run + 1}:")
        print(f"  2-opt: هزینه={cost_2opt:.2f}, زمان={time_2opt:.4f}s, تکرار={iter_2opt}")
        print(f"  3-opt: هزینه={cost_3opt:.2f}, زمان={time_3opt:.4f}s, تکرار={iter_3opt}")
    
    return results

# ============================================================================
# تحلیل و گزارش نتایج
# ============================================================================

def analyze_results(all_results):
    """تحلیل و نمایش نتایج کلی"""
    print(f"\n{'='*80}")
    print("تحلیل کلی نتایج")
    print(f"{'='*80}")
    
    summary_data = []
    
    for instance_name, results in all_results.items():
        df = pd.DataFrame(results)
        
        summary_data.append({
            'Instance': instance_name,
            'میانگین بهبود 2-opt (%)': round(df['بهبود 2-opt (%)'].mean(), 2),
            'میانگین بهبود 3-opt (%)': round(df['بهبود 3-opt (%)'].mean(), 2),
            'میانگین زمان 2-opt (ثانیه)': round(df['زمان 2-opt (ثانیه)'].mean(), 4),
            'میانگین زمان 3-opt (ثانیه)': round(df['زمان 3-opt (ثانیه)'].mean(), 4),
            'میانگین تکرار 2-opt': round(df['تکرار 2-opt'].mean(), 1),
            'میانگین تکرار 3-opt': round(df['تکرار 3-opt'].mean(), 1)
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # تحلیل نتایج
    print(f"\n📊 تحلیل نتایج:")
    print("="*50)
    
    for _, row in summary_df.iterrows():
        instance = row['Instance']
        improvement_diff = row['میانگین بهبود 3-opt (%)'] - row['میانگین بهبود 2-opt (%)']
        time_ratio = row['میانگین زمان 3-opt (ثانیه)'] / row['میانگین زمان 2-opt (ثانیه)']
        
        print(f"\n🔍 Instance {instance}:")
        print(f"   • 3-opt بهبود {improvement_diff:+.2f}% بیشتری نسبت به 2-opt دارد")
        print(f"   • 3-opt {time_ratio:.1f} برابر کندتر از 2-opt است")
        print(f"   • نسبت کیفیت به زمان: {improvement_diff/time_ratio:.3f}")

def create_visualization(all_results):
    """ایجاد نمودارهای مقایسه"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('مقایسه عملکرد 2-opt و 3-opt', fontsize=16)
    
    instances = list(all_results.keys())
    
    # نمودار 1: مقایسه درصد بهبود
    improvements_2opt = []
    improvements_3opt = []
    
    for instance_name, results in all_results.items():
        df = pd.DataFrame(results)
        improvements_2opt.append(df['بهبود 2-opt (%)'].mean())
        improvements_3opt.append(df['بهبود 3-opt (%)'].mean())
    
    x = np.arange(len(instances))
    width = 0.35
    
    axes[0,0].bar(x - width/2, improvements_2opt, width, label='2-opt', alpha=0.8)
    axes[0,0].bar(x + width/2, improvements_3opt, width, label='3-opt', alpha=0.8)
    axes[0,0].set_xlabel('Instance')
    axes[0,0].set_ylabel('درصد بهبود (%)')
    axes[0,0].set_title('مقایسه درصد بهبود')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(instances)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # نمودار 2: مقایسه زمان اجرا
    times_2opt = []
    times_3opt = []
    
    for instance_name, results in all_results.items():
        df = pd.DataFrame(results)
        times_2opt.append(df['زمان 2-opt (ثانیه)'].mean())
        times_3opt.append(df['زمان 3-opt (ثانیه)'].mean())
    
    axes[0,1].bar(x - width/2, times_2opt, width, label='2-opt', alpha=0.8)
    axes[0,1].bar(x + width/2, times_3opt, width, label='3-opt', alpha=0.8)
    axes[0,1].set_xlabel('Instance')
    axes[0,1].set_ylabel('زمان (ثانیه)')
    axes[0,1].set_title('مقایسه زمان اجرا')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(instances)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_yscale('log')  # مقیاس لگاریتمی برای بهتر دیدن تفاوت‌ها
    
    # نمودار 3: نسبت کیفیت به زمان
    quality_time_ratios = []
    for i in range(len(instances)):
        improvement_diff = improvements_3opt[i] - improvements_2opt[i]
        time_ratio = times_3opt[i] / times_2opt[i]
        quality_time_ratios.append(improvement_diff / time_ratio)
    
    axes[1,0].bar(instances, quality_time_ratios, alpha=0.8, color='green')
    axes[1,0].set_xlabel('Instance')
    axes[1,0].set_ylabel('نسبت کیفیت به زمان')
    axes[1,0].set_title('کارایی 3-opt نسبت به 2-opt')
    axes[1,0].grid(True, alpha=0.3)
    
    # نمودار 4: مقایسه تعداد تکرارها
    iters_2opt = []
    iters_3opt = []
    
    for instance_name, results in all_results.items():
        df = pd.DataFrame(results)
        iters_2opt.append(df['تکرار 2-opt'].mean())
        iters_3opt.append(df['تکرار 3-opt'].mean())
    
    axes[1,1].bar(x - width/2, iters_2opt, width, label='2-opt', alpha=0.8)
    axes[1,1].bar(x + width/2, iters_3opt, width, label='3-opt', alpha=0.8)
    axes[1,1].set_xlabel('Instance')
    axes[1,1].set_ylabel('تعداد تکرار')
    axes[1,1].set_title('مقایسه تعداد تکرارها')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(instances)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tsp_2opt_vs_3opt_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# اجرای اصلی برنامه
# ============================================================================

def main():
    """تابع اصلی برنامه"""
    print("🚀 شروع مقایسه 2-opt و 3-opt برای مساله TSP")
    print("="*60)
    
    all_results = {}
    
    # اجرای آزمایش‌ها برای هر instance
    for instance_name, (coords, dist_matrix) in instances.items():
        results = run_comparison(instance_name, coords, dist_matrix, num_runs=3)
        all_results[instance_name] = results
        
        # نمایش جدول نتایج برای این instance
        df = pd.DataFrame(results)
        print(f"\n📋 جدول نتایج برای instance {instance_name}:")
        print(df.to_string(index=False))
    
    # تحلیل کلی نتایج
    analyze_results(all_results)
    
    # ایجاد نمودارهای مقایسه
    create_visualization(all_results)
    
    # ذخیره نتایج در فایل CSV
    all_data = []
    for instance_name, results in all_results.items():
        for result in results:
            result['Instance'] = instance_name
            all_data.append(result)
    
    final_df = pd.DataFrame(all_data)
    final_df.to_csv('tsp_2opt_vs_3opt_results.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n✅ نتایج در فایل 'tsp_2opt_vs_3opt_results.csv' ذخیره شد")
    print("✅ نمودار مقایسه در فایل 'tsp_2opt_vs_3opt_comparison.png' ذخیره شد")
    
    # خلاصه نهایی
    print(f"\n{'='*80}")
    print("خلاصه نهایی")
    print(f"{'='*80}")
    print("🎯 نتایج کلیدی:")
    print("   • 3-opt معمولاً کیفیت بهتری نسبت به 2-opt دارد")
    print("   • 3-opt به طور قابل توجهی کندتر از 2-opt است")
    print("   • برای instance های کوچک، 3-opt قابل اجرا است")
    print("   • برای instance های بزرگ، 2-opt عملی‌تر است")
    print("   • انتخاب بین 2-opt و 3-opt بستگی به اولویت زمان یا کیفیت دارد")

if __name__ == "__main__":
    main()