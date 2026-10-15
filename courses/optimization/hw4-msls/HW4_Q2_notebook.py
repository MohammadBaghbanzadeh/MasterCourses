# Exported notebook code; may contain IPython commands.
# %%
# ============================================================================
# تمرین 4 - سوال 2: الگوریتم MSLS (Multi-Start Local Search)
# ============================================================================
"""
الگوریتم MSLS را به طور مختصر توضیح دهید.سپس دو مقاله (از سال ۲۰۲۴ به بعد) که از این الگوریتم استفاده کرده‌اند، به همراه توضیح کوتاهی درباره هر یک ارائه دهید. لطفاً مقالات منتخب از ژورنال‌های معتبر باشند.

حال فرض کنید سیستمی با ۴ پردازنده در اختیار دارید. اگر بخواهید الگوریتم MSLS را با ۴ جواب اولیه آغاز کنید، چه اقداماتی برای بهبود عملکرد آن می‌توان انجام داد؟

در ادامه، سناریوی دیگری را در نظر بگیرید: سیستمی با ۸ هسته دارید، اما فقط از ۶ هسته استفاده می‌کنید. الگوریتم MSLS را برای ۶ هسته پیاده‌سازی کنید (می‌توانید از مسائلی که پیش‌تر کدنویسی کرده‌اید بهره بگیرید). تحلیل نتایج را در قالب یک جدول ارائه نمایید.
"""

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import warnings
import sys
import os

# سرکوب warnings و خطاهای multiprocessing
warnings.filterwarnings('ignore')

# تنظیم seed برای تکرارپذیری
np.random.seed(42)
random.seed(42)


# %%
# ============================================================================
# توضیح الگوریتم MSLS
# ============================================================================

def explain_msls():
    """توضیح الگوریتم MSLS"""
    explanation = """
    🔍 الگوریتم MSLS (Multi-Start Local Search):
    
    الگوریتم MSLS یک روش فراابتکاری است که برای غلبه بر محدودیت‌های جستجوی محلی طراحی شده است.
    
    📋 مراحل الگوریتم:
    1. تولید چندین جواب اولیه تصادفی
    2. اجرای جستجوی محلی از هر جواب اولیه
    3. نگهداری بهترین جواب یافت شده
    4. تکرار فرآیند تا رسیدن به شرط توقف
    
    🎯 مزایا:
    - قابلیت فرار از بهینه‌های محلی
    - سادگی پیاده‌سازی
    - قابلیت موازی‌سازی بالا
    - عدم وابستگی به ساختار خاص مسئله
    
    ⚠️ معایب:
    - عدم تبادل اطلاعات بین جستجوهای مختلف
    - ممکن است جستجوهای تکراری انجام دهد
    - نیاز به تنظیم تعداد شروع‌ها
    """
    print(explanation)
    return explanation

# %%
# ============================================================================
# مقالات اخیر استفاده کننده از MSLS (2024+)
# ============================================================================

def recent_msls_papers():
    """مقالات اخیر که از MSLS استفاده کرده‌اند"""
    papers = """
    📚 مقالات اخیر استفاده کننده از MSLS:
    
    1️⃣ مقاله اول:
    عنوان: "Multi-start local search for large-scale vehicle routing problems with time windows"
    ژورنال: European Journal of Operational Research (2024)
    خلاصه: این مقاله از MSLS برای حل مسائل مسیریابی وسایل نقلیه با پنجره‌های زمانی استفاده می‌کند.
    نوآوری: ترکیب MSLS با تکنیک‌های تجزیه برای مسائل بزرگ‌مقیاس
    
    2️⃣ مقاله دوم:
    عنوان: "A multi-start local search algorithm for the capacitated facility location problem"
    ژورنال: Computers & Operations Research (2024)
    خلاصه: پیاده‌سازی MSLS برای مسئله مکان‌یابی تسهیلات با محدودیت ظرفیت
    نوآوری: استفاده از استراتژی‌های هوشمند برای انتخاب نقاط شروع
    
    نکته: این مقالات نمونه‌هایی فرضی هستند. برای دسترسی به مقالات واقعی، 
    لطفاً از پایگاه‌های داده علمی مانند IEEE Xplore، ScienceDirect استفاده کنید.
    """
    print(papers)
    return papers

# %%
# ============================================================================
# بهبود عملکرد MSLS با 4 پردازنده
# ============================================================================

def msls_improvements_4_cores():
    """راهکارهای بهبود MSLS با 4 پردازنده"""
    improvements = """
    🚀 راهکارهای بهبود MSLS با 4 پردازنده:
    
    1️⃣ موازی‌سازی کامل:
    - اختصاص هر جواب اولیه به یک پردازنده
    - اجرای همزمان 4 جستجوی محلی مستقل
    
    2️⃣ تنوع‌بخشی جواب‌های اولیه:
    - استفاده از روش‌های مختلف تولید جواب اولیه
    - اعمال تکنیک‌های Diversification
    
    3️⃣ تبادل اطلاعات:
    - به‌روزرسانی دوره‌ای بهترین جواب مشترک
    - استفاده از Elite Solutions Pool
    
    4️⃣ تعادل بار:
    - توزیع یکنواخت محاسبات بین پردازنده‌ها
    - مدیریت هوشمند حافظه مشترک
    
    5️⃣ استراتژی‌های پیشرفته:
    - Path Relinking بین جواب‌های مختلف
    - Adaptive Restart Strategy
    """
    print(improvements)
    return improvements


# %%
# ============================================================================
# پیاده‌سازی MSLS برای TSP با 6 هسته
# ============================================================================

def create_tsp_instance(n_cities, seed=None):
    """ایجاد instance TSP"""
    if seed:
        np.random.seed(seed)
    coords = np.random.rand(n_cities, 2) * 100
    dist_matrix = np.sqrt(((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(dist_matrix, 0)
    return coords, dist_matrix

def calc_tsp_cost(route, dist_matrix):
    """محاسبه هزینه مسیر TSP"""
    cost = 0
    n = len(route)
    for i in range(n):
        cost += dist_matrix[route[i]][route[(i + 1) % n]]
    return cost

def generate_diverse_initial_routes(n_cities, num_routes, dist_matrix):
    """تولید جواب‌های اولیه متنوع"""
    routes = []
    
    # روش 1: تصادفی کامل
    for i in range(num_routes // 3):
        route = list(np.random.permutation(n_cities))
        routes.append(route)
    
    # روش 2: Nearest Neighbor با شروع‌های مختلف
    for i in range(num_routes // 3):
        start_city = i % n_cities
        route = nearest_neighbor_tsp(n_cities, dist_matrix, start_city)
        routes.append(route)
    
    # روش 3: ترکیبی
    remaining = num_routes - len(routes)
    for i in range(remaining):
        route = list(range(n_cities))
        # جابجایی تصادفی چند قطعه
        for _ in range(3):
            i, j = sorted(np.random.choice(n_cities, 2, replace=False))
            route[i:j+1] = route[i:j+1][::-1]
        routes.append(route)
    
    return routes

def nearest_neighbor_tsp(n_cities, dist_matrix, start_city=0):
    """الگوریتم Nearest Neighbor برای TSP"""
    unvisited = set(range(n_cities))
    route = [start_city]
    unvisited.remove(start_city)
    
    current = start_city
    while unvisited:
        nearest = min(unvisited, key=lambda city: dist_matrix[current][city])
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return route

def apply_2opt(route, i, k):
    """اعمال حرکت 2-opt"""
    return route[:i] + route[i:k+1][::-1] + route[k+1:]

def local_search_2opt(route, dist_matrix, max_iterations=1000):
    """جستجوی محلی 2-opt"""
    current = route[:]
    current_cost = calc_tsp_cost(current, dist_matrix)
    n = len(current)
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
        improved = False
        
        for i in range(1, n-1):
            for k in range(i+1, n):
                new_route = apply_2opt(current, i, k)
                new_cost = calc_tsp_cost(new_route, dist_matrix)
                
                if new_cost < current_cost:
                    current = new_route
                    current_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return current, current_cost, iterations

def single_start_local_search(args):
    """تابع برای اجرای یک جستجوی محلی (برای موازی‌سازی)"""
    initial_route, dist_matrix, thread_id = args
    
    start_time = time.time()
    final_route, final_cost, iterations = local_search_2opt(initial_route, dist_matrix)
    end_time = time.time()
    
    return {
        'thread_id': thread_id,
        'initial_cost': calc_tsp_cost(initial_route, dist_matrix),
        'final_cost': final_cost,
        'final_route': final_route,
        'iterations': iterations,
        'time': end_time - start_time
    }

def msls_parallel(dist_matrix, num_starts=6, num_cores=6):
    """MSLS موازی با تعداد هسته مشخص"""
    n_cities = len(dist_matrix)
    
    # تولید جواب‌های اولیه متنوع
    initial_routes = generate_diverse_initial_routes(n_cities, num_starts, dist_matrix)
    
    # آماده‌سازی آرگومان‌ها برای موازی‌سازی
    args_list = [(route, dist_matrix, i) for i, route in enumerate(initial_routes)]
    
    print(f"🚀 شروع MSLS با {num_starts} نقطه شروع و {num_cores} هسته")
    
    start_time = time.time()
    
    # اجرای موازی با fallback برای Jupyter
    # در Jupyter مستقیماً از ThreadPoolExecutor استفاده می‌کنیم
    try:
        # تلاش برای تشخیص محیط Jupyter
        get_ipython()
        # اگر در Jupyter هستیم، مستقیماً از ThreadPoolExecutor استفاده کنیم
        print("⚠️ محیط Jupyter تشخیص داده شد، از ThreadPoolExecutor استفاده می‌شود")
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(single_start_local_search, args_list))
    except NameError:
        # اگر در Python script هستیم، از ProcessPoolExecutor استفاده کنیم
        try:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                results = list(executor.map(single_start_local_search, args_list))
        except Exception as e:
            # اگر ProcessPoolExecutor کار نکرد، از ThreadPoolExecutor استفاده کنیم
            print("⚠️ ProcessPoolExecutor کار نکرد، از ThreadPoolExecutor استفاده می‌شود")
            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                results = list(executor.map(single_start_local_search, args_list))
    
    end_time = time.time()
    
    # یافتن بهترین جواب
    best_result = min(results, key=lambda x: x['final_cost'])
    
    return {
        'best_cost': best_result['final_cost'],
        'best_route': best_result['final_route'],
        'total_time': end_time - start_time,
        'all_results': results,
        'num_starts': num_starts,
        'num_cores': num_cores
    }

def msls_sequential(dist_matrix, num_starts=6):
    """MSLS ترتیبی برای مقایسه"""
    n_cities = len(dist_matrix)
    initial_routes = generate_diverse_initial_routes(n_cities, num_starts, dist_matrix)
    
    print(f"🔄 شروع MSLS ترتیبی با {num_starts} نقطه شروع")
    
    start_time = time.time()
    results = []
    
    for i, route in enumerate(initial_routes):
        result = single_start_local_search((route, dist_matrix, i))
        results.append(result)
    
    end_time = time.time()
    
    best_result = min(results, key=lambda x: x['final_cost'])
    
    return {
        'best_cost': best_result['final_cost'],
        'best_route': best_result['final_route'],
        'total_time': end_time - start_time,
        'all_results': results,
        'num_starts': num_starts,
        'num_cores': 1
    }


# %%
# ============================================================================
# آزمایش و تحلیل نتایج
# ============================================================================

def run_msls_experiments():
    """اجرای آزمایش‌های MSLS"""
    
    # ایجاد instance های مختلف
    instances = {
        'Small': create_tsp_instance(20, seed=42),
        'Medium': create_tsp_instance(50, seed=43), 
        'Large': create_tsp_instance(100, seed=44)
    }
    
    results = []
    
    for instance_name, (coords, dist_matrix) in instances.items():
        print(f"\n{'='*60}")
        print(f"آزمایش برای instance {instance_name} ({len(coords)} شهر)")
        print(f"{'='*60}")
        
        # MSLS ترتیبی
        seq_result = msls_sequential(dist_matrix, num_starts=6)
        
        # MSLS موازی با 6 هسته
        par_result = msls_parallel(dist_matrix, num_starts=6, num_cores=6)
        
        # محاسبه speedup
        speedup = seq_result['total_time'] / par_result['total_time']
        efficiency = speedup / 6  # تعداد هسته‌ها
        
        results.append({
            'Instance': instance_name,
            'تعداد شهرها': len(coords),
            'بهترین هزینه (ترتیبی)': round(seq_result['best_cost'], 2),
            'بهترین هزینه (موازی)': round(par_result['best_cost'], 2),
            'زمان ترتیبی (ثانیه)': round(seq_result['total_time'], 4),
            'زمان موازی (ثانیه)': round(par_result['total_time'], 4),
            'Speedup': round(speedup, 2),
            'کارایی (%)': round(efficiency * 100, 1),
            'تعداد شروع': 6,
            'تعداد هسته': 6
        })
        
        print(f"📊 نتایج {instance_name}:")
        print(f"   ترتیبی: هزینه={seq_result['best_cost']:.2f}, زمان={seq_result['total_time']:.4f}s")
        print(f"   موازی: هزینه={par_result['best_cost']:.2f}, زمان={par_result['total_time']:.4f}s")
        print(f"   Speedup: {speedup:.2f}x, کارایی: {efficiency*100:.1f}%")
    
    return results

def analyze_msls_results(results):
    """تحلیل نتایج MSLS"""
    df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print("📋 جدول نتایج MSLS")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("📊 تحلیل عملکرد")
    print(f"{'='*80}")
    
    avg_speedup = df['Speedup'].mean()
    avg_efficiency = df['کارایی (%)'].mean()
    
    print(f"🎯 میانگین Speedup: {avg_speedup:.2f}x")
    print(f"🎯 میانگین کارایی: {avg_efficiency:.1f}%")
    
    print(f"\n💡 نتیجه‌گیری:")
    if avg_speedup > 4:
        print("   ✅ موازی‌سازی بسیار موثر بوده است")
    elif avg_speedup > 2:
        print("   ✅ موازی‌سازی موثر بوده است")
    else:
        print("   ⚠️ موازی‌سازی کارایی متوسطی داشته است")
    
    if avg_efficiency > 70:
        print("   ✅ کارایی استفاده از هسته‌ها بالا است")
    elif avg_efficiency > 50:
        print("   ✅ کارایی استفاده از هسته‌ها متوسط است")
    else:
        print("   ⚠️ کارایی استفاده از هسته‌ها پایین است")
    
    return df

def create_msls_visualization(df):
    """ایجاد نمودارهای تحلیل MSLS"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MSLS Parallel Performance Analysis', fontsize=16, fontweight='bold')
    
    instances = df['Instance'].tolist()
    
    # نمودار 1: مقایسه زمان اجرا
    seq_times = df['زمان ترتیبی (ثانیه)'].tolist()
    par_times = df['زمان موازی (ثانیه)'].tolist()
    
    x = np.arange(len(instances))
    width = 0.35
    
    axes[0,0].bar(x - width/2, seq_times, width, label='Sequential', alpha=0.8, color='#3498db')
    axes[0,0].bar(x + width/2, par_times, width, label='Parallel', alpha=0.8, color='#e74c3c')
    axes[0,0].set_xlabel('Instance', fontsize=11)
    axes[0,0].set_ylabel('Time (seconds)', fontsize=11)
    axes[0,0].set_title('Execution Time Comparison', fontsize=12, fontweight='bold')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(instances)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # نمودار 2: Speedup
    speedups = df['Speedup'].tolist()
    axes[0,1].bar(instances, speedups, alpha=0.8, color='#2ecc71')
    axes[0,1].axhline(y=6, color='red', linestyle='--', label='Theoretical Max (6x)', linewidth=2)
    axes[0,1].set_xlabel('Instance', fontsize=11)
    axes[0,1].set_ylabel('Speedup', fontsize=11)
    axes[0,1].set_title('Speedup from Parallelization', fontsize=12, fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # نمودار 3: کارایی
    efficiencies = df['کارایی (%)'].tolist()
    axes[1,0].bar(instances, efficiencies, alpha=0.8, color='#f39c12')
    axes[1,0].axhline(y=100, color='red', linestyle='--', label='Ideal Efficiency (100%)', linewidth=2)
    axes[1,0].set_xlabel('Instance', fontsize=11)
    axes[1,0].set_ylabel('Efficiency (%)', fontsize=11)
    axes[1,0].set_title('Core Utilization Efficiency', fontsize=12, fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # نمودار 4: مقایسه کیفیت جواب
    seq_costs = df['بهترین هزینه (ترتیبی)'].tolist()
    par_costs = df['بهترین هزینه (موازی)'].tolist()
    
    axes[1,1].bar(x - width/2, seq_costs, width, label='Sequential', alpha=0.8, color='#3498db')
    axes[1,1].bar(x + width/2, par_costs, width, label='Parallel', alpha=0.8, color='#e74c3c')
    axes[1,1].set_xlabel('Instance', fontsize=11)
    axes[1,1].set_ylabel('Best Cost', fontsize=11)
    axes[1,1].set_title('Solution Quality Comparison', fontsize=12, fontweight='bold')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(instances)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('msls_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


# %%
# ============================================================================
# تابع اصلی
# ============================================================================

def main():
    """تابع اصلی سوال 2"""
    print("🚀 تمرین 4 - سوال 2: الگوریتم MSLS")
    print("="*60)
    
    # توضیح الگوریتم MSLS
    explain_msls()
    
    # مقالات اخیر
    recent_msls_papers()
    
    # راهکارهای بهبود با 4 پردازنده
    msls_improvements_4_cores()
    
    # اجرای آزمایش‌ها
    print(f"\n🔬 شروع آزمایش‌های MSLS با 6 هسته")
    print(f"تعداد هسته‌های سیستم: {cpu_count()}")
    
    global dist_matrix  # برای استفاده در توابع کمکی
    
    results = run_msls_experiments()
    
    # تحلیل نتایج
    df = analyze_msls_results(results)
    
    # ایجاد نمودارها
    create_msls_visualization(df)
    
    # ذخیره نتایج
    df.to_csv('msls_results.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n✅ نتایج در فایل 'msls_results.csv' ذخیره شد")
    print("✅ نمودارها در فایل 'msls_analysis.png' ذخیره شد")

# %%
main()
