# Exported notebook code; may contain IPython commands.
# %%
import time
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pulp
import pandas as pd

# تنظیم شیوه نمایش نمودارها برای داشتن ظاهری مدرن و زیبا
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#EEEEEE'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

# تنظیم دانه تصادفی برای تکرارپذیری آزمایش‌ها
random.seed(42)
np.random.seed(42)


# %%
def generate_weighted_graph(n, p=0.05, weight_range=(10, 100)):
    """
    تولید یک گراف تصادفی غیرجهت‌دار بر اساس مدل Erdős-Rényi و انتساب وزن‌های مثبت به گره‌ها
    """
    G = nx.erdos_renyi_graph(n, p)
    
    # انتساب وزن‌های مثبت تصادفی به گره‌ها
    for u in G.nodes():
        G.nodes[u]['weight'] = random.randint(weight_range[0], weight_range[1])
        
    return G

# تولید یک نمونه کوچک برای نمایش بصری و صحت‌سنجی اولیه
small_G = generate_weighted_graph(n=15, p=0.25)
print("تعداد گره‌ها:", small_G.number_of_nodes())
print("تعداد یال‌ها:", small_G.number_of_edges())
print("وزن ۵ گره اول:")
for u in list(small_G.nodes())[:5]:
    print(f"گره {u}: وزن {small_G.nodes[u]['weight']}")


# %%
def solve_wvc_ip(G):
    """
    حل مسئله پوشش رأس وزن‌دار با استفاده از برنامه‌ریزی اعداد صحیح (IP)
    """
    start_time = time.time()
    
    # تعریف مسئله کمینه‌سازی
    prob = pulp.LpProblem("Weighted_Vertex_Cover_IP", pulp.LpMinimize)
    
    # تعریف متغیرهای تصمیم‌گیری (باینری 0 و 1)
    x = pulp.LpVariable.dicts("x", G.nodes(), cat=pulp.LpBinary)
    
    # تعریف تابع هدف: مجموع وزن گره‌های انتخاب شده
    prob += pulp.lpSum(G.nodes[i]['weight'] * x[i] for i in G.nodes())
    
    # قیدها: برای هر یال (i, j) حداقل یکی از دو سر یال باید در پوشش باشد
    for u, v in G.edges():
        prob += x[u] + x[v] >= 1
        
    # حل مسئله با خاموش کردن خروجی متنی سالور و محدودیت زمانی ۱۵ ثانیه‌ای
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=15))
    
    end_time = time.time()
    
    # استخراج جواب
    solution = {i: int(x[i].varValue) for i in G.nodes()}
    objective_val = pulp.value(prob.objective)
    
    return objective_val, solution, end_time - start_time

# تست تابع روی گراف کوچک
ip_val, ip_sol, ip_time = solve_wvc_ip(small_G)
print(f"جواب بهینه IP: {ip_val}")
print(f"زمان اجرا: {ip_time:.4f} ثانیه")
print(f"تعداد گره‌های انتخاب شده در پوشش: {sum(ip_sol.values())} از {small_G.number_of_nodes()}")


# %%
# رسم گراف کوچک و مشخص کردن گره‌های پوشش رأس به دست آمده
pos = nx.spring_layout(small_G, seed=42)
node_colors = ['#E74C3C' if ip_sol[node] == 1 else '#3498DB' for node in small_G.nodes()]
node_sizes = [small_G.nodes[node]['weight'] * 8 for node in small_G.nodes()]

plt.figure(figsize=(9, 7))
nx.draw_networkx_nodes(small_G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, edgecolors='#2C3E50')
nx.draw_networkx_edges(small_G, pos, width=1.0, alpha=0.4, edge_color='#BDC3C7')
nx.draw_networkx_labels(small_G, pos, font_size=10, font_family='sans-serif', font_color='white')

# اضافه کردن لژند بصری برای درک بهتر
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='#E74C3C', label='Selected in Cover')
blue_patch = mpatches.Patch(color='#3498DB', label='Not Selected')
plt.legend(handles=[red_patch, blue_patch], loc='upper right', frameon=True, facecolor='white')

plt.title("Optimal IP Weighted Vertex Cover on a 15-Node Graph\n(Node Size is Proportional to Vertex Weight)", fontsize=13, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()


# %%
def solve_wvc_lp_case1(G):
    """
    حل آزادسازی خطی با قید 0 <= x_i <= 1
    """
    start_time = time.time()
    
    prob = pulp.LpProblem("Weighted_Vertex_Cover_LP_Case1", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", G.nodes(), lowBound=0, upBound=1, cat=pulp.LpContinuous)
    
    # تابع هدف
    prob += pulp.lpSum(G.nodes[i]['weight'] * x[i] for i in G.nodes())
    
    # قیدها
    for u, v in G.edges():
        prob += x[u] + x[v] >= 1
        
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    end_time = time.time()
    
    solution = {i: x[i].varValue for i in G.nodes()}
    objective_val = pulp.value(prob.objective)
    
    return objective_val, solution, end_time - start_time

def solve_wvc_lp_case2(G):
    """
    حل آزادسازی خطی با قید x_i >= 0 (بدون سقف بالا)
    """
    start_time = time.time()
    
    prob = pulp.LpProblem("Weighted_Vertex_Cover_LP_Case2", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", G.nodes(), lowBound=0, cat=pulp.LpContinuous)
    
    # تابع هدف
    prob += pulp.lpSum(G.nodes[i]['weight'] * x[i] for i in G.nodes())
    
    # قیدها
    for u, v in G.edges():
        prob += x[u] + x[v] >= 1
        
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    end_time = time.time()
    
    solution = {i: x[i].varValue for i in G.nodes()}
    objective_val = pulp.value(prob.objective)
    
    return objective_val, solution, end_time - start_time

# اجرای هر دو حالت روی گراف کوچک و مقایسه مقادیر هدف و زمان اجرا
lp1_val, lp1_sol, lp1_time = solve_wvc_lp_case1(small_G)
lp2_val, lp2_sol, lp2_time = solve_wvc_lp_case2(small_G)

print(f"مقدار هدف LP حالت اول (0 <= x_i <= 1): {lp1_val:.4f}")
print(f"مقدار هدف LP حالت دوم (x_i >= 0): {lp2_val:.4f}")
print(f"اختلاف دو جواب: {abs(lp1_val - lp2_val):.6f}")

max_val_case2 = max(lp2_sol.values())
print(f"بزرگترین مقدار متغیر در جواب بهینه حالت دوم: {max_val_case2:.4f}")


# %%
def round_lp_solution(G, lp_solution):
    """
    اعمال مکانیسم گرد کردن روی خروجی LP:
    اگر مقدار متغیر بزرگتر یا مساوی 0.5 بود 1، در غیر این صورت 0.
    """
    rounded_solution = {}
    for i in G.nodes():
        if lp_solution[i] >= 0.5:
            rounded_solution[i] = 1
        else:
            rounded_solution[i] = 0
            
    # محاسبه وزن کل جواب گرد شده
    total_weight = sum(G.nodes[i]['weight'] * rounded_solution[i] for i in G.nodes())
    
    # بررسی شدنی بودن (آیا تمام یال‌ها پوشش داده شده‌اند؟)
    is_feasible = True
    for u, v in G.edges():
        if rounded_solution[u] + rounded_solution[v] < 1:
            is_feasible = False
            break
            
    return total_weight, rounded_solution, is_feasible

# اجرای گرد کردن روی جواب گراف کوچک
round_weight, round_sol, round_feasible = round_lp_solution(small_G, lp1_sol)
print(f"وزن پوشش رأس حاصل از گرد کردن: {round_weight}")
print(f"آیا جواب شدنی است؟ {'بله' if round_feasible else 'خیر'}")
print(f"نسبت جواب گرد شده به بهینه IP: {round_weight / ip_val:.4f}")


# %%
def local_search_wvc(G, initial_cover):
    """
    بهبود جواب پوشش رأس با جستجوی محلی (Hill Climbing)
    """
    start_time = time.time()
    
    # کپی کردن پوشش رأس اولیه به صورت مجموعه‌ای از گره‌ها
    cover = {u for u, val in initial_cover.items() if val == 1}
    
    def remove_redundancies(curr_cover):
        # حذف گره‌هایی که تمام همسایگانشان در پوشش هستند
        # کاندیداها را بر اساس وزن نزولی مرتب می‌کنیم تا اول گره‌های سنگین‌تر حذف شوند
        redundant_candidates = sorted(list(curr_cover), key=lambda x: G.nodes[x]['weight'], reverse=True)
        modified = False
        for u in redundant_candidates:
            if all(neighbor in curr_cover for neighbor in G.neighbors(u)):
                curr_cover.remove(u)
                modified = True
        return curr_cover, modified

    # اولین پاکسازی افزونگی‌ها
    cover, _ = remove_redundancies(cover)
    
    improved = True
    iterations = 0
    max_iterations = 1000
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # مرتب‌سازی گره‌های داخل پوشش برای انتخاب به ترتیب نزولی وزن
        cover_sorted = sorted(list(cover), key=lambda x: G.nodes[x]['weight'], reverse=True)
        
        for u in cover_sorted:
            # پیدا کردن همسایگانی از u که در پوشش نیستند
            uncovered_neighbors = [n for n in G.neighbors(u) if n not in cover]
            
            if len(uncovered_neighbors) == 0:
                cover.remove(u)
                cover, _ = remove_redundancies(cover)
                improved = True
                break
                
            elif len(uncovered_neighbors) == 1:
                v = uncovered_neighbors[0]
                # شرط تعویض: وزن گره خارج از پوشش کمتر از گره داخل پوشش باشد
                if G.nodes[v]['weight'] < G.nodes[u]['weight']:
                    cover.remove(u)
                    cover.add(v)
                    cover, _ = remove_redundancies(cover)
                    improved = True
                    break
            else:
                pass
                
    end_time = time.time()
    
    # ساخت دیکشنری جواب نهایی
    final_cover = {u: (1 if u in cover else 0) for u in G.nodes()}
    final_weight = sum(G.nodes[u]['weight'] for u in cover)
    
    return final_weight, final_cover, end_time - start_time

# تست جستجوی محلی روی گراف کوچک
ls_weight, ls_sol, ls_time = local_search_wvc(small_G, round_sol)
print(f"وزن پس از جستجوی محلی: {ls_weight}")
print(f"میزان بهبود نسبت به گرد کردن ساده: {round_weight - ls_weight}")
print(f"نسبت جواب جستجوی محلی به بهینه IP: {ls_weight / ip_val:.4f}")


# %%
# تعریف ابعاد بزرگ مسئله
node_sizes = [100, 150, 200, 250, 300, 350, 400, 450, 500]
p_edge = 0.05  # چگالی یال‌ها در گراف تصادفی
results = []

print("شروع اجرای آزمایش‌های عددی...")
print("-" * 105)
print(f"{'Nodes':<8}{'Edges':<8}{'IP Weight':<12}{'LP Weight':<12}{'Round W.':<10}{'LS Weight':<10}{'IP Time(s)':<12}{'LP Time(s)':<12}{'LS Time(s)':<12}")
print("-" * 105)

for n in node_sizes:
    # ۱. تولید گراف تصادفی با وزن‌های مثبت
    G = generate_weighted_graph(n, p=p_edge, weight_range=(10, 100))
    num_edges = G.number_of_edges()
    
    # ۲. حل به روش باینری IP
    ip_weight, ip_sol, ip_time = solve_wvc_ip(G)
    
    # ۳. حل به روش آزادسازی خطی LP (حالت اول)
    lp_weight, lp_sol, lp_time = solve_wvc_lp_case1(G)
    
    # ۴. حل به روش آزادسازی خطی LP (حالت دوم) برای صحت‌سنجی تجربی فرضیه عدم تجاوز متغیرها از ۱
    _, lp2_sol, _ = solve_wvc_lp_case2(G)
    violating_vars = sum(1 for v in lp2_sol.values() if v > 1.0001)  # تعداد متغیرهای بزرگتر از ۱
    
    # ۵. گرد کردن جواب LP
    round_weight, round_sol, round_feasible = round_lp_solution(G, lp_sol)
    
    # ۶. بهبود با جستجوی محلی
    ls_weight, ls_sol, ls_time = local_search_wvc(G, round_sol)
    
    # محاسبات تحلیل نتایج
    gap_pct = ((ip_weight - lp_weight) / ip_weight) * 100 if ip_weight > 0 else 0
    round_ratio = round_weight / ip_weight if ip_weight > 0 else 1
    ls_ratio = ls_weight / ip_weight if ip_weight > 0 else 1
    
    results.append({
        'nodes': n,
        'edges': num_edges,
        'ip_weight': ip_weight,
        'lp_weight': lp_weight,
        'round_weight': round_weight,
        'ls_weight': ls_weight,
        'ip_time': ip_time,
        'lp_time': lp_time,
        'ls_time': ls_time,
        'gap_pct': gap_pct,
        'round_ratio': round_ratio,
        'ls_ratio': ls_ratio,
        'round_feasible': round_feasible,
        'violating_vars_case2': violating_vars
    })
    
    print(f"{n:<8}{num_edges:<8}{ip_weight:<12.1f}{lp_weight:<12.2f}{round_weight:<10.1f}{ls_weight:<10.1f}{ip_time:<12.4f}{lp_time:<12.4f}{ls_time:<12.4f}")

print("-" * 105)
print("آزمایش‌های تجربی با موفقیت پایان یافت.")


# %%
# ایجاد و نمایش جدول نتایج با Pandas
df = pd.DataFrame(results)
df


# %%
plt.figure(figsize=(10, 6))
plt.plot(df['nodes'], df['ip_weight'], marker='o', linewidth=2.5, color='#2C3E50', label='Optimal IP')
plt.plot(df['nodes'], df['lp_weight'], marker='s', linewidth=2.0, color='#7F8C8D', linestyle='--', label='LP Relaxation')
plt.plot(df['nodes'], df['round_weight'], marker='^', linewidth=2.0, color='#E74C3C', label='Rounding (Approximation)')
plt.plot(df['nodes'], df['ls_weight'], marker='d', linewidth=2.0, color='#27AE60', label='Local Search (Improved)')

plt.title('Total Vertex Cover Weight by Graph Size', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Number of Nodes ($N$)', fontsize=12)
plt.ylabel('Total Weight', fontsize=12)
plt.xticks(df['nodes'])
plt.legend(frameon=True, facecolor='white', edgecolor='#EAEAEA', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# %%
plt.figure(figsize=(10, 6))
plt.plot(df['nodes'], df['ip_time'], marker='o', linewidth=2.5, color='#8E44AD', label='IP Solving Time')
plt.plot(df['nodes'], df['lp_time'], marker='s', linewidth=2.0, color='#2980B9', label='LP Solving Time')
plt.plot(df['nodes'], df['ls_time'], marker='d', linewidth=2.0, color='#27AE60', label='Local Search Time')

plt.title('Execution Time Comparison (Seconds) by Graph Size', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Number of Nodes ($N$)', fontsize=12)
plt.ylabel('Time (Seconds)', fontsize=12)
plt.yscale('log')  # استفاده از مقیاس لگاریتمی به دلیل اختلاف مقیاس زمانی زیاد
plt.xticks(df['nodes'])
plt.legend(frameon=True, facecolor='white', edgecolor='#EAEAEA', fontsize=11)
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# %%
plt.figure(figsize=(10, 6))
plt.plot(df['nodes'], df['round_ratio'], marker='^', linewidth=2.0, color='#E74C3C', label='Rounding / Optimal IP')
plt.plot(df['nodes'], df['ls_ratio'], marker='d', linewidth=2.0, color='#27AE60', label='Local Search / Optimal IP')
plt.axhline(y=2.0, color='#C0392B', linestyle=':', linewidth=2.0, label='Theoretical Approximation Bound (2.0)')
plt.axhline(y=1.0, color='black', linestyle='-', linewidth=1.0)

plt.title('Approximation Ratio vs. Theoretical Bound', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Number of Nodes ($N$)', fontsize=12)
plt.ylabel('Approximation Ratio', fontsize=12)
plt.ylim(0.8, 2.2)
plt.xticks(df['nodes'])
plt.legend(frameon=True, facecolor='white', edgecolor='#EAEAEA', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# %%
plt.figure(figsize=(10, 6))
bars = plt.bar(df['nodes'], df['gap_pct'], color='#34495E', width=20, edgecolor='#2C3E50', alpha=0.85, label='IP-LP Gap %')

plt.title('LP Optimality Gap Percentage', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Number of Nodes ($N$)', fontsize=12)
plt.ylabel('Gap (%)', fontsize=12)
plt.xticks(df['nodes'])

# افزودن مقادیر بالای هر ستون
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=10)
    
plt.ylim(0, max(df['gap_pct']) + 1.5)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

