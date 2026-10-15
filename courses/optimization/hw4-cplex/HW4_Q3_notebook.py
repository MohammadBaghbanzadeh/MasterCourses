# Exported notebook code; may contain IPython commands.
# %%
# وارد کردن کتابخانه‌های مورد نیاز
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random

# تنظیم seed برای تکرارپذیری
np.random.seed(42)
random.seed(42)

# تنظیمات نمایش
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("✅ کتابخانه‌ها با موفقیت بارگذاری شدند")

# %%
class GraphColoringProblem:
    """
    مسئله رنگ‌آمیزی گراف:
    هدف: تخصیص رنگ به راس‌های گراف به گونه‌ای که دو راس مجاور رنگ یکسان نداشته باشند
    و تعداد رنگ‌های استفاده شده حداقل باشد.
    """
    
    def __init__(self, n_vertices: int, edge_probability: float = 0.3):
        self.n = n_vertices
        self.adj_matrix = self._generate_random_graph(edge_probability)
        
    def _generate_random_graph(self, p: float) -> np.ndarray:
        """تولید گراف تصادفی"""
        adj = np.random.choice([0, 1], size=(self.n, self.n), p=[1-p, p])
        adj = np.tril(adj, -1) + np.tril(adj, -1).T
        np.fill_diagonal(adj, 0)
        return adj.astype(int)
    
    def calculate_conflicts(self, coloring: List[int]) -> int:
        """محاسبه تعداد تضادها (یال‌هایی که دو سر آن هم‌رنگ هستند)"""
        conflicts = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                if self.adj_matrix[i][j] == 1 and coloring[i] == coloring[j]:
                    conflicts += 1
        return conflicts
    
    def num_colors_used(self, coloring: List[int]) -> int:
        """تعداد رنگ‌های استفاده شده"""
        return len(set(coloring))
    
    def is_valid_coloring(self, coloring: List[int]) -> bool:
        """بررسی معتبر بودن رنگ‌آمیزی"""
        return self.calculate_conflicts(coloring) == 0


class GraphColoringLS:
    """الگوریتم Local Search برای Graph Coloring"""
    
    def __init__(self, problem: GraphColoringProblem):
        self.problem = problem
        
    def generate_initial_solution(self, max_colors: int = None) -> List[int]:
        """تولید جواب اولیه تصادفی"""
        if max_colors is None:
            max_colors = self.problem.n
        return [random.randint(0, max_colors-1) for _ in range(self.problem.n)]
    
    def get_neighbors(self, coloring: List[int]) -> List[List[int]]:
        """تولید همسایگان با تغییر رنگ یک راس"""
        neighbors = []
        max_color = max(coloring)
        
        for vertex in range(self.problem.n):
            for new_color in range(max_color + 2):  # شامل یک رنگ جدید
                if new_color != coloring[vertex]:
                    neighbor = coloring.copy()
                    neighbor[vertex] = new_color
                    neighbors.append(neighbor)
        
        return neighbors
    
    def local_search(self, max_iterations: int = 1000, mode: str = "best") -> Tuple[List[int], int, int]:
        """
        الگوریتم Local Search
        mode: "best" یا "first"
        """
        current = self.generate_initial_solution()
        current_conflicts = self.problem.calculate_conflicts(current)
        current_colors = self.problem.num_colors_used(current)
        
        iterations = 0
        start_time = time.time()
        
        while iterations < max_iterations and time.time() - start_time < 60:
            iterations += 1
            
            neighbors = self.get_neighbors(current)
            best_neighbor = None
            best_conflicts = current_conflicts
            best_colors = current_colors
            improved = False
            
            for neighbor in neighbors:
                n_conflicts = self.problem.calculate_conflicts(neighbor)
                n_colors = self.problem.num_colors_used(neighbor)
                
                # اولویت: کاهش تضاد، سپس کاهش تعداد رنگ
                if n_conflicts < best_conflicts or \
                   (n_conflicts == best_conflicts and n_colors < best_colors):
                    best_neighbor = neighbor
                    best_conflicts = n_conflicts
                    best_colors = n_colors
                    improved = True
                    
                    if mode == "first":
                        break
            
            if improved:
                current = best_neighbor
                current_conflicts = best_conflicts
                current_colors = best_colors
            else:
                break
        
        return current, current_colors, iterations


class GraphColoringMSLS:
    """Multi-Start Local Search برای Graph Coloring"""
    
    def __init__(self, problem: GraphColoringProblem):
        self.problem = problem
        self.ls = GraphColoringLS(problem)
    
    def multi_start_search(self, num_starts: int = 10, max_iterations: int = 1000) -> Tuple[List[int], int, float]:
        """اجرای MSLS با چندین نقطه شروع"""
        best_solution = None
        best_colors = float('inf')
        best_conflicts = float('inf')
        
        start_time = time.time()
        
        for i in range(num_starts):
            solution, n_colors, iters = self.ls.local_search(max_iterations)
            conflicts = self.problem.calculate_conflicts(solution)
            
            if conflicts < best_conflicts or \
               (conflicts == best_conflicts and n_colors < best_colors):
                best_solution = solution
                best_colors = n_colors
                best_conflicts = conflicts
        
        elapsed = time.time() - start_time
        
        return best_solution, best_colors, elapsed

# %%
class UFLPProblem:
    """
    مسئله مکان‌یابی تسهیلات بدون محدودیت ظرفیت:
    هدف: انتخاب زیرمجموعه‌ای از تسهیلات برای باز کردن و تخصیص مشتریان به آنها
    به گونه‌ای که مجموع هزینه‌های باز کردن و تخصیص حداقل شود.
    """
    
    def __init__(self, n_facilities: int, n_customers: int):
        self.n_facilities = n_facilities
        self.n_customers = n_customers
        
        # هزینه باز کردن تسهیلات
        self.opening_costs = np.random.randint(100, 500, n_facilities)
        
        # هزینه تخصیص (فاصله بین مشتری و تسهیل)
        self.assignment_costs = np.random.randint(10, 100, (n_customers, n_facilities))
    
    def calculate_cost(self, open_facilities: List[int], assignments: List[int]) -> float:
        """محاسبه هزینه کل"""
        # هزینه باز کردن تسهیلات
        opening_cost = sum(self.opening_costs[f] for f in open_facilities)
        
        # هزینه تخصیص مشتریان
        assignment_cost = sum(self.assignment_costs[c][assignments[c]] 
                            for c in range(self.n_customers))
        
        return opening_cost + assignment_cost
    
    def is_valid_solution(self, open_facilities: List[int], assignments: List[int]) -> bool:
        """بررسی معتبر بودن جواب"""
        # همه مشتریان باید به تسهیلات باز تخصیص یابند
        return all(assignments[c] in open_facilities for c in range(self.n_customers))


class UFLPLS:
    """الگوریتم Local Search برای UFLP"""
    
    def __init__(self, problem: UFLPProblem):
        self.problem = problem
    
    def generate_initial_solution(self) -> Tuple[List[int], List[int]]:
        """تولید جواب اولیه: همه تسهیلات باز، هر مشتری به نزدیک‌ترین تخصیص"""
        open_facilities = list(range(self.problem.n_facilities))
        
        # تخصیص هر مشتری به ارزان‌ترین تسهیل
        assignments = []
        for c in range(self.problem.n_customers):
            best_facility = min(range(self.problem.n_facilities),
                              key=lambda f: self.problem.assignment_costs[c][f])
            assignments.append(best_facility)
        
        return open_facilities, assignments
    
    def optimize_assignments(self, open_facilities: List[int]) -> List[int]:
        """بهینه‌سازی تخصیص مشتریان به تسهیلات باز"""
        assignments = []
        for c in range(self.problem.n_customers):
            best_facility = min(open_facilities,
                              key=lambda f: self.problem.assignment_costs[c][f])
            assignments.append(best_facility)
        return assignments
    
    def local_search(self, max_iterations: int = 1000, mode: str = "best") -> Tuple[List[int], List[int], float, int]:
        """الگوریتم Local Search"""
        open_facilities, assignments = self.generate_initial_solution()
        current_cost = self.problem.calculate_cost(open_facilities, assignments)
        
        iterations = 0
        start_time = time.time()
        
        while iterations < max_iterations and time.time() - start_time < 60:
            iterations += 1
            improved = False
            best_cost = current_cost
            best_open = open_facilities
            best_assign = assignments
            
            # حرکت 1: بستن یک تسهیل
            for f in open_facilities:
                if len(open_facilities) > 1:  # حداقل یک تسهیل باید باز باشد
                    new_open = [x for x in open_facilities if x != f]
                    new_assign = self.optimize_assignments(new_open)
                    new_cost = self.problem.calculate_cost(new_open, new_assign)
                    
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_open = new_open
                        best_assign = new_assign
                        improved = True
                        
                        if mode == "first":
                            break
            
            # حرکت 2: باز کردن یک تسهیل بسته
            if not improved or mode == "best":
                closed_facilities = [f for f in range(self.problem.n_facilities) 
                                   if f not in open_facilities]
                
                for f in closed_facilities:
                    new_open = open_facilities + [f]
                    new_assign = self.optimize_assignments(new_open)
                    new_cost = self.problem.calculate_cost(new_open, new_assign)
                    
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_open = new_open
                        best_assign = new_assign
                        improved = True
                        
                        if mode == "first":
                            break
            
            if improved:
                open_facilities = best_open
                assignments = best_assign
                current_cost = best_cost
            else:
                break
        
        elapsed = time.time() - start_time
        return open_facilities, assignments, current_cost, iterations


class UFLPMSLS:
    """Multi-Start Local Search برای UFLP"""
    
    def __init__(self, problem: UFLPProblem):
        self.problem = problem
        self.ls = UFLPLS(problem)
    
    def generate_random_start(self) -> Tuple[List[int], List[int]]:
        """تولید نقطه شروع تصادفی"""
        # انتخاب تصادفی تعدادی تسهیل برای باز کردن
        n_open = random.randint(1, self.problem.n_facilities)
        open_facilities = random.sample(range(self.problem.n_facilities), n_open)
        
        # تخصیص بهینه مشتریان
        assignments = self.ls.optimize_assignments(open_facilities)
        
        return open_facilities, assignments
    
    def multi_start_search(self, num_starts: int = 10, max_iterations: int = 1000) -> Tuple[List[int], List[int], float, float]:
        """اجرای MSLS"""
        best_open = None
        best_assign = None
        best_cost = float('inf')
        
        start_time = time.time()
        
        for i in range(num_starts):
            # شروع از یک نقطه تصادفی
            init_open, init_assign = self.generate_random_start()
            
            # اجرای Local Search
            open_fac, assign, cost, iters = self.ls.local_search(max_iterations)
            
            if cost < best_cost:
                best_cost = cost
                best_open = open_fac
                best_assign = assign
        
        elapsed = time.time() - start_time
        
        return best_open, best_assign, best_cost, elapsed

# %%
def test_graph_coloring():
    """تست الگوریتم‌های Graph Coloring"""
    print("="*80)
    print("تست Graph Coloring Problem")
    print("="*80)
    
    sizes = [
        ("کوچک", 10, 0.3),
        ("متوسط", 20, 0.3),
        ("بزرگ", 30, 0.3)
    ]
    
    results = []
    
    for size_name, n_vertices, edge_prob in sizes:
        print(f"\n🔍 نمونه {size_name}: {n_vertices} راس")
        
        problem = GraphColoringProblem(n_vertices, edge_prob)
        
        # Local Search
        ls = GraphColoringLS(problem)
        sol_ls, colors_ls, iters_ls = ls.local_search(max_iterations=1000, mode="best")
        conflicts_ls = problem.calculate_conflicts(sol_ls)
        
        # MSLS
        msls = GraphColoringMSLS(problem)
        sol_msls, colors_msls, time_msls = msls.multi_start_search(num_starts=10)
        conflicts_msls = problem.calculate_conflicts(sol_msls)
        
        results.append({
            "اندازه": size_name,
            "تعداد راس": n_vertices,
            "LS - تعداد رنگ": colors_ls,
            "LS - تضاد": conflicts_ls,
            "MSLS - تعداد رنگ": colors_msls,
            "MSLS - تضاد": conflicts_msls,
            "MSLS - زمان (ثانیه)": round(time_msls, 3)
        })
        
        print(f"  LS:   {colors_ls} رنگ, {conflicts_ls} تضاد")
        print(f"  MSLS: {colors_msls} رنگ, {conflicts_msls} تضاد ({time_msls:.3f}s)")
    
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("خلاصه نتایج Graph Coloring")
    print("="*80)
    print(df.to_string(index=False))
    
    return results

# %%
# اجرای تست Graph Coloring
print("🚀 شروع تست Graph Coloring...")
gc_results = test_graph_coloring()

# %%
def test_uflp():
    """تست الگوریتم‌های UFLP"""
    print("\n\n" + "="*80)
    print("تست Uncapacitated Facility Location Problem")
    print("="*80)
    
    sizes = [
        ("کوچک", 5, 10),
        ("متوسط", 10, 20),
        ("بزرگ", 15, 30)
    ]
    
    results = []
    
    for size_name, n_fac, n_cust in sizes:
        print(f"\n🔍 نمونه {size_name}: {n_fac} تسهیل, {n_cust} مشتری")
        
        problem = UFLPProblem(n_fac, n_cust)
        
        # Local Search
        ls = UFLPLS(problem)
        open_ls, assign_ls, cost_ls, iters_ls = ls.local_search(max_iterations=1000)
        
        # MSLS
        msls = UFLPMSLS(problem)
        open_msls, assign_msls, cost_msls, time_msls = msls.multi_start_search(num_starts=10)
        
        results.append({
            "اندازه": size_name,
            "تسهیلات": n_fac,
            "مشتریان": n_cust,
            "LS - هزینه": int(cost_ls),
            "LS - تسهیلات باز": len(open_ls),
            "MSLS - هزینه": int(cost_msls),
            "MSLS - تسهیلات باز": len(open_msls),
            "MSLS - زمان (ثانیه)": round(time_msls, 3),
            "بهبود (%)": round((cost_ls - cost_msls) / cost_ls * 100, 2)
        })
        
        print(f"  LS:   هزینه={int(cost_ls)}, {len(open_ls)} تسهیل باز")
        print(f"  MSLS: هزینه={int(cost_msls)}, {len(open_msls)} تسهیل باز ({time_msls:.3f}s)")
        print(f"  بهبود: {((cost_ls - cost_msls) / cost_ls * 100):.2f}%")
    
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("خلاصه نتایج UFLP")
    print("="*80)
    print(df.to_string(index=False))
    
    return results

# %%
# اجرای تست UFLP
print("🚀 شروع تست UFLP...")
uflp_results = test_uflp()
