#!/usr/bin/env python3
"""
مقایسه الگوریتم‌های LS و MSLS با CPLEX
برای مسائل Graph Coloring و UFLP

⚠️ نکته: نسخه Community Edition از CPLEX محدودیت دارد:
   - حداکثر 1000 متغیر
   - حداکثر 1000 محدودیت
   برای مسائل بزرگ‌تر، نیاز به لایسنس کامل است.
"""

import numpy as np
import time
import pandas as pd
from typing import List, Tuple
import random

# بررسی وجود CPLEX
try:
    from docplex.mp.model import Model
    CPLEX_AVAILABLE = True
    print("✅ CPLEX در دسترس است")
except ImportError:
    CPLEX_AVAILABLE = False
    print("❌ CPLEX در دسترس نیست. لطفاً نصب کنید:")
    print("   pip install cplex docplex")

np.random.seed(42)
random.seed(42)

# ============================================================================
# کلاس‌های مسائل
# ============================================================================

class GraphColoringProblem:
    """مسئله رنگ‌آمیزی گراف"""
    
    def __init__(self, n_vertices: int, edge_probability: float = 0.3):
        self.n = n_vertices
        self.adj_matrix = self._generate_random_graph(edge_probability)
        
    def _generate_random_graph(self, p: float) -> np.ndarray:
        adj = np.random.choice([0, 1], size=(self.n, self.n), p=[1-p, p])
        adj = np.tril(adj, -1) + np.tril(adj, -1).T
        np.fill_diagonal(adj, 0)
        return adj.astype(int)
    
    def calculate_conflicts(self, coloring: List[int]) -> int:
        conflicts = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                if self.adj_matrix[i][j] == 1 and coloring[i] == coloring[j]:
                    conflicts += 1
        return conflicts
    
    def num_colors_used(self, coloring: List[int]) -> int:
        return len(set(coloring))


class UFLPProblem:
    """مسئله مکان‌یابی تسهیلات"""
    
    def __init__(self, n_facilities: int, n_customers: int):
        self.n_facilities = n_facilities
        self.n_customers = n_customers
        self.opening_costs = np.random.randint(100, 500, n_facilities)
        self.assignment_costs = np.random.randint(10, 100, (n_customers, n_facilities))
    
    def calculate_cost(self, open_facilities: List[int], assignments: List[int]) -> float:
        opening_cost = sum(self.opening_costs[f] for f in open_facilities)
        assignment_cost = sum(self.assignment_costs[c][assignments[c]] 
                            for c in range(self.n_customers))
        return opening_cost + assignment_cost


# ============================================================================
# حل با CPLEX - Graph Coloring
# ============================================================================

def solve_graph_coloring_cplex(problem: GraphColoringProblem, max_colors: int = None, time_limit: int = 60):
    """حل مسئله Graph Coloring با CPLEX"""
    
    if not CPLEX_AVAILABLE:
        return None
    
    try:
        n = problem.n
        if max_colors is None:
            max_colors = n
        
        model = Model(name='GraphColoring')
        model.set_time_limit(time_limit)
        
        # متغیرها
        x = {}
        for i in range(n):
            for c in range(max_colors):
                x[i, c] = model.binary_var(name=f'x_{i}_{c}')
        
        y = {}
        for c in range(max_colors):
            y[c] = model.binary_var(name=f'y_{c}')
        
        # محدودیت‌ها
        for i in range(n):
            model.add_constraint(model.sum(x[i, c] for c in range(max_colors)) == 1)
        
        for i in range(n):
            for j in range(i+1, n):
                if problem.adj_matrix[i][j] == 1:
                    for c in range(max_colors):
                        model.add_constraint(x[i, c] + x[j, c] <= 1)
        
        for i in range(n):
            for c in range(max_colors):
                model.add_constraint(x[i, c] <= y[c])
        
        # تابع هدف
        model.minimize(model.sum(y[c] for c in range(max_colors)))
        
        # حل
        start_time = time.time()
        solution = model.solve(log_output=False)
        elapsed = time.time() - start_time
        
        if solution:
            coloring = []
            for i in range(n):
                for c in range(max_colors):
                    if x[i, c].solution_value > 0.5:
                        coloring.append(c)
                        break
            
            num_colors = int(solution.objective_value)
            
            return {
                'coloring': coloring,
                'num_colors': num_colors,
                'conflicts': problem.calculate_conflicts(coloring),
                'time': elapsed,
                'status': 'optimal'
            }
        else:
            return None
            
    except Exception as e:
        print(f"  ⚠️  خطا در CPLEX: {str(e)[:100]}")
        return None


# ============================================================================
# حل با CPLEX - UFLP
# ============================================================================

def solve_uflp_cplex(problem: UFLPProblem, time_limit: int = 60):
    """حل مسئله UFLP با CPLEX"""
    
    if not CPLEX_AVAILABLE:
        return None
    
    try:
        n_fac = problem.n_facilities
        n_cust = problem.n_customers
        
        model = Model(name='UFLP')
        model.set_time_limit(time_limit)
        
        # متغیرها
        y = {}
        for i in range(n_fac):
            y[i] = model.binary_var(name=f'y_{i}')
        
        x = {}
        for i in range(n_fac):
            for j in range(n_cust):
                x[i, j] = model.binary_var(name=f'x_{i}_{j}')
        
        # محدودیت‌ها
        for j in range(n_cust):
            model.add_constraint(model.sum(x[i, j] for i in range(n_fac)) == 1)
        
        for i in range(n_fac):
            for j in range(n_cust):
                model.add_constraint(x[i, j] <= y[i])
        
        # تابع هدف
        opening_cost = model.sum(problem.opening_costs[i] * y[i] for i in range(n_fac))
        assignment_cost = model.sum(problem.assignment_costs[j][i] * x[i, j] 
                                    for i in range(n_fac) for j in range(n_cust))
        model.minimize(opening_cost + assignment_cost)
        
        # حل
        start_time = time.time()
        solution = model.solve(log_output=False)
        elapsed = time.time() - start_time
        
        if solution:
            open_facilities = [i for i in range(n_fac) if y[i].solution_value > 0.5]
            
            assignments = []
            for j in range(n_cust):
                for i in range(n_fac):
                    if x[i, j].solution_value > 0.5:
                        assignments.append(i)
                        break
            
            cost = solution.objective_value
            
            return {
                'open_facilities': open_facilities,
                'assignments': assignments,
                'cost': cost,
                'time': elapsed,
                'status': 'optimal'
            }
        else:
            return None
            
    except Exception as e:
        print(f"  ⚠️  خطا در CPLEX: {str(e)[:100]}")
        return None


# ============================================================================
# الگوریتم‌های LS (نسخه ساده)
# ============================================================================

class GraphColoringLS:
    """الگوریتم Local Search ساده برای Graph Coloring"""
    
    def __init__(self, problem: GraphColoringProblem):
        self.problem = problem
    
    def solve(self, max_iterations: int = 1000):
        n = self.problem.n
        current = [random.randint(0, n-1) for _ in range(n)]
        
        start_time = time.time()
        
        for _ in range(max_iterations):
            improved = False
            best_coloring = current
            best_colors = self.problem.num_colors_used(current)
            
            for v in range(n):
                for c in range(max(current) + 2):
                    if c != current[v]:
                        neighbor = current.copy()
                        neighbor[v] = c
                        
                        if self.problem.calculate_conflicts(neighbor) == 0:
                            n_colors = self.problem.num_colors_used(neighbor)
                            if n_colors < best_colors:
                                best_coloring = neighbor
                                best_colors = n_colors
                                improved = True
            
            if improved:
                current = best_coloring
            else:
                break
        
        elapsed = time.time() - start_time
        
        return {
            'coloring': current,
            'num_colors': self.problem.num_colors_used(current),
            'conflicts': self.problem.calculate_conflicts(current),
            'time': elapsed
        }


class UFLPLS:
    """الگوریتم Local Search ساده برای UFLP"""
    
    def __init__(self, problem: UFLPProblem):
        self.problem = problem
    
    def solve(self, max_iterations: int = 1000):
        n_fac = self.problem.n_facilities
        n_cust = self.problem.n_customers
        
        open_facilities = list(range(n_fac))
        assignments = [min(range(n_fac), 
                          key=lambda f: self.problem.assignment_costs[c][f])
                      for c in range(n_cust)]
        
        current_cost = self.problem.calculate_cost(open_facilities, assignments)
        
        start_time = time.time()
        
        for _ in range(max_iterations):
            improved = False
            
            for f in open_facilities:
                if len(open_facilities) > 1:
                    new_open = [x for x in open_facilities if x != f]
                    new_assign = [min(new_open, 
                                     key=lambda i: self.problem.assignment_costs[c][i])
                                 for c in range(n_cust)]
                    new_cost = self.problem.calculate_cost(new_open, new_assign)
                    
                    if new_cost < current_cost:
                        open_facilities = new_open
                        assignments = new_assign
                        current_cost = new_cost
                        improved = True
                        break
            
            if not improved:
                break
        
        elapsed = time.time() - start_time
        
        return {
            'open_facilities': open_facilities,
            'assignments': assignments,
            'cost': current_cost,
            'time': elapsed
        }


# ============================================================================
# مقایسه و تولید گزارش
# ============================================================================

def compare_graph_coloring():
    """مقایسه LS و CPLEX برای Graph Coloring"""
    
    print("\n" + "="*80)
    print("مقایسه Graph Coloring: LS vs CPLEX")
    print("="*80)
    print("\n⚠️  نکته: نسخه Community Edition محدودیت دارد (1000 vars, 1000 consts)")
    print("   برای مسائل بزرگ‌تر، نیاز به لایسنس کامل است.\n")
    
    # نمونه‌های کوچک‌تر برای Community Edition
    sizes = [
        ("کوچک", 8, 0.3),
        ("متوسط", 10, 0.3),
        ("بزرگ", 12, 0.3)
    ]
    
    results = []
    
    for size_name, n_vertices, edge_prob in sizes:
        print(f"🔍 نمونه {size_name}: {n_vertices} راس")
        
        problem = GraphColoringProblem(n_vertices, edge_prob)
        
        # حل با LS
        ls = GraphColoringLS(problem)
        result_ls = ls.solve()
        
        print(f"  LS:    {result_ls['num_colors']} رنگ، {result_ls['time']:.3f}s")
        
        # حل با CPLEX
        if CPLEX_AVAILABLE:
            result_cplex = solve_graph_coloring_cplex(problem, time_limit=60)
            if result_cplex:
                print(f"  CPLEX: {result_cplex['num_colors']} رنگ، {result_cplex['time']:.3f}s")
                
                gap = ((result_ls['num_colors'] - result_cplex['num_colors']) / 
                       result_cplex['num_colors'] * 100) if result_cplex['num_colors'] > 0 else 0
                speedup = result_cplex['time'] / result_ls['time'] if result_ls['time'] > 0 else 1
                
                results.append({
                    'اندازه': size_name,
                    'راس': n_vertices,
                    'LS - رنگ': result_ls['num_colors'],
                    'LS - زمان': f"{result_ls['time']:.3f}",
                    'CPLEX - رنگ': result_cplex['num_colors'],
                    'CPLEX - زمان': f"{result_cplex['time']:.3f}",
                    'Gap (%)': f"{gap:.2f}",
                    'Speedup': f"{speedup:.2f}x"
                })
            else:
                results.append({
                    'اندازه': size_name,
                    'راس': n_vertices,
                    'LS - رنگ': result_ls['num_colors'],
                    'LS - زمان': f"{result_ls['time']:.3f}",
                    'CPLEX - رنگ': 'Error',
                    'CPLEX - زمان': 'N/A',
                    'Gap (%)': 'N/A',
                    'Speedup': 'N/A'
                })
        else:
            results.append({
                'اندازه': size_name,
                'راس': n_vertices,
                'LS - رنگ': result_ls['num_colors'],
                'LS - زمان': f"{result_ls['time']:.3f}",
                'CPLEX - رنگ': 'N/A',
                'CPLEX - زمان': 'N/A',
                'Gap (%)': 'N/A',
                'Speedup': 'N/A'
            })
    
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("جدول مقایسه Graph Coloring")
    print("="*80)
    print(df.to_string(index=False))
    
    return df


def compare_uflp():
    """مقایسه LS و CPLEX برای UFLP"""
    
    print("\n" + "="*80)
    print("مقایسه UFLP: LS vs CPLEX")
    print("="*80)
    
    # نمونه‌های کوچک‌تر
    sizes = [
        ("کوچک", 5, 10),
        ("متوسط", 7, 12),
        ("بزرگ", 10, 15)
    ]
    
    results = []
    
    for size_name, n_fac, n_cust in sizes:
        print(f"\n🔍 نمونه {size_name}: {n_fac} تسهیل، {n_cust} مشتری")
        
        problem = UFLPProblem(n_fac, n_cust)
        
        # حل با LS
        ls = UFLPLS(problem)
        result_ls = ls.solve()
        
        print(f"  LS:    هزینه={int(result_ls['cost'])}, {result_ls['time']:.3f}s")
        
        # حل با CPLEX
        if CPLEX_AVAILABLE:
            result_cplex = solve_uflp_cplex(problem, time_limit=60)
            if result_cplex:
                print(f"  CPLEX: هزینه={int(result_cplex['cost'])}, {result_cplex['time']:.3f}s")
                
                gap = ((result_ls['cost'] - result_cplex['cost']) / 
                       result_cplex['cost'] * 100) if result_cplex['cost'] > 0 else 0
                speedup = result_cplex['time'] / result_ls['time'] if result_ls['time'] > 0 else 1
                
                results.append({
                    'اندازه': size_name,
                    'تسهیل': n_fac,
                    'مشتری': n_cust,
                    'LS - هزینه': int(result_ls['cost']),
                    'LS - زمان': f"{result_ls['time']:.3f}",
                    'CPLEX - هزینه': int(result_cplex['cost']),
                    'CPLEX - زمان': f"{result_cplex['time']:.3f}",
                    'Gap (%)': f"{gap:.2f}",
                    'Speedup': f"{speedup:.2f}x"
                })
            else:
                results.append({
                    'اندازه': size_name,
                    'تسهیل': n_fac,
                    'مشتری': n_cust,
                    'LS - هزینه': int(result_ls['cost']),
                    'LS - زمان': f"{result_ls['time']:.3f}",
                    'CPLEX - هزینه': 'Error',
                    'CPLEX - زمان': 'N/A',
                    'Gap (%)': 'N/A',
                    'Speedup': 'N/A'
                })
        else:
            results.append({
                'اندازه': size_name,
                'تسهیل': n_fac,
                'مشتری': n_cust,
                'LS - هزینه': int(result_ls['cost']),
                'LS - زمان': f"{result_ls['time']:.3f}",
                'CPLEX - هزینه': 'N/A',
                'CPLEX - زمان': 'N/A',
                'Gap (%)': 'N/A',
                'Speedup': 'N/A'
            })
    
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("جدول مقایسه UFLP")
    print("="*80)
    print(df.to_string(index=False))
    
    return df


# ============================================================================
# اجرای اصلی
# ============================================================================

if __name__ == "__main__":
    print("🚀 شروع مقایسه با CPLEX")
    print("="*80)
    
    if not CPLEX_AVAILABLE:
        print("\n⚠️  CPLEX در دسترس نیست!")
        print("برای نصب:")
        print("  pip install cplex docplex")
        print("\nنکته: برای استفاده از CPLEX نیاز به لایسنس دارید.")
        print("      لایسنس دانشجویی رایگان از IBM قابل دریافت است.")
    else:
        # مقایسه Graph Coloring
        gc_results = compare_graph_coloring()
        
        # مقایسه UFLP
        uflp_results = compare_uflp()
        
        print("\n" + "="*80)
        print("✅ مقایسه با موفقیت انجام شد!")
        print("="*80)
        print("\n📝 نکته: نتایج بالا با نسخه Community Edition از CPLEX است.")
        print("   برای مسائل بزرگ‌تر، نیاز به لایسنس کامل دارید.")
