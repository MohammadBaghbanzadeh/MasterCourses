# Exported notebook code; may contain IPython commands.
# %%
#!pip install -i https://mirror-pypi.runflare.com/simple networkx matplotlib pulp
!pip install networkx matplotlib pulp

# %%
# Vertex Cover Exercise - Combinatorial Optimization
import networkx as nx
import matplotlib.pyplot as plt
from pulp import *

# ==================== بخش ب و و: ILP و LP Relaxation ====================
def solve_vertex_cover_ilp(G, relaxation=False):
    prob = LpProblem("Vertex_Cover", LpMinimize)
    x = {v: LpVariable(f"x_{v}", cat="Continuous" if relaxation else "Binary", lowBound=0, upBound=1)
         for v in G.nodes()}

    # Objective
    prob += lpSum(x[v] for v in G.nodes())

    # Constraints
    for u, v in G.edges():
        prob += x[u] + x[v] >= 1

    prob.solve(PULP_CBC_CMD(msg=0))

    # تصحیح: شرط rounding برابر با >= 0.5 (پوشش همواره معتبر)
    cover = [v for v in G.nodes() if value(x[v]) >= 0.5]
    obj_value = value(prob.objective)
    return obj_value, cover, {v: value(x[v]) for v in G.nodes()}

# ==================== بخش ج و د: الگوریتم حریصانه ====================
def greedy_vertex_cover(G):
    G_copy = G.copy()
    cover = []
    while G_copy.edges():
        u, v = next(iter(G_copy.edges()))
        cover.extend([u, v])
        # حذف رأس‌ها و یال‌های incident
        G_copy.remove_node(u)
        if v in G_copy:
            G_copy.remove_node(v)
    return len(set(cover)), set(cover)

# ==================== نمونه‌های گراف برای تست ====================
G1 = nx.Graph()
G1.add_edges_from([(0,1), (1,2), (2,0), (2,3)])

G2 = nx.complete_bipartite_graph(3, 3)

G3 = nx.cycle_graph(5)

graphs = {"G1": G1, "G2": G2, "G3": G3}

print("نتایج برای نمونه‌ها:\n")

for name, G in graphs.items():
    print(f"--- گراف {name} ---")
    print("تعداد رأس:", G.number_of_nodes(), "تعداد یال:", G.number_of_edges())

    # ILP
    ilp_val, ilp_cover, _ = solve_vertex_cover_ilp(G, relaxation=False)
    print(f"ILP (بهینه): اندازه = {ilp_val}, پوشش = {sorted(ilp_cover)}")

    # LP Relaxation
    lp_val, lp_rounded, lp_values = solve_vertex_cover_ilp(G, relaxation=True)
    print(f"LP Relaxation: مقدار = {lp_val:.2f}")
    print(f"   مقادیر fractional: { {v: round(lp_values[v],3) for v in lp_values} }")
    print(f"   بعد از rounding ساده (>=0.5): اندازه تقریبی = {len(lp_rounded)}")

    # Greedy
    greedy_size, greedy_cover = greedy_vertex_cover(G)
    print(f"Greedy: اندازه = {greedy_size}, پوشش = {sorted(greedy_cover)}")

    print("\n")

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue')
    plt.title(f"Graph {name}")
    plt.show()

# %%

