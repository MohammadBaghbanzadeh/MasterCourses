# Exported notebook code; may contain IPython commands.
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

np.set_printoptions(precision=4, suppress=True)

def softmax(x, axis=-1):
    """Numerically stable softmax along the given axis."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def draw_box(ax, xy, w, h, label, color="#cfe2ff", edge="#1f4e79", fontsize=12):
    """Helper to draw a rounded rectangle with a centered label."""
    box = FancyBboxPatch(xy, w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         linewidth=1.5, edgecolor=edge, facecolor=color)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, label,
            ha="center", va="center", fontsize=fontsize, weight="bold")

def draw_arrow(ax, p1, p2, label="", color="#444", offset=(0, 0.15), fontsize=11):
    """Helper to draw an arrow between two points with an optional label."""
    arrow = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                            color=color, linewidth=1.5)
    ax.add_patch(arrow)
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + offset[0], (p1[1] + p2[1]) / 2 + offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=fontsize, color="#1f4e79", weight="bold")

# %%
# Token embeddings (each row is one token)
X = np.array([
    [1.0, 0.0],   # x1 - cow
    [0.0, 1.0],   # x2 - brown
    [1.0, 1.0],   # x3 - sheep
])

Wq = np.array([
    [1.0, 1.0],
    [0.0, 1.0],
])

Wk = np.array([
    [1.0, 0.0],
    [1.0, 1.0],
])

Q = X @ Wq
K = X @ Wk

print("X (Token Embeddings):\n", X)
print("\nWq:\n", Wq)
print("Wk:\n", Wk)
print("\nQ = X @ Wq:\n", Q)
print("\nK = X @ Wk:\n", K)

# %%
# Diagram for Exercise 1
fig, ax = plt.subplots(figsize=(11, 4))
ax.set_xlim(0, 11); ax.set_ylim(0, 4); ax.axis("off")
ax.set_title("Exercise 1: Embedding -> Q, K spaces", fontsize=13)

draw_box(ax, (0.5, 1.5), 2.2, 1.0, "Embedding\nSpace (X)", color="#fff2cc", edge="#bf9000")
draw_box(ax, (7.5, 2.6), 2.2, 1.0, "Query Space (Q)", color="#cfe2ff")
draw_box(ax, (7.5, 0.4), 2.2, 1.0, "Key Space (K)", color="#d5e8d4", edge="#2d6a4f")

draw_arrow(ax, (2.7, 2.2), (7.5, 3.1), label="x Wq", offset=(0, 0.25))
draw_arrow(ax, (2.7, 1.8), (7.5, 0.9), label="x Wk", offset=(0, -0.35))

plt.tight_layout(); plt.show()

# %%
# Diagram for Exercise 2
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
ax.set_title("Exercise 2: Q, K -> Dot -> Softmax -> Relevancy Matrix", fontsize=13)

draw_box(ax, (0.3, 2.5), 1.8, 1.0, "Q", color="#cfe2ff")
draw_box(ax, (0.3, 0.5), 1.8, 1.0, "K", color="#d5e8d4", edge="#2d6a4f")

# Dot symbol
circ = Circle((3.7, 2.0), 0.45, facecolor="#fde2e4", edgecolor="#9b2226", linewidth=1.5)
ax.add_patch(circ)
ax.text(3.7, 2.0, "·", ha="center", va="center", fontsize=28, weight="bold")
ax.text(3.7, 1.35, "Dot", ha="center", fontsize=11)

# Softmax symbol
draw_box(ax, (5.6, 1.55), 2.1, 0.9, "Softmax", color="#ffe5b4", edge="#a5673f")

# Final relevancy matrix
draw_box(ax, (9.3, 1.4), 2.4, 1.2, "Relevancy\nMatrix (3x3)", color="#e1d5e7", edge="#6c3483")

draw_arrow(ax, (2.1, 3.0), (3.3, 2.3))
draw_arrow(ax, (2.1, 1.0), (3.3, 1.7))
draw_arrow(ax, (4.15, 2.0), (5.6, 2.0), label="QK^T", offset=(0, 0.25))
draw_arrow(ax, (7.7, 2.0), (9.3, 2.0))

plt.tight_layout(); plt.show()

# %%
d_k = K.shape[1]
scores = Q @ K.T
scaled_scores = scores / np.sqrt(d_k)
R = softmax(scaled_scores, axis=1)

print("Raw scores (Q @ K^T):\n", scores)
print("\nScaled scores (/ sqrt(d_k)):\n", scaled_scores)
print("\nRelevancy Matrix R = softmax(scaled scores):\n", R)
print("\nRow sums (must equal 1):", R.sum(axis=1))

# %%
# Visualize the relevancy matrix as a heatmap
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
labels = ["cow (1)", "brown (2)", "sheep (3)"]
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(labels); ax.set_yticklabels(labels)
ax.set_xlabel("Key tokens"); ax.set_ylabel("Query tokens")
ax.set_title("Relevancy matrix (rows = queries)")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{R[i, j]:.3f}", ha="center", va="center",
                color="white" if R[i, j] > 0.5 else "black")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
ax.set_title("Exercise 4: relevancy of token 1 -> 2 and token 3 -> 2", fontsize=13)

centers = [(1.5, 1.5), (5.0, 1.5), (8.5, 1.5)]
names = [f"x1 = {tuple(X[0])}\ncow",
         f"x2 = {tuple(X[1])}\nbrown",
         f"x3 = {tuple(X[2])}\nsheep"]
colors = ["#fff2cc", "#cfe2ff", "#d5e8d4"]
edges = ["#bf9000", "#1f4e79", "#2d6a4f"]

for (cx, cy), label, c, ec in zip(centers, names, colors, edges):
    draw_box(ax, (cx - 0.9, cy - 0.55), 1.8, 1.1, label, color=c, edge=ec, fontsize=10)

# Arrows from token 1 -> 2 with score R[1, 0]
draw_arrow(ax, (2.4, 1.9), (4.1, 1.9),
           label=f"R[2->1] = {R[1, 0]:.3f}", offset=(0, 0.2))
# Arrow from token 3 -> 2 with score R[1, 2]
draw_arrow(ax, (7.6, 1.9), (5.9, 1.9),
           label=f"R[2->3] = {R[1, 2]:.3f}", offset=(0, 0.2))

plt.tight_layout(); plt.show()

print(f"R[query=token2, key=token1] = {R[1, 0]:.4f}")
print(f"R[query=token2, key=token3] = {R[1, 2]:.4f}")

# %%
Wv = np.array([
    [1.0, 2.0],
    [0.0, 1.0],
])

V = X @ Wv
print("Wv:\n", Wv)
print("\nV = X @ Wv:\n", V)

# %%
# Diagram for Exercise 5
fig, ax = plt.subplots(figsize=(9, 3))
ax.set_xlim(0, 9); ax.set_ylim(0, 3); ax.axis("off")
ax.set_title("Exercise 5: Embedding -> Value space (V)", fontsize=13)

draw_box(ax, (0.5, 1.0), 2.2, 1.0, "Embedding (X)", color="#fff2cc", edge="#bf9000")
draw_box(ax, (6.0, 1.0), 2.2, 1.0, "Value Space (V)", color="#f4cccc", edge="#a61c00")
draw_arrow(ax, (2.7, 1.5), (6.0, 1.5), label="x Wv", offset=(0, 0.25))

plt.tight_layout(); plt.show()

# %%
ATT = R @ V

for i in range(3):
    print(f"ATT[{i+1}] = sum_j R[{i+1}, j] * V[j]")
    parts = [f"{R[i, j]:.3f}*{tuple(V[j])}" for j in range(3)]
    print("        = " + " + ".join(parts))
    print(f"        = {ATT[i]}\n")

print("ATT matrix (rows = tokens):\n", ATT)

# %%
# Diagram for Exercise 6
fig, ax = plt.subplots(figsize=(11, 4))
ax.set_xlim(0, 11); ax.set_ylim(0, 4); ax.axis("off")
ax.set_title("Exercise 6: V and Relevancy -> Sum -> ATT", fontsize=13)

draw_box(ax, (0.3, 2.6), 2.2, 1.0, "V", color="#f4cccc", edge="#a61c00")
draw_box(ax, (0.3, 0.4), 2.2, 1.0, "Relevancy\nMatrix R", color="#e1d5e7", edge="#6c3483")

circ = Circle((5.2, 2.0), 0.55, facecolor="#fde2e4", edgecolor="#9b2226", linewidth=1.5)
ax.add_patch(circ)
ax.text(5.2, 2.0, r"$\sum$", ha="center", va="center", fontsize=24)

draw_box(ax, (8.2, 1.55), 2.2, 0.9, "ATT space", color="#cfe2ff")

draw_arrow(ax, (2.5, 3.1), (4.7, 2.3))
draw_arrow(ax, (2.5, 0.9), (4.7, 1.7))
draw_arrow(ax, (5.75, 2.0), (8.2, 2.0), label="R . V", offset=(0, 0.25))

plt.tight_layout(); plt.show()

# %%
# Diagram for Exercise 7
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
ax.set_title("Exercise 7: ATT -> Feed Forward NN -> Output", fontsize=13)

draw_box(ax, (0.3, 1.0), 2.0, 1.0, "ATT", color="#cfe2ff")

# NN as 2 layers of small circles
for i, y in enumerate([0.6, 1.5, 2.4]):
    ax.add_patch(Circle((4.5, y), 0.22, facecolor="#ffe5b4", edgecolor="#a5673f"))
for i, y in enumerate([1.0, 2.0]):
    ax.add_patch(Circle((6.0, y), 0.22, facecolor="#ffe5b4", edgecolor="#a5673f"))
for x_a in [4.5]:
    for ya in [0.6, 1.5, 2.4]:
        for yb in [1.0, 2.0]:
            ax.plot([x_a + 0.22, 6.0 - 0.22], [ya, yb], color="#a5673f", linewidth=0.6)
ax.text(5.25, 0.15, "Feed Forward NN", ha="center", fontsize=11, color="#a5673f")

draw_box(ax, (7.7, 1.0), 2.0, 1.0, "Output Space", color="#d5e8d4", edge="#2d6a4f")

draw_arrow(ax, (2.3, 1.5), (4.28, 1.5))
draw_arrow(ax, (6.22, 1.5), (7.7, 1.5))

plt.tight_layout(); plt.show()

# %%
# Optional: A tiny feed-forward network applied independently to each ATT vector
rng = np.random.default_rng(seed=0)

W1 = rng.standard_normal((2, 4))   # expand 2 -> 4
b1 = rng.standard_normal(4)
W2 = rng.standard_normal((4, 2))   # back 4 -> 2
b2 = rng.standard_normal(2)

def relu(x):
    return np.maximum(x, 0.0)

hidden = relu(ATT @ W1 + b1)
output = hidden @ W2 + b2

print("ATT input to FFN:\n", ATT)
print("\nHidden activations (after ReLU):\n", hidden)
print("\nFinal output (one row per token):\n", output)

# %%
def scaled_dot_product_attention(X, Wq, Wk, Wv):
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    d_k = K.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)
    R = softmax(scores, axis=1)
    return R @ V, R

ATT_check, R_check = scaled_dot_product_attention(X, Wq, Wk, Wv)
print("Relevancy matrix:\n", R_check)
print("\nATT vectors:\n", ATT_check)
print("\nMatches earlier ATT?", np.allclose(ATT_check, ATT))
