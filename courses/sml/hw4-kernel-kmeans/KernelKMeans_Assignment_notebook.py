# Exported notebook code; may contain IPython commands.
# %%
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state

class KernelKMeans(BaseEstimator, ClusterMixin):
    """
    Kernel K-means

    Reference
    ---------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.
    """

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-3, random_state=None,
                 kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None, verbose=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]

        K = self._get_kernel(X)

        sw = sample_weight if sample_weight is not None else np.ones(n_samples)
        self.sample_weight_ = sw

        rs = check_random_state(self.random_state)
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)

        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)

        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_,
                               update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            # Compute the number of samples whose cluster did not change
            # since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_samples < self.tol:
                if self.verbose:
                    print()
                    print("Converged at iteration", it + 1)
                break

        self.X_fit_ = X

        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the
        kernel trick."""
        sw = self.sample_weight_

        for j in range(self.n_clusters):
            mask = self.labels_ == j

            if np.sum(mask) == 0:
                raise ValueError("Empty cluster found, try smaller n_clusters or different initialization.")

            denom = sw[mask].sum()
            denomsq = denom * denom

            if update_within:
                KK = K[mask][:, mask]
                dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]

            dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom

    def predict(self, X):
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_,
                           update_within=False)
        return dist.argmin(axis=1)

# %%
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score

def plot_clustering_results(X, y_true, y_std, y_kernel, title, kernel_name="Kernel K-Means"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # Ground Truth
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', edgecolor='k', alpha=0.8, s=50)
    axes[0].set_title("Ground Truth Labels", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Feature 1", fontsize=10)
    axes[0].set_ylabel("Feature 2", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Standard K-Means
    ari_std = adjusted_rand_score(y_true, y_std)
    sil_std = silhouette_score(X, y_std) if len(np.unique(y_std)) > 1 else 0.0
    axes[1].scatter(X[:, 0], X[:, 1], c=y_std, cmap='plasma', edgecolor='k', alpha=0.8, s=50)
    axes[1].set_title(f"Standard K-Means\nARI: {ari_std:.4f} | Silh: {sil_std:.4f}", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Feature 1", fontsize=10)
    axes[1].set_ylabel("Feature 2", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # Kernel K-Means
    ari_kernel = adjusted_rand_score(y_true, y_kernel)
    sil_kernel = silhouette_score(X, y_kernel) if len(np.unique(y_kernel)) > 1 else 0.0
    axes[2].scatter(X[:, 0], X[:, 1], c=y_kernel, cmap='coolwarm', edgecolor='k', alpha=0.8, s=50)
    axes[2].set_title(f"{kernel_name}\nARI: {ari_kernel:.4f} | Silh: {sil_kernel:.4f}", fontsize=13, fontweight='bold')
    axes[2].set_xlabel("Feature 1", fontsize=10)
    axes[2].set_ylabel("Feature 2", fontsize=10)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.show()

# %%
# Generate Blobs
X_blobs, y_blobs = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

# Fit Standard K-Means
km_std = KMeans(n_clusters=3, random_state=42, n_init='auto')
y_std_blobs = km_std.fit_predict(X_blobs)

# Fit Kernel K-Means with Linear Kernel
km_kernel_blobs = KernelKMeans(n_clusters=3, kernel="linear", max_iter=100, random_state=42)
y_kernel_blobs = km_kernel_blobs.fit_predict(X_blobs)

# Plot Comparison
plot_clustering_results(X_blobs, y_blobs, y_std_blobs, y_kernel_blobs,
                        title="Experiment 1: Linearly Separable Blobs (Linear Kernel)",
                        kernel_name="Kernel K-Means (Linear)")

# %%
# Generate Concentric Circles
X_circles, y_circles = make_circles(n_samples=400, factor=0.4, noise=0.05, random_state=42)

# Fit Standard K-Means
km_std_circles = KMeans(n_clusters=2, random_state=42, n_init='auto')
y_std_circles = km_std_circles.fit_predict(X_circles)

# Fit Kernel K-Means with RBF Kernel (gamma = 7.0)
km_kernel_circles = KernelKMeans(n_clusters=2, kernel="rbf", gamma=7.0, max_iter=100, random_state=42)
y_kernel_circles = km_kernel_circles.fit_predict(X_circles)

# Plot Comparison
plot_clustering_results(X_circles, y_circles, y_std_circles, y_kernel_circles,
                        title="Experiment 2: Concentric Circles (RBF Kernel)",
                        kernel_name="Kernel K-Means (RBF, gamma=7.0)")

# %%
# Generate Moons
X_moons, y_moons = make_moons(n_samples=400, noise=0.05, random_state=42)

# Fit Standard K-Means
km_std_moons = KMeans(n_clusters=2, random_state=42, n_init='auto')
y_std_moons = km_std_moons.fit_predict(X_moons)

# Fit Kernel K-Means with RBF Kernel (gamma = 12.0)
km_kernel_moons = KernelKMeans(n_clusters=2, kernel="rbf", gamma=12.0, max_iter=100, random_state=42)
y_kernel_moons = km_kernel_moons.fit_predict(X_moons)

# Plot Comparison
plot_clustering_results(X_moons, y_moons, y_std_moons, y_kernel_moons,
                        title="Experiment 3: Interlocking Moons (RBF Kernel)",
                        kernel_name="Kernel K-Means (RBF, gamma=12.0)")

# %%
gammas = [0.1, 2.0, 15.0, 150.0]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, gamma in enumerate(gammas):
    try:
        km_gamma = KernelKMeans(n_clusters=2, kernel="rbf", gamma=gamma, random_state=42)
        y_gamma = km_gamma.fit_predict(X_moons)
        ari = adjusted_rand_score(y_moons, y_gamma)
        
        axes[idx].scatter(X_moons[:, 0], X_moons[:, 1], c=y_gamma, cmap='coolwarm', edgecolor='k', alpha=0.8, s=40)
        axes[idx].set_title(f"RBF Kernel with $\gamma = {gamma}$\nAdjusted Rand Index (ARI): {ari:.4f}", fontsize=12, fontweight='bold')
    except Exception as e:
        axes[idx].text(0.5, 0.5, f"Failed for $\gamma = {gamma}$:\n{str(e)}", 
                      ha='center', va='center', fontsize=12, color='red', weight='bold')
        axes[idx].set_title(f"RBF Kernel with $\gamma = {gamma}$ (Numerical Error)", fontsize=12, fontweight='bold')
        
    axes[idx].grid(True, linestyle='--', alpha=0.5)
    axes[idx].set_xlabel("Feature 1")
    axes[idx].set_ylabel("Feature 2")

plt.suptitle("Hyperparameter Sensitivity: Effect of Gamma on Moons Clustering", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %%
class KernelKMeansWithHistory(KernelKMeans):
    """
    Extended KernelKMeans that tracks the objective function value
    at each iteration for convergence analysis.
    """
    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]
        K = self._get_kernel(X)
        sw = sample_weight if sample_weight is not None else np.ones(n_samples)
        self.sample_weight_ = sw
        rs = check_random_state(self.random_state)
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)
        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)
        self.objective_history_ = []

        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_, update_within=True)
            # Compute objective: sum of min distances for each sample
            obj = sum(dist[i, self.labels_[i]] for i in range(n_samples))
            self.objective_history_.append(obj)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_samples < self.tol:
                break

        self.X_fit_ = X
        return self

# Run convergence analysis on Moons dataset
km_conv = KernelKMeansWithHistory(n_clusters=2, kernel='rbf', gamma=12.0,
                                   max_iter=50, tol=1e-4, random_state=42)
km_conv.fit(X_moons)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(km_conv.objective_history_) + 1), km_conv.objective_history_,
        'b-o', linewidth=2, markersize=6)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Objective Function (Within-Cluster Distance Sum)', fontsize=12)
ax.set_title('Convergence of Kernel K-Means (RBF, $\\gamma=12.0$) on Moons',
             fontsize=14, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# %%
# Test sensitivity to initialization with 10 different random seeds
n_runs = 10
ari_scores_circles = []
ari_scores_moons = []

for seed in range(n_runs):
    # Circles
    km_c = KernelKMeans(n_clusters=2, kernel='rbf', gamma=7.0,
                        max_iter=100, random_state=seed)
    y_pred_c = km_c.fit_predict(X_circles)
    ari_scores_circles.append(adjusted_rand_score(y_circles, y_pred_c))
    
    # Moons
    km_m = KernelKMeans(n_clusters=2, kernel='rbf', gamma=12.0,
                        max_iter=100, random_state=seed)
    y_pred_m = km_m.fit_predict(X_moons)
    ari_scores_moons.append(adjusted_rand_score(y_moons, y_pred_m))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(n_runs), ari_scores_circles, color='steelblue', edgecolor='k')
axes[0].axhline(y=np.mean(ari_scores_circles), color='red', linestyle='--',
               label=f'Mean ARI = {np.mean(ari_scores_circles):.4f}')
axes[0].set_xlabel('Random Seed', fontsize=11)
axes[0].set_ylabel('ARI', fontsize=11)
axes[0].set_title('Initialization Sensitivity - Circles\n'
                  f'(Std: {np.std(ari_scores_circles):.4f})',
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.4)
axes[0].set_ylim([0, 1.1])

axes[1].bar(range(n_runs), ari_scores_moons, color='coral', edgecolor='k')
axes[1].axhline(y=np.mean(ari_scores_moons), color='red', linestyle='--',
               label=f'Mean ARI = {np.mean(ari_scores_moons):.4f}')
axes[1].set_xlabel('Random Seed', fontsize=11)
axes[1].set_ylabel('ARI', fontsize=11)
axes[1].set_title('Initialization Sensitivity - Moons\n'
                  f'(Std: {np.std(ari_scores_moons):.4f})',
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.4)
axes[1].set_ylim([0, 1.1])

plt.suptitle('Sensitivity to Random Initialization', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print(f'Circles - Mean ARI: {np.mean(ari_scores_circles):.4f}, '
      f'Std: {np.std(ari_scores_circles):.4f}')
print(f'Moons   - Mean ARI: {np.mean(ari_scores_moons):.4f}, '
      f'Std: {np.std(ari_scores_moons):.4f}')

# %%
degrees = [2, 3, 4, 5]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, degree in enumerate(degrees):
    try:
        km_poly = KernelKMeans(n_clusters=2, kernel='poly', degree=degree,
                              gamma=1.0, coef0=1, max_iter=100, random_state=42)
        y_poly = km_poly.fit_predict(X_circles)
        ari = adjusted_rand_score(y_circles, y_poly)
        
        axes[idx].scatter(X_circles[:, 0], X_circles[:, 1], c=y_poly,
                         cmap='coolwarm', edgecolor='k', alpha=0.8, s=40)
        axes[idx].set_title(f'Polynomial Kernel (degree={degree})\n'
                           f'ARI: {ari:.4f}', fontsize=12, fontweight='bold')
    except Exception as e:
        axes[idx].text(0.5, 0.5, f'Failed for degree={degree}:\n{str(e)}',
                      ha='center', va='center', fontsize=11, color='red')
        axes[idx].set_title(f'Polynomial Kernel (degree={degree}) - Error',
                           fontsize=12, fontweight='bold')
    axes[idx].grid(True, linestyle='--', alpha=0.5)
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.suptitle('Polynomial Kernel with Different Degrees on Circles',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %%
from sklearn.metrics import normalized_mutual_info_score

def purity_score(y_true, y_pred):
    """Compute purity of clustering."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

# Evaluate all experiments with comprehensive metrics
datasets = {
    'Blobs (Linear)': (y_blobs, y_kernel_blobs),
    'Circles (RBF)': (y_circles, y_kernel_circles),
    'Moons (RBF)': (y_moons, y_kernel_moons)
}

print(f'{"Dataset":<20} {"ARI":<10} {"NMI":<10} {"Purity":<10} {"Silhouette":<10}')
print('=' * 60)

results = []
for name, (y_true, y_pred) in datasets.items():
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    purity = purity_score(y_true, y_pred)
    # Get corresponding X for silhouette
    if 'Blobs' in name:
        X_data = X_blobs
    elif 'Circles' in name:
        X_data = X_circles
    else:
        X_data = X_moons
    sil = silhouette_score(X_data, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    print(f'{name:<20} {ari:<10.4f} {nmi:<10.4f} {purity:<10.4f} {sil:<10.4f}')
    results.append([name, ari, nmi, purity, sil])

# Also compare with Standard K-Means
print('\n--- Standard K-Means for comparison ---')
print(f'{"Dataset":<20} {"ARI":<10} {"NMI":<10} {"Purity":<10} {"Silhouette":<10}')
print('=' * 60)

datasets_std = {
    'Blobs': (y_blobs, y_std_blobs, X_blobs),
    'Circles': (y_circles, y_std_circles, X_circles),
    'Moons': (y_moons, y_std_moons, X_moons)
}

for name, (y_true, y_pred, X_data) in datasets_std.items():
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    purity = purity_score(y_true, y_pred)
    sil = silhouette_score(X_data, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    print(f'{name:<20} {ari:<10.4f} {nmi:<10.4f} {purity:<10.4f} {sil:<10.4f}')

# %%
from sklearn.cluster import SpectralClustering

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# --- Circles ---
# Ground Truth
axes[0, 0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles,
                   cmap='viridis', edgecolor='k', alpha=0.8, s=40)
axes[0, 0].set_title('Circles - Ground Truth', fontsize=12, fontweight='bold')

# Kernel K-Means
axes[0, 1].scatter(X_circles[:, 0], X_circles[:, 1], c=y_kernel_circles,
                   cmap='coolwarm', edgecolor='k', alpha=0.8, s=40)
ari_kkm_c = adjusted_rand_score(y_circles, y_kernel_circles)
axes[0, 1].set_title(f'Circles - Kernel K-Means\nARI: {ari_kkm_c:.4f}',
                     fontsize=12, fontweight='bold')

# Spectral Clustering
sc_circles = SpectralClustering(n_clusters=2, affinity='rbf', gamma=7.0,
                                random_state=42, assign_labels='kmeans')
y_sc_circles = sc_circles.fit_predict(X_circles)
ari_sc_c = adjusted_rand_score(y_circles, y_sc_circles)
axes[0, 2].scatter(X_circles[:, 0], X_circles[:, 1], c=y_sc_circles,
                   cmap='plasma', edgecolor='k', alpha=0.8, s=40)
axes[0, 2].set_title(f'Circles - Spectral Clustering\nARI: {ari_sc_c:.4f}',
                     fontsize=12, fontweight='bold')

# --- Moons ---
# Ground Truth
axes[1, 0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons,
                   cmap='viridis', edgecolor='k', alpha=0.8, s=40)
axes[1, 0].set_title('Moons - Ground Truth', fontsize=12, fontweight='bold')

# Kernel K-Means
axes[1, 1].scatter(X_moons[:, 0], X_moons[:, 1], c=y_kernel_moons,
                   cmap='coolwarm', edgecolor='k', alpha=0.8, s=40)
ari_kkm_m = adjusted_rand_score(y_moons, y_kernel_moons)
axes[1, 1].set_title(f'Moons - Kernel K-Means\nARI: {ari_kkm_m:.4f}',
                     fontsize=12, fontweight='bold')

# Spectral Clustering
sc_moons = SpectralClustering(n_clusters=2, affinity='rbf', gamma=12.0,
                              random_state=42, assign_labels='kmeans')
y_sc_moons = sc_moons.fit_predict(X_moons)
ari_sc_m = adjusted_rand_score(y_moons, y_sc_moons)
axes[1, 2].scatter(X_moons[:, 0], X_moons[:, 1], c=y_sc_moons,
                   cmap='plasma', edgecolor='k', alpha=0.8, s=40)
axes[1, 2].set_title(f'Moons - Spectral Clustering\nARI: {ari_sc_m:.4f}',
                     fontsize=12, fontweight='bold')

for ax in axes.ravel():
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.suptitle('Kernel K-Means vs Spectral Clustering',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %%
import time

sample_sizes = [100, 200, 500, 1000, 2000, 3000]
times_std = []
times_kernel = []

for n in sample_sizes:
    X_test, _ = make_moons(n_samples=n, noise=0.05, random_state=42)
    
    # Standard K-Means
    start = time.time()
    KMeans(n_clusters=2, random_state=42, n_init='auto').fit(X_test)
    times_std.append(time.time() - start)
    
    # Kernel K-Means
    start = time.time()
    KernelKMeans(n_clusters=2, kernel='rbf', gamma=12.0,
                 max_iter=100, random_state=42).fit(X_test)
    times_kernel.append(time.time() - start)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(sample_sizes, times_std, 'b-o', linewidth=2, markersize=7, label='Standard K-Means')
ax.plot(sample_sizes, times_kernel, 'r-s', linewidth=2, markersize=7, label='Kernel K-Means (RBF)')
ax.set_xlabel('Number of Samples (N)', fontsize=12)
ax.set_ylabel('Runtime (seconds)', fontsize=12)
ax.set_title('Runtime Scalability: Standard K-Means vs Kernel K-Means',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print('\nRuntime Summary:')
print(f'{"N":<8} {"Std K-Means (s)":<18} {"Kernel K-Means (s)":<20} {"Ratio":<8}')
print('=' * 55)
for i, n in enumerate(sample_sizes):
    ratio = times_kernel[i] / times_std[i] if times_std[i] > 0 else float('inf')
    print(f'{n:<8} {times_std[i]:<18.4f} {times_kernel[i]:<20.4f} {ratio:<8.1f}x')

# %%
from sklearn.metrics.pairwise import rbf_kernel

# Sort data by true labels for better visualization
sort_idx = np.argsort(y_circles)
X_sorted = X_circles[sort_idx]

gammas_viz = [0.1, 2.0, 7.0, 50.0]
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

for idx, gamma in enumerate(gammas_viz):
    K_matrix = rbf_kernel(X_sorted, gamma=gamma)
    im = axes[idx].imshow(K_matrix, cmap='hot', interpolation='nearest')
    axes[idx].set_title(f'$\\gamma = {gamma}$', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('Sample Index')
    axes[idx].set_ylabel('Sample Index')
    plt.colorbar(im, ax=axes[idx], fraction=0.046)

plt.suptitle('RBF Kernel Matrix Visualization (Circles, sorted by label)',
             fontsize=15, fontweight='bold', y=1.05)
plt.tight_layout()
plt.show()
