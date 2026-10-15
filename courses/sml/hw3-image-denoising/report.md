# Assignment: Image Denoising using Sparse Coding and OMP

## Course: Sparse Optimization

**Due Date: Two weeks from distribution**

**Submission: One Jupyter notebook (.ipynb) with code, outputs, and answers in Markdown cells**

---

## Overview

In this assignment, you will explore image denoising using sparse coding. You will work with the provided SPORCO example (color image denoising) as your reference, then adapt it to grayscale images and implement an OMP-based solver.

---

## Provided Materials

1. **Reference code**: Complete example of RGB image denoising using BPDN
2. **Two test images**: `sarbaz1.jpg` and `sarbaz2.jpg` (choose ONE to work with)
3. **SPORCO library** with multiple grayscale dictionaries

---

## Learning Objectives

- Understand sparse coding for image denoising
- Visualize dictionary atoms and their coefficients
- Investigate effects of block size and stride (overlap)
- Implement OMP as an alternative sparse coding solver
- Compare different denoising approaches

---

## Part 1: Exploring Dictionary Atoms (15 points)

**Task:** Load two different grayscale dictionaries and visualize their atoms.

```python
from sporco import util

# Load dictionaries
D1 = util.convdicts()['G:8x8x64']   # shape: (8, 8, 64)
D2 = util.convdicts()['G:12x12x64'] # shape: (12, 12, 64)

# Display all atoms (arranged in 8x8 grid for 8x8 dictionary)
```

**Questions to answer (in Markdown):**
1. What patterns do you observe in the dictionary atoms (edges, textures, gradients, smooth regions)?
2. How do the 8x8 atoms differ from 12x12 atoms? Which one has more "structured" patterns?
3. Why might a dictionary need different types of atoms for natural images?

**Deliverable:** Two figures showing dictionary atoms.

---

## Part 2: Single Block Analysis - Atoms and Coefficients (15 points)

**Task:** Select ONE block from the noisy image and show:
1. Which dictionary atoms are selected for representing that block
2. Their corresponding coefficients (weights)

**Implementation steps:**

```python
# Extract a single block from the noisy image (e.g., at position (100, 100))
blksz = (8, 8)
stpsz = (2, 2)  # for extraction, but we'll take just one block

# Extract all blocks
blocks = array.extract_blocks(imgn, blksz, stpsz)
blockmeans = np.mean(blocks, axis=(0,1))
blocks -= blockmeans
blocks = blocks.reshape(np.prod(blksz), -1)

# Select a specific block (e.g., block index 500)
single_block = blocks[:, 500]

# Solve using BPDN or OMP to get coefficients
X_single = b.solve()  # or your OMP solver
coefficients = X_single[:, 500]  # coefficients for this block

# Find indices of top 5 non-zero coefficients
top_indices = np.argsort(np.abs(coefficients))[-5:][::-1]
top_coeffs = coefficients[top_indices]

# Display:
# 1. The original block (reshape to 8x8)
# 2. The top 5 selected dictionary atoms (reshaped to 8x8)
# 3. Their coefficient values
```

**Questions to answer:**
1. What do the selected atoms look like? Are they similar to the block content?
2. How does the linear combination of these atoms reconstruct the block?
3. Show the reconstruction error (difference between original block and reconstructed block).

**Deliverable:** A figure showing: original block, top 5 atoms with their coefficients, and reconstructed block.

---

## Part 3: Block Size and Stride Effects (25 points)

**Task:** Experiment with **4 different configurations** (reduced from 6) to understand the effects of block size and stride.

| Config | Block Size | Stride | Overlap | Dictionary |
|--------|-----------|--------|---------|------------|
| A | (8,8) | (8,8) | 0% | G:8x8x64 |
| B | (8,8) | (2,2) | 75% | G:8x8x64 |
| C | (12,12) | (12,12) | 0% | G:12x12x64 |
| D | (12,12) | (3,3) | 75% | G:12x12x64 |

**For each configuration, you need to:**
1. Denoise the image using the BPDN solver (from the original example)
2. Record PSNR and SSIM values
3. Visually inspect for blocking artifacts

**Code template:**
```python
configs = [
    {'blksz': (8,8), 'stpsz': (8,8), 'dict_key': 'G:8x8x64', 'name': '8x8_no_overlap'},
    {'blksz': (8,8), 'stpsz': (2,2), 'dict_key': 'G:8x8x64', 'name': '8x8_overlap'},
    {'blksz': (12,12), 'stpsz': (12,12), 'dict_key': 'G:12x12x64', 'name': '12x12_no_overlap'},
    {'blksz': (12,12), 'stpsz': (3,3), 'dict_key': 'G:12x12x64', 'name': '12x12_overlap'},
]

results = []
for cfg in configs:
    # Run denoising
    # Store PSNR, SSIM, and reconstructed image
    pass
```

**Questions to answer:**
1. Which configurations produce visible **blocking artifacts**? Why?
2. How does increasing overlap (stride=2 vs stride=8) affect PSNR?
3. Compare configurations A vs B (same block size, different stride). Trade-offs?
4. Compare A vs C (same stride, different block size). Which works better and why?
5. Describe the computational cost differences between these configurations.

**Deliverable:** 
- Table with PSNR/SSIM for all 4 configurations
- Display denoised images side-by-side for visual comparison
- Zoomed-in regions showing artifacts where present

---

## Part 4: Replacing BPDN with OMP (25 points)

**Task:** Implement OMP and replace the BPDN solver. Compare performance.

### OMP Implementation (provided to students):

```python
import numpy as np
from numpy.linalg import pinv

def omp(X, y, k, tol=1e-6):
    """
    Orthogonal Matching Pursuit
    
    Parameters:
    X: design matrix (n_features, n_atoms) - the dictionary
    y: target vector (n_features,) - one image block
    k: maximum sparsity (number of non-zero coefficients)
    tol: residual tolerance for early stopping
    
    Returns:
    w: coefficient vector (n_atoms,)
    """
    n, p = X.shape
    w = np.zeros(p)
    residual = y.copy()
    active = []
    
    for _ in range(k):
        # Find most correlated atom
        correlations = np.abs(X.T @ residual)
        correlations[active] = -np.inf  # exclude already selected
        new_idx = np.argmax(correlations)
        active.append(new_idx)
        
        # Solve least squares on active set
        X_active = X[:, active]
        w_active = pinv(X_active) @ y
        
        # Update residual
        residual = y - X_active @ w_active
        
        # Stop if residual is small
        if np.linalg.norm(residual) < tol:
            break
    
    # Convert to full coefficient vector
    w_sparse = np.zeros(p)
    w_sparse[active] = w_active
    return w_sparse
```

### OMP-Based Solver Class:

```python
class OMP_Denoise:
    def __init__(self, D, blocks, sparsity):
        """
        D: dictionary (n_features, n_atoms)
        blocks: signals to code (n_features, n_blocks)
        sparsity: maximum number of non-zero coefficients (k)
        """
        self.D = D
        self.blocks = blocks
        self.sparsity = sparsity
    
    def solve(self):
        """Returns coefficient matrix X of shape (n_atoms, n_blocks)"""
        n_atoms = self.D.shape[1]
        n_blocks = self.blocks.shape[1]
        X = np.zeros((n_atoms, n_blocks))
        
        for i in range(n_blocks):
            if i % 500 == 0:  # progress indicator
                print(f"Processing block {i}/{n_blocks}")
            X[:, i] = omp(self.D, self.blocks[:, i], self.sparsity)
        
        return X
```

### Experiments:

Compare **three methods** using the best configuration from Part 3:

1. **BPDN** (λ = 0.1) - from original example
2. **OMP-sparse** (k = 5) - very sparse
3. **OMP-medium** (k = 15) - moderately sparse  
4. **OMP-dense** (k = 30) - less sparse

**Record for each:**
- Denoising PSNR
- Computation time (use `time.time()` or `%timeit`)

**Questions to answer:**
1. How does PSNR change as k (sparsity) increases in OMP?
2. What is the relationship between λ in BPDN and k in OMP?
3. Which method is faster? Which gives better quality for your image?
4. If you had to denoise a large batch of images, which method would you choose and why?

**Deliverable:** 
- Table comparing all 4 methods (PSNR, SSIM, time)
- Your observation about the relationship between BPDN's λ and OMP's k

---

## Part 5: Comparison with Traditional Median Filter (10 points)

**Task:** Compare sparse coding with a simple median filter.

```python
from scipy.signal import medfilt2d

# Apply median filter with different window sizes
denoised_med_3 = medfilt2d(noisy_image, kernel_size=3)
denoised_med_5 = medfilt2d(noisy_image, kernel_size=5)
denoised_med_7 = medfilt2d(noisy_image, kernel_size=7)
```

**Test with two noise levels:**
- σ = 0.05 (low noise)
- σ = 0.1 (medium noise - same as previous parts)

**Compare:**
- Best sparse coding result (from Part 3 or Part 4)
- Best median filter result

**Questions to answer:**
1. Create a bar chart comparing PSNR for both noise levels across all methods
2. For which noise level does sparse coding show the biggest improvement over median filtering?
3. Look at the denoised images: where does median filtering fail (e.g., edges, textures)?
4. Based on your results, would you always choose sparse coding over median filtering? Why or why not?

---

## Submission Requirements

### Format:
**Single Jupyter Notebook** with:
1. Code cells that run sequentially (use `Cell → Run All` to verify)
2. Markdown cells answering all questions
3. All figures displayed inline
4. Comments explaining important steps

### Notebook Structure:

```markdown
# Image Denoising with Sparse Coding - Assignment

## Student Name: [Your Name]
## Image Used: sarbaz1.jpg (or sarbaz2.jpg)

## Part 1: Dictionary Atoms
[Code + Output + Answers]

## Part 2: Single Block Analysis
[Code + Output + Answers]

## Part 3: Block Size and Stride Effects
[Code + Output + Table + Answers]

## Part 4: OMP Implementation
[Code + Output + Table + Answers]

## Part 5: Comparison with Median Filter
[Code + Output + Chart + Answers]

## Conclusion
[Summary of key findings]
```

### Required Tables:

**Part 3 Table:**
| Configuration | PSNR (dB) | SSIM | Time (s) | Blocking Artifacts? |
|---------------|-----------|------|----------|---------------------|
| 8x8, stride 8 | | | | |
| 8x8, stride 2 | | | | |
| 12x12, stride 12 | | | | |
| 12x12, stride 3 | | | | |

**Part 4 Table:**
| Method | PSNR (dB) | SSIM | Time (s) |
|--------|-----------|------|----------|
| BPDN (λ=0.1) | | | |
| OMP (k=5) | | | |
| OMP (k=15) | | | |
| OMP (k=30) | | | |

**Part 5 Table (σ=0.05):**
| Method | PSNR (dB) |
|--------|-----------|
| Median 3x3 | |
| Median 5x5 | |
| Median 7x7 | |
| Best Sparse | |

**Part 5 Table (σ=0.1):** (similar)

---

## Important Notes

1. **Fixed random seed**: Always use `np.random.seed(12345)` when adding noise
2. **Choose ONE image** from the two provided
3. **Part 4 may be slow** (OMP loops over blocks). For testing, you can use a subset of blocks, but report the estimated full-image time
4. **Use mean aggregation** for all experiments (simpler than median)
5. **Include all outputs** - your notebook should show the results when opened

---

## Grading Rubric (Total 100 points)

| Section | Points | Criteria |
|---------|--------|----------|
| Part 1 | 15 | Correct atom visualization + thoughtful answers |
| Part 2 | 15 | Single block analysis + atom/coefficient visualization |
| Part 3 | 25 | All 4 configurations tested + artifact identification + complete table |
| Part 4 | 25 | OMP works correctly + comparison with BPDN + relationship analysis |
| Part 5 | 10 | Complete comparison + chart + practical conclusion |
| Code Quality | 10 | Clean, commented, reproducible notebook |

---

## Hints and Tips

1. **For Part 2**: The block index affects which atoms are selected. Try a few different blocks (e.g., smooth region, edge region, texture region) and compare.

2. **For Part 3**: Blocking artifacts are most visible in smooth regions (sky, skin, background). Zoom in on these areas.

3. **For Part 4**: If OMP is too slow:
   ```python
   # Run on first 500 blocks only for testing
   n_blocks_test = min(500, blocks.shape[1])
   blocks_test = blocks[:, :n_blocks_test]
   X_test = omp_solver.solve()
   # Then estimate full time: time_per_block × total_blocks
   ```

4. **For part timing**: Use `%%time` at the beginning of a cell to measure execution time.

---

## Good luck! 🍀

**Remember:** The goal is to understand sparse coding, not just get the highest PSNR. Explain your observations clearly in Markdown cells.