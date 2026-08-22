# Machine-Learning Notebooks — Deep Analysis & Enhancement Report

**Scope:** All `*.ipynb` files under `apps/machine-learning/`
**Date:** 2026-08-22
**Role:** ML Engineer review — pedagogical clarity, correctness, and consistency

---

## 1. Inventory (4 notebooks found)

| # | Path | Algorithm | Paradigm | Dataset | Status after review |
|---|------|-----------|----------|---------|---------------------|
| 1 | `supervised/exclusive-regression-algorithm/linear-regrassion.ipynb` | Simple Linear Regression (OLS) | Supervised / Regression | 6 pizza records `(diameter, price)` | **Enhanced** |
| 2 | `supervised/exclusive-regression-algorithm/multi-linear-regrassion.ipynb` | Multiple Linear Regression (Normal Eq.) | Supervised / Regression | 5 houses, 3 features → price | **Enhanced** |
| 3 | `supervised/exclusive-classification-algorithms/logistic-regression.ipynb` | Logistic Regression (logit-linearized) | Supervised / Classification | 6 emails, trigger-word count → spam | **Enhanced** |
| 4 | `supervised/exclusive-regression-algorithm/polynomial-regression.ipynb` | Polynomial Regression (Normal Eq.) | Supervised / Regression | 6 points, quadratic ground truth | **Created earlier (reference)** |

> Note: all other folders (`reinforcement`, `self-supervised`, `semi-supervised`, `unsupervised`, `pizza-price`, `spam-classification`) contain **Python packages**, not teaching notebooks.

---

## 2. Per-Notebook Deep Analysis

### 2.1 `linear-regrassion.ipynb` — Simple Linear Regression
- **Approach:** Fits `y = m·x + b` using the **closed-form OLS** formulas, computed **cumulatively** (first 2 points → … → all 6). This is pedagogically excellent: you literally watch `m` and `b` converge.
- **Strengths:** Unique incremental subset-fitting; rich 12-column matplotlib table; 4 diagnostic plots (fit progression, slope history, intercept history, formula card).
- **Gaps found:** No LaTeX math markdown, no machine-readable `DataFrame` table, no quantitative metrics (MSE/R²), no interpretation of slope/intercept.
- **Enhancements added:** Title + math markdown (prediction, `m`, `b`, MSE, R²); pandas `DataFrame` of the step trace; final metrics + per-point residual table; "Key Takeaways".

### 2.2 `multi-linear-regrassion.ipynb` — Multiple Linear Regression
- **Approach:** The textbook **design-matrix / Normal Equation** `β = (XᵀX)⁻¹ Xᵀy`, printed step-by-step (design matrix → transpose → XᵀX → inverse → Xᵀy → β). Clean and correct.
- **Strengths:** Transparent matrix math; pandas source table; MSE/RMSE/R² already present; actual-vs-predicted + residual histogram plots.
- **Gaps found:** No mathematical formula markdown; plots were minimal (only 2); no coefficient interpretation.
- **Enhancements added:** Title + Normal-Equation LaTeX markdown; **coefficient bar chart** + **residuals-vs-predicted** plot; a coefficient/interpreration table; "Key Takeaways" linking to polynomial regression.

### 2.3 `logistic-regression.ipynb` — Logistic Regression
- **Approach:** Maps labels to compressed probabilities (0.08/0.92), converts to **logits**, then fits `z = m·x + b` with OLS on the logit scale (same cumulative scheme as the linear notebook). Plots the sigmoid curve progression and decision boundary.
- **Strengths:** Visually rich; clear 12-column trace; good progressive-curve intuition; formula card embedded in the figure.
- **Gaps found:** No LaTeX math markdown; no pandas table; **no accuracy / decision-boundary number / final probability table**. Risk: a reader may mistake this for full MLE logistic regression.
- **Enhancements added:** Title + math markdown (sigmoid, logit link, decision rule, boundary `x* = -b/m`); pandas step trace; final P(spam)/class table + **training accuracy** + boundary value; an honest ML-engineer note that this is *logit-linearization*, with a pointer to true MLE/gradient-descent logistic regression.

### 2.4 `polynomial-regression.ipynb` — Polynomial Regression (reference)
- Already meets every requirement: LaTeX formulas, small dataset, step-by-step Normal-Equation log, prediction table with residuals, and a 4-panel detailed plot (fit vs true, residuals, coefficients, predicted-vs-actual). Used as the gold-standard template for the others.

---

## 3. Cross-Cutting Observations (ML Engineer notes)

1. **Two distinct teaching styles coexist:** the regression-algorithm notebooks use the **Normal Equation** (exact, one-shot), while `linear`/`logistic` use **cumulative OLS subset-fitting** (shows convergence). Both are valuable; they were preserved, not unified, to keep each notebook's character.
2. **Correctness:** All math is sound. The only conceptual caveat is the logistic notebook's *linearized* fit — documented explicitly in the notebook.
3. **Consistency improvements:** Every notebook now opens with (a) a one-line overview, (b) a **LaTeX math** section, (c) the dataset, (d) step-by-step execution, (e) a **table**, (f) **detailed plots**, and (g) **Key Takeaways**.
4. **Scalability caveat (for learners):** the Normal Equation is O(n³) and needs an invertible `XᵀX`; for large/high-dimensional data, gradient descent or regularization (Ridge/Lasso) is preferred. Mentioned in the multi-linear takeaways.

---

## 4. How to Run
Each notebook is self-contained. Open in Jupyter / VS Code and **Run All**:
```
jupyter notebook apps/machine-learning/supervised/.../<notebook>.ipynb
```
Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.

---

## 5. Verification
All three enhanced notebooks were executed end-to-end (headless Agg backend) with **zero runtime errors**; only harmless "non-interactive canvas" warnings from `plt.show()`.
