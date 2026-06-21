import numpy as np
import matplotlib.pyplot as plt

# ── Reproducible seed ─────────────────────────────────
np.random.seed(0)

# ── Generate training data: sin(x) + Gaussian noise ──
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, 100)

# ── Gaussian kernel function ──────────────────────────
def kernel(x0, X, t):
    """Compute Gaussian kernel weights for query point x0."""
    return np.exp(-np.sum((X - x0)**2, axis=1) / (2 * t**2))

# ── LWR prediction at a single query point ────────────
def predict(x0, t=0.5):
    """Locally Weighted Regression prediction at x0."""
    W     = np.diag(kernel(x0, X, t))          # Diagonal weight matrix
    X1    = np.c_[np.ones(len(X)), X]           # Add bias column [1, x]
    theta = np.linalg.pinv(X1.T @ W @ X1) @ X1.T @ W @ y  # Weighted normal eqn
    return np.r_[1, x0] @ theta                 # Local prediction

# ── Generate predictions over dense test grid ─────────
Xt = np.linspace(-3, 3, 300).reshape(-1, 1)
yp = np.array([predict(x) for x in Xt])

# ── Compute RMSE ──────────────────────────────────────
y_true = np.sin(Xt).ravel()
rmse = np.sqrt(np.mean((yp - y_true)**2))
print(f"RMSE vs true sin(x): {rmse:.4f}")

# ── Plot ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#050510')
ax.set_facecolor('#0b0b1e')

ax.scatter(X, y, color='#f472b6', s=50, alpha=0.7,
           edgecolors='rgba(244,114,182,0.3)', linewidths=0.5,
           label='Training data (sin(x) + noise)')

ax.plot(Xt, yp, color='#fb923c', linewidth=2.5, label='LWR fit (τ=0.5)')
ax.plot(Xt, y_true, color='#94a3b8', linewidth=1.2,
        linestyle='--', alpha=0.5, label='True sin(x)')

ax.set_xlabel('x', color='#8888aa', fontsize=12)
ax.set_ylabel('y', color='#8888aa', fontsize=12)
ax.set_title('Locally Weighted Regression on sin(x) + noise',
             color='#f1f1ff', fontsize=15, fontweight='bold', pad=15)
ax.legend(framealpha=0.2, facecolor='#0b0b1e',
          edgecolor='#333355', labelcolor='#ccccdd', fontsize=10)
ax.tick_params(colors='#44445a')
for spine in ax.spines.values():
    spine.set_edgecolor('#1e1e3a')
ax.grid(True, alpha=0.07, color='white')

plt.tight_layout()
plt.savefig('lwr_output.png', dpi=150, bbox_inches='tight', facecolor='#050510')
plt.show()
print("Saved as lwr_output.png")
