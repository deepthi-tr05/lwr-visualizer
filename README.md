# 🔔 Locally Weighted Regression Visualizer

<div align="center">

![LWR Visualizer](https://img.shields.io/badge/ML-Locally%20Weighted%20Regression-fb923c?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)
![Vercel](https://img.shields.io/badge/Live%20Demo-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)

**An interactive web visualizer for Locally Weighted Regression (LWR / LOESS).**  
Adjust the Gaussian kernel bandwidth **τ** in real-time and watch the model shift between overfitting and underfitting.

[🚀 Live Demo](https://lwr-visualizer.vercel.app) • [📊 Try the Visualizer](https://lwr-visualizer.vercel.app/#demo-section) • [💻 Source Code](https://github.com/deepthi-tr05/lwr-visualizer)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎛️ **Real-time τ slider** | Drag to change bandwidth and instantly see the curve update |
| 📊 **Noise control (σ)** | Adjust Gaussian noise level on training data |
| 🔄 **Regenerate data** | Generate fresh random datasets with one click |
| 👁️ **True curve overlay** | Toggle the true `sin(x)` function for comparison |
| 📐 **Live RMSE** | Computes RMSE vs true `sin(x)` in real time |
| 🔔 **Kernel visualizer** | Shows the Gaussian kernel shape for narrow/balanced/wide τ |
| ⚡ **Bias-Variance demo** | One-click presets for underfitting, optimal, and overfitting |
| 📋 **Copy source code** | One-click copy of the full Python implementation |
| 🌌 **Particle background** | Animated floating particles for visual polish |

---

## 🖥️ Live Demo

👉 **[https://lwr-visualizer.vercel.app](https://lwr-visualizer.vercel.app)**

---

## 🔬 What is Locally Weighted Regression?

LWR (also called **LOESS / LOWESS**) is a **non-parametric regression** technique. Instead of fitting one global model, it fits a **separate weighted linear model at every query point** using nearby training data.

### Algorithm Steps

```
1. Pick a query point x₀
2. Assign weights using Gaussian kernel:
      w(i) = exp(−‖x⁽ⁱ⁾ − x₀‖² / 2τ²)
3. Solve weighted normal equations:
      θ = (XᵀWX)⁻¹ XᵀWy
4. Predict:  ŷ(x₀) = [1, x₀]ᵀ · θ
5. Repeat for every test point
```

### Bandwidth Parameter τ

| τ Value | Effect | Bias | Variance |
|---|---|---|---|
| τ < 0.2 | Overfitting — fits noise | Low | **High** |
| τ ≈ 0.5 | **Optimal** — smooth curve | Balanced | Balanced |
| τ > 1.5 | Underfitting — too smooth | **High** | Low |

---

## 🐍 Python Source Code

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, 100)

def kernel(x0, X, t):
    return np.exp(-np.sum((X - x0)**2, axis=1) / (2 * t**2))

def predict(x0, t=0.5):
    W     = np.diag(kernel(x0, X, t))
    X1    = np.c_[np.ones(len(X)), X]
    theta = np.linalg.pinv(X1.T @ W @ X1) @ X1.T @ W @ y
    return np.r_[1, x0] @ theta

Xt = np.linspace(-3, 3, 300).reshape(-1, 1)
yp = np.array([predict(x) for x in Xt])

plt.scatter(X, y)
plt.plot(Xt, yp)
plt.show()
```

---

## 🚀 Run Locally (Python)

```bash
# Install dependencies
pip install numpy matplotlib

# Run
python lwr.py
```

---

## 🌐 Run Locally (Web App)

```bash
# Clone the repo
git clone https://github.com/deepthi-tr05/lwr-visualizer.git
cd lwr-visualizer

# Open in browser (no build step needed — pure HTML/CSS/JS)
start index.html        # Windows
open index.html         # macOS
xdg-open index.html     # Linux
```

---

## 📁 Project Structure

```
lwr-visualizer/
├── index.html        # Full web app (4 sections)
├── style.css         # Dark glassmorphism design + animations
├── app.js            # LWR implemented in JS + Chart.js + particle BG
├── lwr.py            # Original Python script (NumPy + Matplotlib)
├── requirements.txt  # Python dependencies
├── vercel.json       # Static site config for Vercel deployment
└── README.md         # This file
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Structure | HTML5 (Semantic) |
| Styling | Vanilla CSS3 (Glassmorphism, Animations) |
| Interactivity | Vanilla JavaScript ES6+ |
| Charts | Chart.js 4.4 |
| ML Algorithm | LWR implemented from scratch in JS |
| Fonts | Inter + JetBrains Mono (Google Fonts) |
| Python ML | NumPy, Matplotlib |
| Hosting | Vercel (static) |

---

## 📊 Results

Running `lwr.py` with default settings (τ = 0.5, σ = 0.1):

| Metric | Value |
|---|---|
| Training points | 100 |
| Test points | 300 |
| Bandwidth τ | 0.5 |
| Noise σ | 0.1 |
| RMSE vs sin(x) | ~0.08 |

---

## 🔗 References

- [Andrew Ng — CS229 Lecture Notes (LWR)](https://cs229.stanford.edu/notes2022fall/main_notes.pdf)
- [Wikipedia — Local Regression (LOESS)](https://en.wikipedia.org/wiki/Local_regression)
- [NumPy Documentation](https://numpy.org/doc/)
- [Chart.js Documentation](https://www.chartjs.org/docs/)

---

## 👩‍💻 Author

**Deepthi** — [@deepthi-tr05](https://github.com/deepthi-tr05)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
Made with ❤️ | Algorithm: LWR/LOESS | Data: sin(x) + Gaussian noise
</div>
