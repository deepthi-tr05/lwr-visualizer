// ===================================================
//  LWR Visualizer — Full Interactive Application
//  Implements Locally Weighted Regression in JS
// ===================================================

// ── Data Generation ──────────────────────────────────
let seed = 42;
function seededRandom() {
  seed = (seed * 1664525 + 1013904223) & 0xffffffff;
  return (seed >>> 0) / 4294967296;
}
function resetSeed(s = 42) { seed = s; }

function generateData(sigma = 0.1, n = 100) {
  resetSeed(42);
  const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const xi = -3 + (6 * i) / (n - 1);
    // Box-Muller for Gaussian noise
    const u1 = seededRandom(), u2 = seededRandom();
    const noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    X.push(xi);
    y.push(Math.sin(xi) + sigma * noise);
  }
  return { X, y };
}

// ── LWR Core ─────────────────────────────────────────
function gaussianKernel(x0, X, tau) {
  return X.map(xi => Math.exp(-Math.pow(xi - x0, 2) / (2 * tau * tau)));
}

function lwrPredict(x0, X, y, tau) {
  const w = gaussianKernel(x0, X, tau);
  const n = X.length;
  // Build X1 = [1, x] and solve (X1^T W X1)^-1 X1^T W y
  let a00 = 0, a01 = 0, a11 = 0, b0 = 0, b1 = 0;
  for (let i = 0; i < n; i++) {
    const wi = w[i];
    a00 += wi;
    a01 += wi * X[i];
    a11 += wi * X[i] * X[i];
    b0  += wi * y[i];
    b1  += wi * X[i] * y[i];
  }
  // 2x2 matrix inverse: [a00 a01; a01 a11]
  const det = a00 * a11 - a01 * a01;
  if (Math.abs(det) < 1e-12) return 0;
  const t0 = (a11 * b0 - a01 * b1) / det;
  const t1 = (-a01 * b0 + a00 * b1) / det;
  return t0 + t1 * x0;
}

function predictCurve(X, y, tau, nPoints = 300) {
  const Xt = Array.from({ length: nPoints }, (_, i) => -3 + 6 * i / (nPoints - 1));
  const yp = Xt.map(x => lwrPredict(x, X, y, tau));
  return { Xt, yp };
}

function trueCurve(nPoints = 300) {
  const Xt = Array.from({ length: nPoints }, (_, i) => -3 + 6 * i / (nPoints - 1));
  return Xt.map(x => Math.sin(x));
}

function computeRMSE(X, y, tau) {
  const pred = X.map(xi => lwrPredict(xi, X, y, tau));
  const trueFn = X.map(xi => Math.sin(xi));
  const mse = pred.reduce((s, p, i) => s + Math.pow(p - trueFn[i], 2), 0) / X.length;
  return Math.sqrt(mse).toFixed(4);
}

// ── State ─────────────────────────────────────────────
let state = {
  tau: 0.5,
  sigma: 0.1,
  showTrue: false,
  data: generateData(0.1),
  randomSeedOffset: 0
};

// ── Particle Background ───────────────────────────────
function initParticles() {
  const canvas = document.getElementById('bgCanvas');
  const ctx = canvas.getContext('2d');
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  const particles = Array.from({ length: 60 }, () => ({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    r: Math.random() * 1.5 + 0.5,
    vx: (Math.random() - 0.5) * 0.3,
    vy: (Math.random() - 0.5) * 0.3,
    hue: Math.random() < 0.5 ? 20 : 330
  }));

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${p.hue},90%,70%,0.6)`;
      ctx.fill();
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = canvas.width;
      if (p.x > canvas.width) p.x = 0;
      if (p.y < 0) p.y = canvas.height;
      if (p.y > canvas.height) p.y = 0;
    });
    requestAnimationFrame(draw);
  }
  draw();
  window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  });
}

// ── Mini Chart (Hero Preview) ─────────────────────────
let miniChart = null;
function initMiniChart() {
  const { X, y } = state.data;
  const { Xt, yp } = predictCurve(X, y, 0.5, 100);
  const ctx = document.getElementById('miniChart').getContext('2d');
  miniChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          type: 'scatter',
          label: 'Data',
          data: X.slice(0, 100).map((xi, i) => ({ x: xi, y: y[i] })),
          backgroundColor: 'rgba(244,114,182,0.5)',
          pointRadius: 2.5, borderWidth: 0
        },
        {
          type: 'line',
          label: 'LWR',
          data: Xt.map((xi, i) => ({ x: xi, y: yp[i] })),
          borderColor: '#fb923c',
          borderWidth: 2, pointRadius: 0, tension: 0.4, fill: false
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: { x: { display: false }, y: { display: false } }
    }
  });
}

// ── Main Chart ────────────────────────────────────────
let mainChart = null;
function initMainChart() {
  const { X, y } = state.data;
  const { Xt, yp } = predictCurve(X, y, state.tau);
  const ctx = document.getElementById('mainChart').getContext('2d');

  mainChart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets: buildMainDatasets(X, y, Xt, yp) },
    options: buildMainOptions()
  });
  updateRMSE();
}

function buildMainDatasets(X, y, Xt, yp) {
  const datasets = [
    {
      type: 'scatter',
      label: 'Training Data',
      data: X.map((xi, i) => ({ x: xi, y: y[i] })),
      backgroundColor: 'rgba(244,114,182,0.65)',
      borderColor: 'rgba(244,114,182,0.3)',
      borderWidth: 1,
      pointRadius: 5,
      pointHoverRadius: 7,
      order: 2
    },
    {
      type: 'line',
      label: 'LWR Prediction',
      data: Xt.map((xi, i) => ({ x: xi, y: yp[i] })),
      borderColor: createGradient(),
      borderWidth: 3,
      pointRadius: 0,
      tension: 0.4,
      fill: false,
      order: 1
    }
  ];
  if (state.showTrue) {
    const trueY = trueCurve(300);
    const Xt2 = Array.from({ length: 300 }, (_, i) => -3 + 6 * i / 299);
    datasets.push({
      type: 'line',
      label: 'True sin(x)',
      data: Xt2.map((xi, i) => ({ x: xi, y: trueY[i] })),
      borderColor: 'rgba(148,163,184,0.45)',
      borderWidth: 1.5,
      borderDash: [6, 4],
      pointRadius: 0,
      tension: 0.4,
      fill: false,
      order: 3
    });
  }
  return datasets;
}

function createGradient() {
  // Return as CSS string — Chart.js will use it for stroke
  return '#fb923c';
}

function buildMainOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300, easing: 'easeOutQuart' },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(5,5,16,0.95)',
        borderColor: 'rgba(251,146,60,0.25)',
        borderWidth: 1,
        titleColor: '#f1f1ff',
        bodyColor: '#8888aa',
        padding: 10,
        cornerRadius: 10,
        callbacks: {
          title: (items) => items[0].dataset.label,
          label: (item) => `x: ${item.parsed.x.toFixed(3)}   y: ${item.parsed.y.toFixed(3)}`
        }
      }
    },
    scales: {
      x: {
        title: { display: true, text: 'x', color: '#8888aa', font: { family: 'Inter', size: 12 } },
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#44445a', font: { family: 'JetBrains Mono', size: 10 } }
      },
      y: {
        title: { display: true, text: 'y', color: '#8888aa', font: { family: 'Inter', size: 12 } },
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#44445a', font: { family: 'JetBrains Mono', size: 10 } }
      }
    }
  };
}

function updateMainChart() {
  if (!mainChart) return;
  const { X, y } = state.data;
  const { Xt, yp } = predictCurve(X, y, state.tau);
  mainChart.data.datasets = buildMainDatasets(X, y, Xt, yp);
  mainChart.update('active');
  updateRMSE();
  // Update legend-true visibility
  document.getElementById('legend-true').classList.toggle('hidden', !state.showTrue);
}

function updateRMSE() {
  const { X, y } = state.data;
  const rmse = computeRMSE(X, y, state.tau);
  document.getElementById('rmseDisplay').textContent = `RMSE: ${rmse}`;
}

// ── Kernel Canvases (Theory Section) ─────────────────
function drawKernelCanvas(canvasId, tau, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let x = 0; x <= W; x += W / 4) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }

  // Gaussian curve
  ctx.beginPath();
  const x0 = 0; // center
  for (let px = 0; px <= W; px++) {
    const xi = -3 + 6 * px / W;
    const weight = Math.exp(-Math.pow(xi - x0, 2) / (2 * tau * tau));
    const py = H - weight * (H - 20) - 10;
    if (px === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  const grad = ctx.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0, 'rgba(244,114,182,0.2)');
  grad.addColorStop(0.5, color);
  grad.addColorStop(1, 'rgba(250,204,21,0.2)');
  ctx.strokeStyle = grad;
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Fill
  ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
  ctx.fillStyle = color.replace(')', ',0.08)').replace('rgb', 'rgba');
  ctx.fill();

  // Label
  ctx.fillStyle = '#8888aa';
  ctx.font = "11px 'JetBrains Mono'";
  ctx.fillText(`τ = ${tau}`, 8, 16);
}

// ── Controls ──────────────────────────────────────────
function setupControls() {
  const tauSlider = document.getElementById('tauSlider');
  const sigmaSlider = document.getElementById('sigmaSlider');

  tauSlider.addEventListener('input', () => {
    state.tau = parseFloat(tauSlider.value);
    document.getElementById('tauDisplay').textContent = state.tau.toFixed(2);
    updateMainChart();
  });

  sigmaSlider.addEventListener('input', () => {
    state.sigma = parseFloat(sigmaSlider.value);
    document.getElementById('sigmaDisplay').textContent = state.sigma.toFixed(2);
    state.data = generateData(state.sigma);
    updateMainChart();
  });

  document.getElementById('regenerateBtn').addEventListener('click', () => {
    seed = Math.floor(Math.random() * 999999);
    state.data = generateData(state.sigma);
    updateMainChart();
    // Bounce animation
    const btn = document.getElementById('regenerateBtn');
    btn.style.transform = 'rotate(360deg)';
    btn.style.transition = 'transform 0.5s ease';
    setTimeout(() => { btn.style.transform = ''; btn.style.transition = ''; }, 500);
  });

  document.getElementById('showTrueBtn').addEventListener('click', () => {
    state.showTrue = !state.showTrue;
    document.getElementById('showTrueBtn').classList.toggle('active-true', state.showTrue);
    updateMainChart();
  });

  document.getElementById('resetBtn').addEventListener('click', () => {
    state.tau = 0.5; state.sigma = 0.1; state.showTrue = false;
    tauSlider.value = 0.5; sigmaSlider.value = 0.1;
    document.getElementById('tauDisplay').textContent = '0.50';
    document.getElementById('sigmaDisplay').textContent = '0.10';
    document.getElementById('showTrueBtn').classList.remove('active-true');
    resetSeed(42);
    state.data = generateData(0.1);
    updateMainChart();
  });
}

// Global function for BV card buttons
function setTau(val) {
  state.tau = val;
  document.getElementById('tauSlider').value = val;
  document.getElementById('tauDisplay').textContent = val.toFixed(2);
  updateMainChart();
  document.getElementById('demo-section').scrollIntoView({ behavior: 'smooth' });
}

// ── Intersection Animations ───────────────────────────
function setupAnimations() {
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((entry, i) => {
      if (entry.isIntersecting) {
        setTimeout(() => entry.target.classList.add('visible'), i * 80);
        obs.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });
  document.querySelectorAll('.bv-card, .theory-card').forEach(el => obs.observe(el));
}

// ── Nav Highlight ─────────────────────────────────────
function setupNavHighlight() {
  const ids = ['hero', 'demo-section', 'theory-section', 'code-section'];
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        const map = { 'hero': 'nav-home', 'demo-section': 'nav-demo', 'theory-section': 'nav-theory', 'code-section': 'nav-code' };
        const el = document.getElementById(map[e.target.id]);
        if (el) el.classList.add('active');
      }
    });
  }, { threshold: 0.4 });
  ids.forEach(id => { const el = document.getElementById(id); if (el) obs.observe(el); });
}

// ── Copy Code ─────────────────────────────────────────
function copyCode() {
  const code = `import numpy as np
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
plt.show()`;

  navigator.clipboard.writeText(code).then(() => {
    const btn = document.getElementById('copyBtn');
    btn.innerHTML = '✓ Copied!';
    btn.style.color = '#4ade80';
    setTimeout(() => {
      btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy';
      btn.style.color = '';
    }, 2000);
  });
}

// ── INIT ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  initMiniChart();
  initMainChart();
  setupControls();
  setupAnimations();
  setupNavHighlight();

  // Draw kernel preview canvases
  setTimeout(() => {
    drawKernelCanvas('kernel1', 0.2, '#f472b6');
    drawKernelCanvas('kernel2', 0.5, '#fb923c');
    drawKernelCanvas('kernel3', 1.5, '#facc15');
  }, 200);
});
