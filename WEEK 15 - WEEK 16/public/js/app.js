// ─── Week 15 & 16: Neural Networks & Model Deployment Studio ────────────────

// Tab Switching
const navButtons = document.querySelectorAll('.nav-btn');
const tabPanes = document.querySelectorAll('.tab-content');

navButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.getAttribute('data-tab');
    navButtons.forEach(b => b.classList.remove('active'));
    tabPanes.forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const pane = document.getElementById(`pane-${target}`);
    if (pane) pane.classList.add('active');

    // Trigger chart redraws on tab switch
    if (target === 'perceptrons') drawCurrentActivation();
    if (target === 'optimizers' && cachedOptData) renderOptimizerCharts(cachedOptData);
  });
});

// Color Tokens
const COLORS = {
  emerald: '#10b981',
  sky: '#38bdf8',
  violet: '#8b5cf6',
  amber: '#f59e0b',
  rose: '#f43f5e',
  textMuted: '#64748b',
  border: 'rgba(255, 255, 255, 0.08)'
};

// ═══════════════════════════════════════════════════════════════════════════════
// 1. MODEL DEPLOYMENT (PICKLE VS JOBLIB)
// ═══════════════════════════════════════════════════════════════════════════════

const elBenchGrid = document.getElementById('deployment-bench-grid');
const formInference = document.getElementById('inference-form');
const elPredResult = document.getElementById('prediction-result-box');

async function loadDeploymentBenchmark() {
  try {
    const res = await fetch('/api/deployment/benchmark');
    const json = await res.json();
    if (json.success) {
      renderDeploymentCards(json.data);
    }
  } catch (err) {
    elBenchGrid.innerHTML = `<div style="color: var(--accent-rose);">Error loading benchmark: ${err.message}</div>`;
  }
}

function renderDeploymentCards(data) {
  const p = data.pickle;
  const j = data.joblib;

  elBenchGrid.innerHTML = `
    <div class="bench-card" style="border-top: 3px solid var(--accent-amber);">
      <div class="bench-card-title">
        <span>🥒 Python Native Pickle</span>
        <span class="hero-pill" style="color: #fbbf24;">Standard Library</span>
      </div>
      <p style="font-size: 0.82rem; color: var(--text-secondary); margin-bottom: 1rem;">
        Default Python object serialization protocol. Works universally across standard Python environments.
      </p>
      <div class="metric-stat"><span class="metric-label">Serialization Time</span><span class="metric-value" style="color: #fbbf24;">${p.save_time_ms} ms</span></div>
      <div class="metric-stat"><span class="metric-label">Deserialization (Load)</span><span class="metric-value" style="color: #6ee7b7;">${p.load_time_ms} ms</span></div>
      <div class="metric-stat"><span class="metric-label">Artifact File Size</span><span class="metric-value" style="color: #38bdf8;">${p.size_kb} KB</span></div>
      <div class="metric-stat"><span class="metric-label">Saved Artifact Path</span><span class="metric-value" style="font-size: 0.72rem; word-break: break-all;">${p.path}</span></div>
    </div>

    <div class="bench-card" style="border-top: 3px solid var(--accent-emerald);">
      <div class="bench-card-title">
        <span>⚡ Joblib (Compressed)</span>
        <span class="hero-pill" style="color: #6ee7b7;">Optimized for NumPy/ML</span>
      </div>
      <p style="font-size: 0.82rem; color: var(--text-secondary); margin-bottom: 1rem;">
        Optimized specifically for NumPy arrays, Scikit-Learn estimators, and disk compression (level 3).
      </p>
      <div class="metric-stat"><span class="metric-label">Serialization Time</span><span class="metric-value" style="color: #fbbf24;">${j.save_time_ms} ms</span></div>
      <div class="metric-stat"><span class="metric-label">Deserialization (Load)</span><span class="metric-value" style="color: #6ee7b7;">${j.load_time_ms} ms</span></div>
      <div class="metric-stat"><span class="metric-label">Artifact File Size</span><span class="metric-value" style="color: #38bdf8;">${j.size_kb} KB (Compressed)</span></div>
      <div class="metric-stat"><span class="metric-label">Saved Artifact Path</span><span class="metric-value" style="font-size: 0.72rem; word-break: break-all;">${j.path}</span></div>
    </div>
  `;
}

// Live Deployed Prediction
formInference.addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('btn-predict');
  const originalText = btn.innerHTML;
  btn.innerHTML = `<span class="spinner"></span> Running inference...`;
  btn.disabled = true;

  const payload = {
    backend: document.getElementById('deploy-backend').value,
    features: {
      Age: document.getElementById('input-age').value,
      StudyHoursPerWeek: document.getElementById('input-study').value,
      AttendanceRate: document.getElementById('input-attendance').value,
      MathScore: document.getElementById('input-math').value,
      ReadingScore: document.getElementById('input-reading').value,
      WritingScore: document.getElementById('input-writing').value,
      Department: document.getElementById('input-dept').value,
      Gender: document.getElementById('input-gender').value
    }
  };

  try {
    const res = await fetch('/api/deployment/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const json = await res.json();
    if (json.success) {
      renderPredictionResult(json.data);
    }
  } catch (err) {
    elPredResult.innerHTML = `<div style="color: var(--accent-rose);">Inference error: ${err.message}</div>`;
    elPredResult.style.display = 'block';
  } finally {
    btn.innerHTML = originalText;
    btn.disabled = false;
  }
});

function renderPredictionResult(data) {
  const isDistinction = data.prediction_code === 1;
  const statusColor = isDistinction ? 'var(--accent-emerald)' : 'var(--accent-rose)';

  elPredResult.innerHTML = `
    <div class="bench-card" style="border-left: 4px solid ${statusColor};">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
        <span style="font-family: var(--font-heading); font-size: 1.2rem; font-weight: 700; color: ${statusColor};">
          ${isDistinction ? '🎖️ Predicted Distinction' : '📘 Predicted Regular'}
        </span>
        <span class="hero-pill" style="color: var(--accent-sky);">Latency: ${data.inference_latency_ms} ms</span>
      </div>
      <p style="font-size: 0.88rem; margin-bottom: 1rem;">
        Model deserialized using <strong>${data.backend_used.toUpperCase()}</strong> directly from disk artifact.
      </p>

      <div style="margin-bottom: 0.5rem; display: flex; justify-content: space-between; font-size: 0.84rem;">
        <span>Distinction Probability: <strong style="color: #6ee7b7;">${data.probability_distinction}%</strong></span>
        <span>Regular Probability: <strong style="color: #94a3b8;">${data.probability_regular}%</strong></span>
      </div>

      <div style="width: 100%; height: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 999px; overflow: hidden;">
        <div style="height: 100%; width: ${data.probability_distinction}%; background: ${statusColor}; border-radius: 999px; transition: width 0.4s ease;"></div>
      </div>
    </div>
  `;
  elPredResult.style.display = 'block';
}


// ═══════════════════════════════════════════════════════════════════════════════
// 2. PERCEPTRONS & ACTIVATIONS
// ═══════════════════════════════════════════════════════════════════════════════

const elGatesGrid = document.getElementById('logic-gates-grid');
let activationData = null;
let currentAct = 'Sigmoid';

async function loadPerceptronGates() {
  try {
    const res = await fetch('/api/perceptron/gates');
    const json = await res.json();
    if (json.success) {
      renderLogicGates(json.data);
    }
  } catch (err) {
    elGatesGrid.innerHTML = `<div style="color: var(--accent-rose);">Error loading logic gates: ${err.message}</div>`;
  }
}

function renderLogicGates(gates) {
  const keys = Object.keys(gates);
  elGatesGrid.innerHTML = keys.map(k => {
    const g = gates[k];
    const isConv = g.converged;
    const badgeColor = isConv ? '#6ee7b7' : '#fb7185';

    return `
      <div class="bench-card" style="border-top: 3px solid ${badgeColor};">
        <div class="bench-card-title">
          <span>${k} Gate</span>
          <span class="hero-pill" style="color: ${badgeColor};">${isConv ? 'Converged' : 'Failed (Non-linear)'}</span>
        </div>
        <p style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 0.5rem;">${g.description}</p>
        <div class="metric-stat"><span class="metric-label">Accuracy</span><span class="metric-value" style="color: ${badgeColor};">${g.accuracy}%</span></div>
        <div class="metric-stat"><span class="metric-label">Epochs</span><span class="metric-value">${g.epochs}</span></div>
        <div class="metric-stat"><span class="metric-label">Weights (w1, w2)</span><span class="metric-value">${g.final_weights.join(', ')}</span></div>
        <div class="metric-stat"><span class="metric-label">Bias (b)</span><span class="metric-value">${g.final_bias}</span></div>

        <table class="truth-table">
          <thead>
            <tr><th>x1</th><th>x2</th><th>Target</th><th>Perceptron</th></tr>
          </thead>
          <tbody>
            ${g.truth_table.map(row => `
              <tr>
                <td>${row.x1}</td>
                <td>${row.x2}</td>
                <td><strong>${row.target}</strong></td>
                <td style="color: ${row.target === row.pred ? '#6ee7b7' : '#fb7185'}; font-weight: 600;">${row.pred}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  }).join('');
}

// Activations Explorer
const actButtons = document.querySelectorAll('.act-btn');
const canvasAct = document.getElementById('canvas-activation');
const ctxAct = canvasAct.getContext('2d');
const elActTitle = document.getElementById('act-chart-title');
const elActInfo = document.getElementById('act-info-box');

actButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    actButtons.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentAct = btn.getAttribute('data-act');
    drawCurrentActivation();
  });
});

async function loadActivationCurves() {
  try {
    const res = await fetch('/api/activations/curves');
    const json = await res.json();
    if (json.success) {
      activationData = json.data;
      drawCurrentActivation();
    }
  } catch (err) {
    console.error('Error loading activations:', err);
  }
}

function drawCurrentActivation() {
  if (!activationData) return;
  const data = activationData.functions[currentAct];
  const z = activationData.z;
  elActTitle.textContent = `${currentAct} Activation & Analytical Derivative`;

  elActInfo.innerHTML = `
    <div class="bench-card-title" style="color: var(--accent-sky);">${currentAct} Formulation</div>
    <div class="code-box" style="margin-bottom: 0.75rem;">${data.formula}</div>
    <div class="bench-card-title" style="color: var(--accent-emerald); font-size: 0.95rem;">Backpropagation Derivative</div>
    <div class="code-box" style="margin-bottom: 0.75rem;">${data.derivative_formula}</div>
    <div class="metric-stat"><span class="metric-label">Output Range</span><span class="metric-value" style="color: #fbbf24;">${data.range}</span></div>
    <div style="margin-top: 0.75rem; font-size: 0.82rem; color: var(--text-secondary);">
      <strong>Use Case:</strong> ${data.use_case}
    </div>
  `;

  // Draw on Canvas
  const w = canvasAct.width;
  const h = canvasAct.height;
  ctxAct.clearRect(0, 0, w, h);

  const padLeft = 45;
  const padRight = 20;
  const padTop = 30;
  const padBottom = 35;
  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  // Find min/max for y
  const allVals = data.values.concat(data.derivatives);
  let minY = Math.min(-0.2, ...allVals);
  let maxY = Math.max(1.2, ...allVals);
  if (currentAct === 'ReLU' || currentAct === 'Leaky ReLU') {
    maxY = 5.2;
    minY = -0.5;
  }
  if (currentAct === 'Tanh') {
    minY = -1.2;
    maxY = 1.2;
  }

  // Draw Grid & Axes
  ctxAct.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctxAct.lineWidth = 1;
  ctxAct.beginPath();
  ctxAct.strokeRect(padLeft, padTop, plotW, plotH);

  // Zero-lines
  const zeroX = padLeft + ((0 - (-5)) / 10) * plotW;
  ctxAct.beginPath();
  ctxAct.moveTo(zeroX, padTop);
  ctxAct.lineTo(zeroX, padTop + plotH);
  ctxAct.strokeStyle = 'rgba(255, 255, 255, 0.15)';
  ctxAct.stroke();

  const zeroY = padTop + plotH - ((0 - minY) / (maxY - minY)) * plotH;
  if (zeroY >= padTop && zeroY <= padTop + plotH) {
    ctxAct.beginPath();
    ctxAct.moveTo(padLeft, zeroY);
    ctxAct.lineTo(padLeft + plotW, zeroY);
    ctxAct.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctxAct.stroke();
  }

  // Function plotting helper
  function plotLine(values, strokeColor) {
    ctxAct.beginPath();
    ctxAct.strokeStyle = strokeColor;
    ctxAct.lineWidth = 2.5;

    for (let i = 0; i < z.length; i++) {
      const px = padLeft + ((z[i] - (-5)) / 10) * plotW;
      const py = padTop + plotH - ((values[i] - minY) / (maxY - minY)) * plotH;
      if (i === 0) ctxAct.moveTo(px, py);
      else ctxAct.lineTo(px, py);
    }
    ctxAct.stroke();
  }

  plotLine(data.values, '#38bdf8'); // blue
  plotLine(data.derivatives, '#10b981'); // green

  // Axis labels
  ctxAct.fillStyle = '#64748b';
  ctxAct.font = '10px Inter, sans-serif';
  ctxAct.fillText('-5', padLeft, padTop + plotH + 18);
  ctxAct.fillText('0', zeroX - 3, padTop + plotH + 18);
  ctxAct.fillText('+5 (z)', padLeft + plotW - 25, padTop + plotH + 18);
  ctxAct.fillText(maxY.toFixed(1), 10, padTop + 10);
  ctxAct.fillText(minY.toFixed(1), 10, padTop + plotH);
}


// ═══════════════════════════════════════════════════════════════════════════════
// 3. LOSSES & OPTIMIZERS (SGD vs MOMENTUM vs ADAM)
// ═══════════════════════════════════════════════════════════════════════════════

const canvasOptLoss = document.getElementById('canvas-opt-loss');
const ctxOptLoss = canvasOptLoss.getContext('2d');
const canvasOptAcc = document.getElementById('canvas-opt-acc');
const ctxOptAcc = canvasOptAcc.getContext('2d');
const elOptGrid = document.getElementById('optimizer-summary-grid');

let cachedOptData = null;

async function loadOptimizerComparison() {
  try {
    const res = await fetch('/api/optimizers/compare');
    const json = await res.json();
    if (json.success) {
      cachedOptData = json.data;
      renderOptimizerCards(json.data);
      renderOptimizerCharts(json.data);
    }
  } catch (err) {
    elOptGrid.innerHTML = `<div style="color: var(--accent-rose);">Error loading optimizers: ${err.message}</div>`;
  }
}

function renderOptimizerCards(data) {
  elOptGrid.innerHTML = data.results.map(r => `
    <div class="bench-card" style="border-top: 3px solid ${r.color};">
      <div class="bench-card-title">
        <span style="color: ${r.color};">${r.name}</span>
        <span class="hero-pill">lr = ${r.lr}</span>
      </div>
      <div class="metric-stat"><span class="metric-label">Final Test Accuracy</span><span class="metric-value" style="color: #6ee7b7;">${r.final_accuracy}%</span></div>
      <div class="metric-stat"><span class="metric-label">Final BCE Loss</span><span class="metric-value" style="color: #fbbf24;">${r.final_loss}</span></div>
      <div class="metric-stat"><span class="metric-label">F1 Score</span><span class="metric-value" style="color: #38bdf8;">${r.final_f1}</span></div>
    </div>
  `).join('');
}

function renderOptimizerCharts(data) {
  // Chart 1: Loss
  drawLineChart(canvasOptLoss, ctxOptLoss, data.results.map(r => ({
    label: r.name,
    color: r.color,
    epochs: r.history.epoch,
    values: r.history.train_loss
  })), 'BCE Loss', false);

  // Chart 2: Validation Accuracy
  drawLineChart(canvasOptAcc, ctxOptAcc, data.results.map(r => ({
    label: r.name,
    color: r.color,
    epochs: r.history.epoch,
    values: r.history.val_acc
  })), 'Accuracy (%)', true);
}

function drawLineChart(canvas, ctx, seriesList, yLabel, isPercentage) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const padLeft = 45;
  const padRight = 20;
  const padTop = 25;
  const padBottom = 35;
  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  // Max/min across all series
  let allVals = [];
  seriesList.forEach(s => allVals = allVals.concat(s.values));
  let minY = isPercentage ? Math.max(0, Math.min(...allVals) - 5) : 0;
  let maxY = isPercentage ? 100 : Math.max(...allVals) * 1.1;

  const maxEpoch = Math.max(...seriesList[0].epochs);

  // Grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;
  ctx.strokeRect(padLeft, padTop, plotW, plotH);

  // Plot each line
  seriesList.forEach(s => {
    ctx.beginPath();
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2.2;

    for (let i = 0; i < s.epochs.length; i++) {
      const ep = s.epochs[i];
      const val = s.values[i];
      const px = padLeft + (ep / maxEpoch) * plotW;
      const py = padTop + plotH - ((val - minY) / (maxY - minY)) * plotH;

      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  });

  // Labels
  ctx.fillStyle = '#64748b';
  ctx.font = '10px Inter, sans-serif';
  ctx.fillText('Epoch 1', padLeft, padTop + plotH + 18);
  ctx.fillText(`Epoch ${maxEpoch}`, padLeft + plotW - 45, padTop + plotH + 18);
  ctx.fillText(maxY.toFixed(isPercentage ? 0 : 2), 10, padTop + 10);
  ctx.fillText(minY.toFixed(isPercentage ? 0 : 2), 10, padTop + plotH);
}


// ═══════════════════════════════════════════════════════════════════════════════
// 4. NEURAL NET FROM SCRATCH (NUMPY)
// ═══════════════════════════════════════════════════════════════════════════════

const formTrainNN = document.getElementById('train-nn-form');
const elNNResults = document.getElementById('nn-results-container');
const elNNArchLabel = document.getElementById('nn-arch-label');
const elNNMetrics = document.getElementById('nn-metrics-grid');
const canvasScratchLoss = document.getElementById('canvas-scratch-loss');
const ctxScratchLoss = canvasScratchLoss.getContext('2d');
const canvasScratchAcc = document.getElementById('canvas-scratch-acc');
const ctxScratchAcc = canvasScratchAcc.getContext('2d');

formTrainNN.addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('btn-train-nn');
  const originalHtml = btn.innerHTML;
  btn.innerHTML = `<span class="spinner"></span> Forward &amp; Backward propagating…`;
  btn.disabled = true;

  const payload = {
    hidden_1: document.getElementById('nn-h1').value,
    hidden_2: document.getElementById('nn-h2').value,
    activation: document.getElementById('nn-act').value,
    optimizer: document.getElementById('nn-opt').value,
    lr: document.getElementById('nn-lr').value,
    epochs: document.getElementById('nn-epochs').value
  };

  try {
    const res = await fetch('/api/neuralnet/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const json = await res.json();
    if (json.success) {
      renderScratchResults(json.data);
    }
  } catch (err) {
    alert('Error training neural network: ' + err.message);
  } finally {
    btn.innerHTML = originalHtml;
    btn.disabled = false;
  }
});

function renderScratchResults(data) {
  elNNArchLabel.textContent = `Architecture: ${data.architecture} | Optimizer: ${data.optimizer} | lr=${data.lr} | Time: ${data.train_duration_ms} ms`;

  const m = data.metrics;
  elNNMetrics.innerHTML = `
    <div class="bench-card"><div class="metric-label">Test Accuracy</div><div class="metric-value" style="color: #6ee7b7; font-size: 1.3rem;">${m.accuracy}%</div></div>
    <div class="bench-card"><div class="metric-label">Precision</div><div class="metric-value" style="color: #38bdf8; font-size: 1.3rem;">${m.precision}%</div></div>
    <div class="bench-card"><div class="metric-label">Recall</div><div class="metric-value" style="color: #c4b5fd; font-size: 1.3rem;">${m.recall}%</div></div>
    <div class="bench-card"><div class="metric-label">F1-Score</div><div class="metric-value" style="color: #fbbf24; font-size: 1.3rem;">${m.f1_score}</div></div>
  `;

  // Draw Train vs Val Loss
  drawLineChart(canvasScratchLoss, ctxScratchLoss, [
    { label: 'Train Loss', color: '#38bdf8', epochs: data.history.epoch, values: data.history.train_loss },
    { label: 'Val Loss', color: '#f43f5e', epochs: data.history.epoch, values: data.history.val_loss }
  ], 'BCE Loss', false);

  // Draw Train vs Val Accuracy
  drawLineChart(canvasScratchAcc, ctxScratchAcc, [
    { label: 'Train Acc', color: '#10b981', epochs: data.history.epoch, values: data.history.train_acc },
    { label: 'Val Acc', color: '#fbbf24', epochs: data.history.epoch, values: data.history.val_acc }
  ], 'Accuracy (%)', true);

  elNNResults.style.display = 'block';
  elNNResults.scrollIntoView({ behavior: 'smooth' });
}

// ═══════════════════════════════════════════════════════════════════════════════
// INITIAL LOAD
// ═══════════════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  loadDeploymentBenchmark();
  loadPerceptronGates();
  loadActivationCurves();
  loadOptimizerComparison();
});
