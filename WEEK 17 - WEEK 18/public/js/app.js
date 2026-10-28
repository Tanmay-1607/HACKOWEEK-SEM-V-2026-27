// ─── Week 17 & 18: Deep Learning & LLMs Studio ──────────────────────────────

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

    // Trigger canvas redraws on tab switch
    if (target === 'cnns' && cachedCnnData) renderCnnChart(cachedCnnData);
    if (target === 'sequence' && cachedSeqData) renderSeqChart(cachedSeqData);
    if (target === 'regularization' && cachedRegData) renderRegCharts(cachedRegData);
  });
});

// Color Tokens
const COLORS = {
  emerald: '#10b981',
  sky: '#38bdf8',
  violet: '#8b5cf6',
  amber: '#f59e0b',
  rose: '#f43f5e',
  orange: '#f97316'
};

// ═══════════════════════════════════════════════════════════════════════════════
// 1. PYTORCH BASICS
// ═══════════════════════════════════════════════════════════════════════════════
const elPyTorchGrid = document.getElementById('pytorch-basics-grid');

async function loadPyTorchBasics() {
  try {
    const res = await fetch('/api/pytorch/basics');
    const json = await res.json();
    if (json.success) {
      renderPyTorchBasics(json.data);
    }
  } catch (err) {
    elPyTorchGrid.innerHTML = `<div style="color: var(--accent-rose);">Error: ${err.message}</div>`;
  }
}

function renderPyTorchBasics(d) {
  const dev = d.device_info;
  const auto = d.autograd;
  const loop = d.training_loop;

  elPyTorchGrid.innerHTML = `
    <div class="bench-card" style="border-top: 3px solid var(--accent-orange);">
      <div class="bench-card-title">
        <span>⚡ Hardware Device</span>
        <span class="hero-pill" style="color: #6ee7b7;">CUDA Active</span>
      </div>
      <div class="metric-stat"><span class="metric-label">Target Device</span><span class="metric-value" style="color: #fbbf24;">${dev.device.toUpperCase()}</span></div>
      <div class="metric-stat"><span class="metric-label">GPU Accelerator</span><span class="metric-value" style="color: #38bdf8; font-size: 0.85rem;">${dev.device_name}</span></div>
      <div class="metric-stat"><span class="metric-label">PyTorch Version</span><span class="metric-value">${dev.pytorch_version}</span></div>
      <div class="metric-stat"><span class="metric-label">CUDA Acceleration</span><span class="metric-value" style="color: #6ee7b7;">${dev.cuda_available ? 'Available (Enabled)' : 'CPU Only'}</span></div>
    </div>

    <div class="bench-card" style="border-top: 3px solid var(--accent-sky);">
      <div class="bench-card-title">
        <span>📐 Autograd Differentiation</span>
        <span class="hero-pill" style="color: #38bdf8;">Automatic Differentiation</span>
      </div>
      <div class="code-box" style="margin-bottom: 0.75rem; font-size: 0.75rem;">${auto.y_formula}
x = ${auto.x}
${auto.analytical_gradient_formula}</div>
      <div class="metric-stat"><span class="metric-label">Forward y</span><span class="metric-value" style="color: #fbbf24;">${auto.computed_y}</span></div>
      <div class="metric-stat"><span class="metric-label">Autograd dy/dx</span><span class="metric-value" style="color: #6ee7b7;">${auto.computed_gradient}</span></div>
      <div class="metric-stat"><span class="metric-label">Analytical Check</span><span class="metric-value" style="color: #6ee7b7;">${auto.expected_gradient} (Match)</span></div>
    </div>

    <div class="bench-card" style="border-top: 3px solid var(--accent-emerald);">
      <div class="bench-card-title">
        <span>🔄 PyTorch Training Loop</span>
        <span class="hero-pill" style="color: #6ee7b7;">SGD Optimizer</span>
      </div>
      <p style="font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 0.5rem;">Target: <code>${loop.target_equation}</code></p>
      <div class="metric-stat"><span class="metric-label">Learned Weight (w)</span><span class="metric-value" style="color: #6ee7b7;">${loop.learned_weight}</span></div>
      <div class="metric-stat"><span class="metric-label">Learned Bias (b)</span><span class="metric-value" style="color: #38bdf8;">${loop.learned_bias}</span></div>
      <div class="metric-stat"><span class="metric-label">Final MSE Loss</span><span class="metric-value" style="color: #fbbf24;">${loop.loss_history[loop.loss_history.length - 1].loss}</span></div>
    </div>
  `;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. CNN IMAGE CLASSIFICATION
// ═══════════════════════════════════════════════════════════════════════════════
const canvasCnn = document.getElementById('canvas-cnn-training');
const ctxCnn = canvasCnn.getContext('2d');
const elCnnStats = document.getElementById('cnn-stats-box');
const elCnnLayers = document.getElementById('cnn-layers-list');

let cachedCnnData = null;

async function loadCnnDemo() {
  try {
    const res = await fetch('/api/cnn/demo');
    const json = await res.json();
    if (json.success) {
      cachedCnnData = json.data;
      renderCnnChart(json.data);
      renderCnnLayers(json.data);
    }
  } catch (err) {
    elCnnStats.innerHTML = `<div style="color: var(--accent-rose);">Error: ${err.message}</div>`;
  }
}

function renderCnnChart(d) {
  elCnnStats.innerHTML = `
    <div class="bench-card-title">
      <span>CNN Architecture &amp; Performance</span>
      <span class="hero-pill" style="color: #6ee7b7;">PyTorch nn.Module</span>
    </div>
    <div class="code-box" style="margin-bottom: 0.75rem; font-size: 0.75rem;">${d.architecture}</div>
    <div class="metric-stat"><span class="metric-label">Validation Accuracy</span><span class="metric-value" style="color: #6ee7b7; font-size: 1.2rem;">${d.final_val_accuracy}%</span></div>
    <div class="metric-stat"><span class="metric-label">Final Training Loss</span><span class="metric-value" style="color: #fbbf24;">${d.final_train_loss}</span></div>
    <div class="metric-stat"><span class="metric-label">Training Execution Time</span><span class="metric-value" style="color: #38bdf8;">${d.train_time_ms} ms</span></div>
  `;

  drawLineChart(canvasCnn, ctxCnn, [
    { label: 'Train Loss', color: '#fbbf24', epochs: d.history.epoch, values: d.history.train_loss },
    { label: 'Val Accuracy (%)', color: '#10b981', epochs: d.history.epoch, values: d.history.val_acc }
  ], 'Metrics', false);
}

function renderCnnLayers(d) {
  elCnnLayers.innerHTML = `
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.75rem;">
      ${d.layers_trace.map(l => `
        <div style="background: rgba(14, 22, 42, 0.6); border: 1px solid var(--border-color); border-radius: var(--radius-sm); padding: 0.75rem;">
          <div style="font-weight: 600; font-size: 0.82rem; color: var(--accent-sky); margin-bottom: 0.25rem;">${l.name}</div>
          <div style="font-family: var(--font-mono); font-size: 0.78rem; color: #6ee7b7; margin-bottom: 0.25rem;">${l.shape}</div>
          <div style="font-size: 0.72rem; color: var(--text-muted);">${l.detail}</div>
        </div>
      `).join('')}
    </div>
  `;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. SEQUENCE MODELING (RNN vs LSTM vs GRU)
// ═══════════════════════════════════════════════════════════════════════════════
const canvasSeq = document.getElementById('canvas-seq-loss');
const ctxSeq = canvasSeq.getContext('2d');
const elSeqGrid = document.getElementById('seq-models-grid');

let cachedSeqData = null;

async function loadSequenceModels() {
  try {
    const res = await fetch('/api/sequence/compare');
    const json = await res.json();
    if (json.success) {
      cachedSeqData = json.data;
      renderSeqChart(json.data);
    }
  } catch (err) {
    elSeqGrid.innerHTML = `<div style="color: var(--accent-rose);">Error: ${err.message}</div>`;
  }
}

function renderSeqChart(d) {
  const series = d.results.map(r => ({
    label: r.name,
    color: r.color,
    epochs: r.loss_curve.map(c => c.epoch),
    values: r.loss_curve.map(c => c.loss)
  }));

  drawLineChart(canvasSeq, ctxSeq, series, 'MSE Loss', false);

  elSeqGrid.innerHTML = d.results.map(r => `
    <div class="bench-card" style="border-left: 3px solid ${r.color};">
      <div class="bench-card-title">
        <span style="color: ${r.color};">${r.name}</span>
        <span class="hero-pill">${r.parameters} parameters</span>
      </div>
      <div style="font-size: 0.78rem; font-family: var(--font-mono); color: #cbd5e1; margin-bottom: 0.5rem;">${r.formula}</div>
      <div style="display: flex; gap: 1.5rem;">
        <span style="font-size: 0.82rem; color: var(--text-secondary);">Final MSE: <strong style="color: #6ee7b7;">${r.final_mse}</strong></span>
        <span style="font-size: 0.82rem; color: var(--text-secondary);">Train Time: <strong style="color: #38bdf8;">${r.train_time_ms} ms</strong></span>
      </div>
    </div>
  `).join('');
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. REGULARIZATION: DROPOUT, BATCHNORM & EARLY STOPPING
// ═══════════════════════════════════════════════════════════════════════════════
const canvasRegUnreg = document.getElementById('canvas-reg-unreg');
const ctxRegUnreg = canvasRegUnreg.getContext('2d');
const canvasRegReg = document.getElementById('canvas-reg-reg');
const ctxRegReg = canvasRegReg.getContext('2d');
const elRegSummary = document.getElementById('reg-summary-box');

let cachedRegData = null;

async function loadRegularizationComparison() {
  try {
    const res = await fetch('/api/regularization/compare');
    const json = await res.json();
    if (json.success) {
      cachedRegData = json.data;
      renderRegCharts(json.data);
    }
  } catch (err) {
    elRegSummary.innerHTML = `<div style="color: var(--accent-rose);">Error: ${err.message}</div>`;
  }
}

function renderRegCharts(d) {
  const u = d.unregularized;
  const r = d.regularized;

  const uEpochs = u.history.train_loss.map((_, i) => i + 1);
  const rEpochs = r.history.train_loss.map((_, i) => i + 1);

  // Unregularized chart
  drawLineChart(canvasRegUnreg, ctxRegUnreg, [
    { label: 'Train Loss', color: '#38bdf8', epochs: uEpochs, values: u.history.train_loss },
    { label: 'Val Loss (Diverging)', color: '#f43f5e', epochs: uEpochs, values: u.history.val_loss }
  ], 'BCE Loss', false);

  // Regularized chart
  drawLineChart(canvasRegReg, ctxRegReg, [
    { label: 'Train Loss', color: '#38bdf8', epochs: rEpochs, values: r.history.train_loss },
    { label: 'Val Loss (Controlled)', color: '#10b981', epochs: rEpochs, values: r.history.val_loss }
  ], 'BCE Loss', false);

  elRegSummary.innerHTML = `
    <div class="bench-card" style="border-top: 3px solid var(--accent-rose);">
      <div class="bench-card-title"><span style="color: #fb7185;">Unregularized Network</span></div>
      <div class="metric-stat"><span class="metric-label">Final Train Loss</span><span class="metric-value" style="color: #38bdf8;">${u.final_train_loss}</span></div>
      <div class="metric-stat"><span class="metric-label">Final Val Loss</span><span class="metric-value" style="color: #fb7185;">${u.final_val_loss}</span></div>
      <div class="metric-stat"><span class="metric-label">Generalization Gap</span><span class="metric-value" style="color: #fb7185;">${u.generalization_gap} (Overfitting!)</span></div>
    </div>

    <div class="bench-card" style="border-top: 3px solid var(--accent-emerald);">
      <div class="bench-card-title"><span style="color: #6ee7b7;">Regularized Network</span></div>
      <div class="metric-stat"><span class="metric-label">Early Stopping Triggered</span><span class="metric-value" style="color: #6ee7b7;">Epoch ${r.stopped_epoch}</span></div>
      <div class="metric-stat"><span class="metric-label">Best Val Loss</span><span class="metric-value" style="color: #6ee7b7;">${r.best_val_loss}</span></div>
      <div class="metric-stat"><span class="metric-label">Generalization Gap</span><span class="metric-value" style="color: #6ee7b7;">${r.generalization_gap} (Controlled)</span></div>
    </div>
  `;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. TRANSFER LEARNING
// ═══════════════════════════════════════════════════════════════════════════════
const elTransferBox = document.getElementById('transfer-info-box');

async function loadTransferLearning() {
  try {
    const res = await fetch('/api/transfer-learning/info');
    const json = await res.json();
    if (json.success) {
      renderTransferLearning(json.data);
    }
  } catch (err) {
    elTransferBox.innerHTML = `<div style="color: var(--accent-rose);">Error: ${err.message}</div>`;
  }
}

function renderTransferLearning(d) {
  elTransferBox.innerHTML = `
    <div class="grid-2" style="margin-bottom: 1.5rem;">
      <div class="bench-card" style="border-top: 3px solid var(--accent-sky);">
        <div class="bench-card-title">
          <span>Backbone: ${d.base_model}</span>
          <span class="hero-pill" style="color: #38bdf8;">Feature Extractor</span>
        </div>
        <div class="metric-stat"><span class="metric-label">Total Parameters</span><span class="metric-value">${d.total_parameters}</span></div>
        <div class="metric-stat"><span class="metric-label">Frozen Base Parameters</span><span class="metric-value" style="color: #94a3b8;">${d.frozen_parameters}</span></div>
        <div class="metric-stat"><span class="metric-label">Trainable Custom Head</span><span class="metric-value" style="color: #6ee7b7;">${d.trainable_parameters}</span></div>
        <div class="metric-stat"><span class="metric-label">Compute Parameter Savings</span><span class="metric-value" style="color: #38bdf8;">${d.parameter_savings_percent}%</span></div>
      </div>

      <div class="bench-card" style="border-top: 3px solid var(--accent-emerald);">
        <div class="bench-card-title"><span>Transfer Learning Methodology</span></div>
        <div style="display: flex; flex-direction: column; gap: 0.75rem;">
          ${d.fine_tuning_strategy.map(s => `
            <div>
              <div style="font-weight: 600; font-size: 0.84rem; color: #6ee7b7;">${s.phase}</div>
              <div style="font-size: 0.78rem; color: var(--text-secondary);">${s.action}</div>
            </div>
          `).join('')}
        </div>
      </div>
    </div>
  `;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. HUGGING FACE TRANSFORMERS
// ═══════════════════════════════════════════════════════════════════════════════
const formTrans = document.getElementById('transformers-form');
const elTransBox = document.getElementById('transformers-result-box');

formTrans.addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('btn-trans-analyze');
  const original = btn.innerHTML;
  btn.innerHTML = `<span class="spinner"></span> Tokenizing &amp; Inferring…`;
  btn.disabled = true;

  try {
    const res = await fetch('/api/transformers/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: document.getElementById('trans-input').value })
    });
    const json = await res.json();
    if (json.success) {
      renderTransformersResult(json.data);
    }
  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    btn.innerHTML = original;
    btn.disabled = false;
  }
});

function renderTransformersResult(d) {
  const p = d.prediction;
  const isPos = p.label === 'POSITIVE';
  const color = isPos ? 'var(--accent-emerald)' : 'var(--accent-rose)';

  elTransBox.innerHTML = `
    <div class="bench-card" style="border-left: 4px solid ${color};">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
        <span style="font-family: var(--font-heading); font-size: 1.25rem; font-weight: 700; color: ${color};">
          ${isPos ? '✨ POSITIVE SENTIMENT' : '⚠️ NEGATIVE SENTIMENT'} (${p.confidence}%)
        </span>
        <span class="hero-pill" style="color: var(--accent-sky);">${d.transformer_architecture}</span>
      </div>

      <div style="margin-bottom: 1rem;">
        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.35rem;">Sub-word Tokenization Breakdown:</div>
        <div style="display: flex; flex-wrap: wrap; gap: 0.35rem;">
          ${d.tokenization_breakdown.tokens.map(t => `<span class="hero-pill" style="font-family: var(--font-mono); color: #38bdf8;">${t}</span>`).join('')}
        </div>
      </div>

      <div class="code-box" style="font-size: 0.75rem; margin-bottom: 0.75rem;">Input IDs: [${d.tokenization_breakdown.input_ids.join(', ')}]
Attention Mask: [${d.tokenization_breakdown.attention_mask.join(', ')}]</div>
    </div>
  `;
  elTransBox.style.display = 'block';
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. LLM STUDIO
// ═══════════════════════════════════════════════════════════════════════════════
const formLlm = document.getElementById('llm-form');
const elLlmOut = document.getElementById('llm-output-box');

formLlm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('btn-llm-run');
  const original = btn.innerHTML;
  btn.innerHTML = `<span class="spinner"></span> Dispatching Prompt…`;
  btn.disabled = true;

  const payload = {
    provider: document.getElementById('llm-provider').value,
    model: document.getElementById('llm-model').value,
    task: document.getElementById('llm-task').value,
    prompt: document.getElementById('llm-prompt').value
  };

  try {
    const res = await fetch('/api/llm/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const json = await res.json();
    if (json.success) {
      renderLlmResult(json.data);
    }
  } catch (err) {
    alert('Error executing LLM: ' + err.message);
  } finally {
    btn.innerHTML = original;
    btn.disabled = false;
  }
});

function renderLlmResult(d) {
  elLlmOut.innerHTML = `
    <div class="bench-card" style="border-top: 3px solid var(--accent-orange);">
      <div class="bench-card-title">
        <span>🤖 Response from ${d.provider.toUpperCase()} (${d.model})</span>
        <span class="hero-pill" style="color: #38bdf8;">Latency: ${d.metrics.latency_ms} ms</span>
      </div>
      <div class="code-box" style="margin-bottom: 0.75rem; white-space: pre-wrap; font-size: 0.85rem; color: #f1f5f9;">${d.response}</div>
      <div style="display: flex; gap: 1.5rem; font-size: 0.8rem; color: var(--text-secondary);">
        <span>Prompt Tokens: <strong style="color: #38bdf8;">${d.metrics.prompt_tokens}</strong></span>
        <span>Completion Tokens: <strong style="color: #6ee7b7;">${d.metrics.completion_tokens}</strong></span>
        <span>Total Tokens: <strong style="color: #fbbf24;">${d.metrics.total_tokens}</strong></span>
      </div>
    </div>
  `;
  elLlmOut.style.display = 'block';
}

// ═══════════════════════════════════════════════════════════════════════════════
// CANVAS DRAWING HELPER
// ═══════════════════════════════════════════════════════════════════════════════
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

  let allVals = [];
  seriesList.forEach(s => allVals = allVals.concat(s.values));
  let minY = isPercentage ? 0 : Math.max(0, Math.min(...allVals) * 0.9);
  let maxY = isPercentage ? 100 : Math.max(...allVals) * 1.1;

  const maxEpoch = Math.max(...seriesList[0].epochs);

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;
  ctx.strokeRect(padLeft, padTop, plotW, plotH);

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

  ctx.fillStyle = '#64748b';
  ctx.font = '10px Inter, sans-serif';
  ctx.fillText('Epoch 1', padLeft, padTop + plotH + 18);
  ctx.fillText(`Epoch ${maxEpoch}`, padLeft + plotW - 45, padTop + plotH + 18);
  ctx.fillText(maxY.toFixed(isPercentage ? 0 : 3), 10, padTop + 10);
  ctx.fillText(minY.toFixed(isPercentage ? 0 : 3), 10, padTop + plotH);
}

// ═══════════════════════════════════════════════════════════════════════════════
// INITIAL LOAD
// ═══════════════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  loadPyTorchBasics();
  loadCnnDemo();
  loadSequenceModels();
  loadRegularizationComparison();
  loadTransferLearning();
});
