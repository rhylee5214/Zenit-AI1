/* ZENIT training dashboard: runs, live loss curves, hardware, registry, evals. */
(() => {
  'use strict';
  const $ = (s, el = document) => el.querySelector(s);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const store = {
    get(k, d) { try { const v = localStorage.getItem('zenit.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem('zenit.' + k, JSON.stringify(v)); } catch { /* unavailable */ } },
  };
  const state = { key: store.get('key', ''), theme: store.get('theme', 'dark'), runs: [], runId: null, detail: null,
    logScale: store.get('logScale', true), showTable: false, timer: null, system: null };

  document.documentElement.dataset.theme = state.theme;
  $('#themeToggle').addEventListener('click', () => {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    store.set('theme', state.theme); document.documentElement.dataset.theme = state.theme;
    if (state.detail) renderDetail();
  });

  async function api(path, opts = {}) {
    const headers = { 'X-Zenit-Client': 'web' };      // needed for cookie (Google sign-in) sessions
    if (state.key) headers.Authorization = 'Bearer ' + state.key;
    let body;
    if (opts.json !== undefined) { headers['Content-Type'] = 'application/json'; body = JSON.stringify(opts.json); }
    const res = await fetch(path, { method: opts.method || 'GET', headers, body, credentials: 'same-origin' });
    let data = null;
    try { data = await res.json(); } catch { /* empty */ }
    if (!res.ok) { const e = new Error(data?.error?.message || res.statusText); e.status = res.status; e.data = data; throw e; }
    return data;
  }
  function toast(msg, kind = '') {
    const t = document.createElement('div'); t.className = 'toast ' + kind; t.textContent = msg;
    $('#toasts').appendChild(t); setTimeout(() => t.remove(), 5000);
  }
  const num = (n, d = 4) => n == null ? 'n/a' : Number(n).toFixed(d);
  const big = (n) => n == null ? 'n/a' : n >= 1e9 ? (n / 1e9).toFixed(2) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(1) + 'K' : String(n);
  const dur = (s) => { if (s == null || !isFinite(s)) return 'n/a'; s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60); return h ? `${h} h ${m} min` : m ? `${m} min ${s % 60} s` : `${s} s`; };

  // ------------------------------------------------------------ runs list
  async function loadRuns() {
    const data = await api('/v1/training/runs');
    state.runs = data.data;
    if (!state.runId && state.runs.length) state.runId = state.runs[0].run_id;
    const list = $('#runList');
    list.innerHTML = state.runs.length ? '' : '<p class="hint">No runs yet. Start one with New run, or run python -m zenit.train.</p>';
    state.runs.forEach((r) => {
      const b = document.createElement('button');
      b.className = 'run-item' + (r.run_id === state.runId ? ' active' : '');
      b.innerHTML = `<div class="name"><span class="status ${esc(r.status)}">${esc(r.status)}</span>${esc(r.run_name)}</div>
        <div class="sub">step ${r.step ?? 0} of ${r.max_steps ?? '?'}, best val ${num(r.best_val_loss, 3)}</div>`;
      b.addEventListener('click', () => { state.runId = r.run_id; state.showTable = false; loadRuns(); loadDetail(); });
      list.appendChild(b);
    });
  }

  async function loadDetail() {
    if (!state.runId) { $('#runDetail').innerHTML = '<p class="hint">No run selected.</p>'; return; }
    state.detail = await api('/v1/training/runs/' + encodeURIComponent(state.runId));
    renderDetail();
  }

  // ---------------------------------------------------------- run detail
  function renderDetail() {
    const { run, train, eval: evals, latest } = state.detail;
    const cfg = run.config || {};
    const tps = latest?.tok_per_sec;
    const perStep = (cfg.batch_size || 0) * (cfg.seq_len || 0) * (cfg.grad_accum || 1);
    const remaining = run.status === 'running' && tps ? ((run.max_steps - run.step) * perStep) / tps : null;
    const pct = run.max_steps ? Math.min(100, (100 * (run.step || 0)) / run.max_steps) : 0;
    const lastVal = evals.length ? evals[evals.length - 1].val_loss : null;
    const hw = latest || {};
    const sys = state.system || {};
    const gpus = (sys.gpus || []).map((g) => `${esc(g.name)}, ${g.memory_used_gb}/${g.memory_total_gb} GB, ${g.utilization_pct}% busy`).join('<br>')
      || (sys.torch_gpus || []).map((g) => esc(g.name) + ', ' + g.memory_gb + ' GB').join('<br>') || 'none detected';
    $('#runDetail').innerHTML = `
      <div class="run-head">
        <h2>${esc(run.run_name)}</h2>
        <span class="status ${esc(run.status)}">${esc(run.status)}</span>
        <span class="where">ZENIT ${esc(run.version)} on ${esc(run.backend)} / ${esc(run.device)}</span>
        ${run.status === 'running' ? '<button class="btn danger small" id="stopRun">Stop run</button>' : ''}
      </div>
      <div class="progress" aria-label="Training progress">
        <div class="bar"><div class="fill" style="width:${pct.toFixed(1)}%"></div></div>
        <div class="label"><span>Step ${run.step ?? 0} of ${run.max_steps} (${pct.toFixed(1)}%)</span>
          <span>${run.status === 'running' ? 'About ' + dur(remaining) + ' left' : 'Ran for ' + dur(run.elapsed_s)}</span></div>
      </div>
      <div class="tiles">
        <div class="tile"><div class="v">${num(run.last_loss, 3)}</div><div class="k">Current training loss</div></div>
        <div class="tile"><div class="v">${num(lastVal, 3)}</div><div class="k">Latest validation loss</div></div>
        <div class="tile"><div class="v">${num(run.best_val_loss, 3)}</div><div class="k">Best validation loss</div></div>
        <div class="tile"><div class="v">${tps ? big(tps) : 'n/a'}</div><div class="k">Tokens per second</div></div>
        <div class="tile"><div class="v">${big(run.dataset?.tokens_train)}</div><div class="k">Training tokens in dataset</div></div>
        <div class="tile"><div class="v">${big(run.tokens_seen)}</div><div class="k">Tokens processed</div></div>
        <div class="tile"><div class="v">${big(run.parameters)}</div><div class="k">Parameters</div></div>
        <div class="tile"><div class="v">${dur(run.elapsed_s)}</div><div class="k">Training time</div></div>
      </div>
      <div class="viz">
        <div class="chart-head">
          <h3>Loss</h3>
          <div class="legend" aria-hidden="true"><span><i style="background:var(--series-train)"></i>Training</span><span><i class="dot" style="background:var(--series-val)"></i>Validation</span></div>
          <div class="chart-btns"><button class="btn small" id="scaleBtn">${state.logScale ? 'Use linear scale' : 'Use log scale'}</button>
          <button class="btn small" id="tableBtn">${state.showTable ? 'Show chart' : 'Show table'}</button></div>
        </div>
        <div class="chart-wrap" id="chart"></div>
      </div>
      <div class="two-col">
        <div><h3>Hardware</h3><dl class="facts">
          <dt>CPU</dt><dd>${hw.cpu_pct != null ? hw.cpu_pct + '% busy' : 'n/a'}</dd>
          <dt>RAM</dt><dd>${hw.ram_used_gb != null ? hw.ram_used_gb + ' of ' + hw.ram_total_gb + ' GB' : 'n/a'}</dd>
          <dt>GPU memory</dt><dd>${hw.gpu_mem_gb != null ? hw.gpu_mem_gb + ' GB in use, peak ' + hw.gpu_mem_peak_gb + ' GB' : 'not used by this run'}</dd>
          <dt>GPU load</dt><dd>${hw.gpu_util_pct != null ? hw.gpu_util_pct + '%' : 'n/a'}</dd>
          <dt>Detected GPUs</dt><dd>${gpus}</dd>
        </dl></div>
        <div><h3>Configuration</h3><dl class="facts">
          ${['mode', 'batch_size', 'seq_len', 'grad_accum', 'lr', 'min_lr', 'warmup_steps', 'weight_decay', 'precision', 'data_dir']
            .map((k) => `<dt>${esc(k)}</dt><dd>${esc(cfg[k])}</dd>`).join('')}
          <dt>model</dt><dd>${esc(JSON.stringify(cfg.model))}</dd>
        </dl></div>
      </div>`;
    $('#scaleBtn').addEventListener('click', () => { state.logScale = !state.logScale; store.set('logScale', state.logScale); renderDetail(); });
    $('#tableBtn').addEventListener('click', () => { state.showTable = !state.showTable; renderDetail(); });
    const stopBtn = $('#stopRun');
    if (stopBtn) stopBtn.addEventListener('click', async () => {
      if (!confirm('Stop this run? It saves a checkpoint first.')) return;
      try { await api(`/v1/training/runs/${encodeURIComponent(run.run_id)}/stop`, { method: 'POST' }); toast('Stop requested. The run finishes its current step and saves.'); }
      catch (e) { toast(e.message, 'error'); }
    });
    if (state.showTable) renderTable(train, evals); else drawChart($('#chart'), train, evals);
  }

  function renderTable(train, evals) {
    const rows = evals.map((e) => `<tr><td>${e.step}</td><td>${num(e.val_loss)}</td><td>${num(e.val_ppl, 2)}</td></tr>`).join('');
    const tr = train.slice(-25).map((t) => `<tr><td>${t.step}</td><td>${num(t.loss)}</td><td>${t.lr?.toExponential(2) ?? ''}</td><td>${t.tok_per_sec ?? ''}</td></tr>`).join('');
    $('#chart').innerHTML = `<div class="two-col" style="margin-top:0">
      <div class="table-scroll"><table class="data"><caption class="sr-only">Validation loss by step</caption><thead><tr><th>Step</th><th>Validation loss</th><th>Perplexity</th></tr></thead><tbody>${rows}</tbody></table></div>
      <div class="table-scroll"><table class="data"><caption class="sr-only">Latest training steps</caption><thead><tr><th>Step</th><th>Training loss</th><th>Learning rate</th><th>Tokens/s</th></tr></thead><tbody>${tr}</tbody></table></div></div>`;
  }

  // --------------------------------------------------------------- chart
  function drawChart(box, train, evals) {
    const W = Math.max(320, box.clientWidth), H = 300;
    const m = { l: 52, r: 86, t: 14, b: 36 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    if (!train.length) { box.innerHTML = '<p class="hint">Waiting for the first logged step.</p>'; return; }
    const maxStep = Math.max(train[train.length - 1].step, evals.length ? evals[evals.length - 1].step : 0, 1);
    const values = train.map((d) => d.loss).concat(evals.map((d) => d.val_loss)).filter((v) => v > 0);
    let lo = Math.min(...values), hi = Math.max(...values);
    const log = state.logScale && lo > 0;
    let yT, ticks;
    if (log) {
      lo = Math.pow(10, Math.floor(Math.log10(lo) * 2) / 2); hi = Math.pow(10, Math.ceil(Math.log10(hi) * 2) / 2);
      const L0 = Math.log10(lo), L1 = Math.log10(hi);
      yT = (v) => m.t + ph - ((Math.log10(Math.max(v, lo)) - L0) / (L1 - L0)) * ph;
      ticks = [];
      for (let e = Math.floor(L0); e <= Math.ceil(L1); e++) [1, 2, 5].forEach((k) => { const v = k * Math.pow(10, e); if (v >= lo * 0.999 && v <= hi * 1.001) ticks.push(v); });
    } else {
      lo = 0; const step = niceStep(hi / 5); hi = Math.ceil(hi / step) * step;
      yT = (v) => m.t + ph - (v / hi) * ph;
      ticks = []; for (let v = 0; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(6));
    }
    const xT = (s) => m.l + (s / maxStep) * pw;
    const xticks = []; const xs = niceStep(maxStep / 6); for (let s = 0; s <= maxStep; s += xs) xticks.push(s);
    const path = (pts, key) => pts.map((d, i) => (i ? 'L' : 'M') + xT(d.step).toFixed(1) + ' ' + yT(d[key]).toFixed(1)).join('');
    const lastT = train[train.length - 1], lastV = evals[evals.length - 1];
    let yLt = yT(lastT.loss), yLv = lastV ? yT(lastV.val_loss) : null;
    if (yLv != null && Math.abs(yLt - yLv) < 15) { if (yLt < yLv) { yLt -= 8; yLv += 8; } else { yLt += 8; yLv -= 8; } }
    box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Training and validation loss by step">
      ${ticks.map((v) => `<line class="gridline" x1="${m.l}" x2="${m.l + pw}" y1="${yT(v)}" y2="${yT(v)}"/><text class="tick" x="${m.l - 8}" y="${yT(v) + 4}" text-anchor="end">${fmtTick(v)}</text>`).join('')}
      ${xticks.map((s) => `<text class="tick" x="${xT(s)}" y="${m.t + ph + 18}" text-anchor="middle">${big(s)}</text>`).join('')}
      <text class="axis-title" x="${m.l + pw / 2}" y="${H - 2}" text-anchor="middle">step</text>
      <text class="axis-title" transform="translate(12 ${m.t + ph / 2}) rotate(-90)" text-anchor="middle">loss, nats per token${log ? ' (log scale)' : ''}</text>
      <path d="${path(train, 'loss')}" fill="none" stroke="var(--series-train)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
      ${evals.length ? `<path d="${path(evals, 'val_loss')}" fill="none" stroke="var(--series-val)" stroke-width="2" stroke-linejoin="round"/>` : ''}
      ${evals.map((d) => `<circle cx="${xT(d.step)}" cy="${yT(d.val_loss)}" r="4" fill="var(--series-val)" stroke="var(--panel)" stroke-width="2"/>`).join('')}
      <text class="end-label" x="${xT(lastT.step) + 8}" y="${yLt + 4}">train ${num(lastT.loss, 3)}</text>
      ${lastV ? `<text class="end-label" x="${xT(lastV.step) + 8}" y="${yLv + 4}">val ${num(lastV.val_loss, 3)}</text>` : ''}
      <line class="crosshair" id="xh" y1="${m.t}" y2="${m.t + ph}" x1="-10" x2="-10"/>
      <rect id="hit" x="${m.l}" y="${m.t}" width="${pw}" height="${ph}" fill="transparent"/>
    </svg><div class="tooltip" id="tip" hidden></div>`;
    const svg = $('svg', box), tip = $('#tip', box), xh = $('#xh', box);
    const steps = train.map((d) => d.step);
    function nearest(arr, s) { let lo2 = 0, hi2 = arr.length - 1; while (lo2 < hi2) { const mid = (lo2 + hi2) >> 1; if (arr[mid] < s) lo2 = mid + 1; else hi2 = mid; } return (lo2 > 0 && Math.abs(arr[lo2 - 1] - s) < Math.abs(arr[lo2] - s)) ? lo2 - 1 : lo2; }
    function move(ev) {
      const r = svg.getBoundingClientRect();
      const x = (ev.clientX - r.left) * (W / r.width);
      const s = Math.max(0, Math.min(maxStep, ((x - m.l) / pw) * maxStep));
      const d = train[nearest(steps, s)];
      const v = evals.filter((e) => e.step <= d.step).pop();
      const px = xT(d.step);
      xh.setAttribute('x1', px); xh.setAttribute('x2', px);
      tip.hidden = false;
      tip.style.left = (px * r.width / W) + 'px';
      tip.style.top = (yT(d.loss) * r.height / H) + 'px';
      tip.innerHTML = `<b>Step ${d.step}</b><br><span class="sw" style="background:var(--series-train)"></span>Training ${num(d.loss)}` +
        (v ? `<br><span class="sw" style="background:var(--series-val)"></span>Validation ${num(v.val_loss)} (step ${v.step})` : '');
    }
    const hit = $('#hit', box);
    hit.addEventListener('pointermove', move);
    hit.addEventListener('pointerleave', () => { tip.hidden = true; xh.setAttribute('x1', -10); xh.setAttribute('x2', -10); });
  }
  function niceStep(raw) { const p = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-9)))); const f = raw / p; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p; }
  function fmtTick(v) { return v >= 10 ? v.toFixed(0) : v >= 1 ? (+v.toFixed(1)).toString() : (+v.toPrecision(2)).toString(); }

  // ------------------------------------------------------------ new run
  $('#newRunBtn').addEventListener('click', () => { const s = $('#newRun'); s.hidden = !s.hidden; if (!s.hidden) s.scrollIntoView({ behavior: 'smooth' }); });
  function formConfig() {
    const f = new FormData($('#runForm'));
    const cfg = {};
    for (const [k, v] of f.entries()) cfg[k] = v;
    const out = { run_name: cfg.run_name, version: cfg.version, backend: cfg.backend, data_dir: cfg.data_dir, tokenizer: cfg.tokenizer,
      mode: cfg.mode, precision: cfg.precision, model: { preset: cfg.preset }, stage: cfg.mode };
    ['max_steps', 'batch_size', 'seq_len', 'grad_accum', 'warmup_steps', 'eval_interval'].forEach((k) => out[k] = parseInt(cfg[k], 10));
    out.lr = parseFloat(cfg.lr); out.min_lr = out.lr / 10;
    if (cfg.init_from) out.init_from = cfg.init_from;
    return out;
  }
  function showEstimate(est, reasons, allowForce) {
    const e = est || {};
    $('#estimateOut').innerHTML = `<div class="estimate">
      ${(reasons || []).map((r) => `<div class="warn">${esc(r)}</div>`).join('')}
      <dl class="facts">
        <dt>Parameters</dt><dd>${(e.parameters ?? 0).toLocaleString()}</dd>
        <dt>Tokens to process</dt><dd>${big(e.total_tokens)} (compute-optimal for this size: about ${big(e.chinchilla_optimal_tokens)})</dd>
        <dt>Compute</dt><dd>${e.total_pflops} PFLOPs</dd>
        <dt>Memory needed</dt><dd>about ${e.memory_gb_estimate} GB${e.gpu_memory_gb ? ', GPU has ' + e.gpu_memory_gb + ' GB' : ''}</dd>
        <dt>Disk for checkpoints</dt><dd>${e.disk_gb_total} GB of ${e.disk_free_gb} GB free</dd>
        <dt>Measured speed</dt><dd>${e.measured_peak_tflops != null ? e.measured_peak_tflops + ' TFLOPS peak on ' + esc(e.device || e.gpu || 'this machine') : esc(e.warning || 'not measured')}</dd>
        <dt>Estimated time</dt><dd>${e.estimated_hours != null ? dur(e.estimated_hours * 3600) : 'unknown'}</dd>
      </dl>
      ${allowForce ? '<div class="row" style="margin-top:12px"><button class="btn primary" id="forceStart">Start anyway</button></div>' : ''}
      <p class="hint" style="margin-top:10px">Estimates come from a quick speed test on the server and assume typical utilisation. Treat them as rough.</p></div>`;
    const f = $('#forceStart');
    if (f) f.addEventListener('click', () => start(true));
  }
  $('#estimateBtn').addEventListener('click', async () => {
    $('#estimateOut').innerHTML = '<p class="hint">Measuring this machine</p>';
    try { const r = await api('/v1/training/estimate', { method: 'POST', json: { config: formConfig() } }); showEstimate(r.estimate, r.needs_confirmation, false); }
    catch (e) { $('#estimateOut').innerHTML = `<p class="error-text">${esc(e.message)}</p>`; }
  });
  async function start(confirmBig) {
    try {
      const r = await api('/v1/training/runs', { method: 'POST', json: { config: formConfig(), confirm: !!confirmBig } });
      toast('Training started (process ' + r.pid + ')');
      showEstimate(r.estimate, [], false);
      setTimeout(async () => { state.runId = null; await loadRuns(); loadDetail(); }, 2500);
    } catch (e) {
      if (e.status === 409) showEstimate(e.data.error.estimate, e.data.error.reasons, true);
      else toast(e.message, 'error');
    }
  }
  $('#runForm').addEventListener('submit', (e) => { e.preventDefault(); start(false); });

  // ------------------------------------------------ registry and evals
  async function loadRegistry() {
    const reg = await api('/v1/registry');
    const models = reg.models || [];
    $('#registry').innerHTML = models.length ? `<div class="table-scroll"><table class="data"><thead><tr><th>Model</th><th>Version</th><th>Stage</th><th>Parameters</th><th>Best val loss</th><th>Status</th></tr></thead><tbody>
      ${models.map((m) => `<tr><td>${esc(m.id)}</td><td>${esc(m.version)}</td><td>${esc(m.stage)}</td><td>${big(m.parameters)}</td><td>${num(m.metrics?.best_val_loss, 3)}</td><td>${esc(m.status)}</td></tr>`).join('')}</tbody></table></div>`
      : '<p class="hint">No models registered yet.</p>';
    $('#roadmap').innerHTML = (reg.roadmap || []).map((r) => {
      const hits = models.filter((m) => String(m.version).split('-')[0] === r.version);
      const released = hits.some((m) => m.status === 'released');
      const stateText = released ? 'Released' : hits.length ? 'Checkpoint trained, ' + esc(hits[0].status) : 'Not started';
      return `<li class="${released ? 'reached' : ''}"><span class="ver">${esc(r.version)}</span><span>${esc(r.goal)}</span><span class="state">${stateText}</span></li>`;
    }).join('');
  }
  async function loadEvals() {
    const data = await api('/v1/evals');
    if (!data.data.length) { $('#evals').innerHTML = '<p class="hint">No evaluation runs yet. Run python -m zenit.evals.run --model MODEL_ID.</p>'; return; }
    const cats = [...new Set(data.data.flatMap((e) => Object.keys(e.summary || {})))];
    $('#evals').innerHTML = `<div class="table-scroll"><table class="data"><thead><tr><th>Model</th><th>Date</th><th>Suite</th>${cats.map((c) => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>
      ${data.data.map((e) => `<tr><td>${esc(e.model)}</td><td>${esc((e.created || '').slice(0, 16).replace('T', ' '))}</td><td>${esc(e.suite_hash)}</td>${cats.map((c) => {
        const v = e.summary?.[c]; return `<td>${v == null ? '' : typeof v === 'number' && c !== 'tokens_per_sec' && c !== 'first_token_ms' ? (v * 100).toFixed(1) + '%' : esc(v)}</td>`; }).join('')}</tr>`).join('')}</tbody></table></div>
      <p class="hint" style="margin-top:8px">Rows sharing a suite hash used identical test sets, so they compare directly.</p>`;
  }

  async function refresh() {
    try {
      await loadRuns();
      await loadDetail();
    } catch (e) {
      if (e.status === 401) { $('#runDetail').innerHTML = '<p class="hint">Connect from the <a href="/">chat page</a> with an API key first, or start the server with --local.</p>'; return; }
      toast(e.message, 'error');
    }
    const running = state.runs.some((r) => r.status === 'running');
    clearTimeout(state.timer);
    state.timer = setTimeout(refresh, running ? 3000 : 15000);
  }

  (async () => {
    try { state.system = await api('/v1/system'); } catch { /* optional */ }
    await refresh();
    loadRegistry().catch(() => {}); loadEvals().catch(() => {});
    window.addEventListener('resize', () => { if (state.detail && !state.showTable) drawChart($('#chart'), state.detail.train, state.detail.eval); });
  })();
})();
