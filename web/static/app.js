/* ZENIT chat client. Talks only to the ZENIT server it was loaded from. */
(() => {
  'use strict';
  const $ = (s, el = document) => el.querySelector(s);
  const MD = window.ZenitMD;

  const store = {
    get(k, d) { try { const v = localStorage.getItem('zenit.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem('zenit.' + k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
  };

  const state = {
    key: store.get('key', ''),
    theme: store.get('theme', 'dark'),
    model: store.get('model', null),
    profile: store.get('profile', 'balanced'),
    prefs: Object.assign({ temperature: '', max_tokens: '', use_documents: true }, store.get('prefs', {})),
    models: [], profiles: {}, convs: [], convId: null, attached: [],
    generating: false, controller: null, genId: null, me: null, memoryEnabled: true,
    auth: { enabled: false, google_client_id: null, local_access: false }, gsiReady: false,
    rbx: { on: store.get('roblox', false) === true, status: null, timer: null, checking: false },
  };

  const el = {
    app: $('#app'), sidebar: $('#sidebar'), history: $('#history'), messages: $('#messages'), thread: $('#thread'),
    empty: $('#empty'), emptyNote: $('#emptyNote'), suggestions: $('#suggestions'), input: $('#input'),
    composer: $('#composer'), send: $('#sendBtn'), stop: $('#stopBtn'), attach: $('#attachBtn'), file: $('#fileInput'),
    attachments: $('#attachments'), modelSelect: $('#modelSelect'), profileSelect: $('#profileSelect'),
    drawer: $('#drawer'), drawerTitle: $('#drawerTitle'), drawerBody: $('#drawerBody'), toasts: $('#toasts'),
    search: $('#searchChats'), dialog: $('#connectDialog'), account: $('#account'), signin: $('#signin'),
    rbxToggle: $('#rbxToggle'), rbxBar: $('#rbxBar'), rbxPill: $('#rbxPill'), rbxStudio: $('#rbxStudio'),
  };

  const DEFAULT_PLACEHOLDER = el.input.placeholder;
  const ROBLOX_PLACEHOLDER = 'Tell ZENIT what to build in Roblox Studio';
  const ROBLOX_IDEAS = ['Build a small wooden house at the origin', 'Make a glowing red part that spins', 'Add a leaderboard that tracks coins'];
  const CUBE = '<svg class="tool-ico" viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9zM4 7.5l8 4.5 8-4.5M12 12v9" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>';

  const AVATAR ='<svg class="avatar" viewBox="0 0 32 32" aria-hidden="true"><path d="M4 24 A12 12 0 0 1 28 24" fill="none" stroke="url(#zg)" stroke-width="2.2" stroke-linecap="round"/><circle cx="16" cy="12" r="2.6" fill="url(#zg)"/></svg>';

  // ------------------------------------------------------------ helpers
  async function api(path, opts = {}) {
    // X-Zenit-Client marks requests from ZENIT's own page; the server requires it
    // for state changes made with the Google sign-in cookie (CSRF protection).
    const headers = Object.assign({ 'X-Zenit-Client': 'web' }, opts.headers || {});
    if (state.key) headers.Authorization = 'Bearer ' + state.key;
    let body = opts.body;
    if (opts.json !== undefined) { headers['Content-Type'] = 'application/json'; body = JSON.stringify(opts.json); }
    const res = await fetch(path, { method: opts.method || 'GET', headers, body, signal: opts.signal, credentials: 'same-origin' });
    if (opts.raw) return res;
    let data = null;
    try { data = await res.json(); } catch { /* empty body */ }
    if (!res.ok) {
      const err = new Error((data && data.error && data.error.message) || res.statusText || 'Request failed');
      err.status = res.status; err.data = data;
      throw err;
    }
    return data;
  }

  function toast(msg, kind = '') {
    const t = document.createElement('div');
    t.className = 'toast ' + kind;
    t.textContent = msg;
    el.toasts.appendChild(t);
    setTimeout(() => t.remove(), kind === 'error' ? 6000 : 3000);
  }

  const fmtNum = (n) => n == null ? 'unknown' : n >= 1e9 ? (n / 1e9).toFixed(1) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e3 ? Math.round(n / 1e3) + 'K' : String(n);
  const fmtBytes = (b) => b > 1e6 ? (b / 1e6).toFixed(1) + ' MB' : Math.max(1, Math.round(b / 1e3)) + ' KB';
  const fmtDate = (t) => new Date(t * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
  const escapeHtml = MD.escape;

  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); }
    catch {
      const ta = document.createElement('textarea');
      ta.value = text; document.body.appendChild(ta); ta.select();
      try { document.execCommand('copy'); } finally { ta.remove(); }
    }
    toast('Copied');
  }

  function nearBottom() { const t = el.thread; return t.scrollHeight - t.scrollTop - t.clientHeight < 140; }
  function scrollDown(force) { if (force || nearBottom()) el.thread.scrollTop = el.thread.scrollHeight; }

  // -------------------------------------------------------------- theme
  function applyTheme() { document.documentElement.dataset.theme = state.theme; }
  $('#themeToggle').addEventListener('click', () => {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    store.set('theme', state.theme); applyTheme();
  });

  // ------------------------------------------------------------ connect
  function showConnect(message) {
    $('#connectError').textContent = message || '';
    $('#keyInput').value = state.key || '';
    if (!el.dialog.open) el.dialog.showModal();
  }
  $('#connectForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    state.key = $('#keyInput').value.trim();
    try {
      state.me = await api('/v1/me');
      store.set('key', state.key);
      el.dialog.close();
      await boot();
    } catch (err) {
      $('#connectError').textContent = err.status === 401 ? 'That key was not accepted.' : 'Server unreachable: ' + err.message;
    }
  });

  // ------------------------------------------------------------- models
  async function loadModels() {
    const data = await api('/v1/models');
    state.models = data.data || [];
    state.profiles = data.profiles || {};
    if (!state.models.find((m) => m.id === state.model)) state.model = data.default;
    el.modelSelect.innerHTML = state.models.map((m) =>
      `<option value="${escapeHtml(m.id)}">${escapeHtml(m.name || m.id)}${m.parameters ? ' (' + fmtNum(m.parameters) + ')' : ''}</option>`).join('')
      || '<option value="">No models yet</option>';
    el.modelSelect.value = state.model || '';
    el.profileSelect.innerHTML = Object.entries(state.profiles).map(([n, d]) =>
      `<option value="${escapeHtml(n)}" title="${escapeHtml(d)}">${escapeHtml(n[0].toUpperCase() + n.slice(1))}</option>`).join('');
    if (!state.profiles[state.profile]) state.profile = Object.keys(state.profiles)[0] || 'balanced';
    el.profileSelect.value = state.profile;
    describeModel();
  }

  function currentModel() { return state.models.find((m) => m.id === state.model) || {}; }

  function describeModel() {
    const m = currentModel();
    if (!m.id) {
      el.emptyNote.textContent = 'No trained model found. Train one with python -m zenit.train, then restart the server.';
      el.send.disabled = true;
      return;
    }
    el.send.disabled = false;
    const tiny = m.parameters && m.parameters < 20e6;
    el.emptyNote.textContent = tiny
      ? `${m.id} is an experimental build with ${fmtNum(m.parameters)} parameters, trained from scratch on a small seed corpus. Expect short, limited answers.`
      : `${m.id}${m.parameters ? ', ' + fmtNum(m.parameters) + ' parameters' : ''}.`;
    const ideas = state.rbx.on ? ROBLOX_IDEAS : tiny
      ?['Who are you?', 'What is 23 + 14?', 'What is the capital of Japan?', 'Write a Python function to reverse a string.']
      : ['Explain how attention works in a transformer', 'Find the bug in a Python function I paste', 'Summarize the document I uploaded', 'Plan a study schedule for linear algebra'];
    el.suggestions.innerHTML = '';
    ideas.forEach((t) => {
      const b = document.createElement('button');
      b.type = 'button'; b.textContent = t;
      b.addEventListener('click', () => send(t));
      el.suggestions.appendChild(b);
    });
  }

  el.modelSelect.addEventListener('change', () => {
    state.model = el.modelSelect.value; store.set('model', state.model); describeModel();
    if (state.rbx.on && state.rbx.status) renderRobloxStatus(state.rbx.status);
  });
  el.profileSelect.addEventListener('change', () => { state.profile = el.profileSelect.value; store.set('profile', state.profile); });

  // ------------------------------------------------------- conversations
  async function loadConversations() {
    const data = await api('/v1/conversations');
    state.convs = data.data || [];
    renderHistory();
  }

  function groupLabel(ts) {
    const d = new Date(ts * 1000), now = new Date();
    const day = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
    const diff = (day(now) - day(d)) / 86400000;
    if (diff < 1) return 'Today';
    if (diff < 2) return 'Yesterday';
    if (diff < 7) return 'Previous 7 days';
    return 'Older';
  }

  function renderHistory() {
    const q = el.search.value.trim().toLowerCase();
    const list = state.convs.filter((c) => !q || (c.title || '').toLowerCase().includes(q));
    el.history.innerHTML = '';
    if (!list.length) {
      el.history.innerHTML = `<p class="history-empty">${q ? 'No chats match that search.' : 'Your chats appear here.'}</p>`;
      return;
    }
    let current = '';
    list.forEach((c) => {
      const g = groupLabel(c.updated);
      if (g !== current) {
        current = g;
        const h = document.createElement('h3'); h.textContent = g; el.history.appendChild(h);
      }
      const row = document.createElement('div');
      row.className = 'conv' + (c.id === state.convId ? ' active' : '');
      row.innerHTML = `<button class="open" title="${escapeHtml(c.title || 'Untitled')}">${escapeHtml(c.title || 'Untitled')}</button>` +
        '<button class="del" aria-label="Delete chat" title="Delete chat">✕</button>';
      $('.open', row).addEventListener('click', () => openConversation(c.id));
      $('.del', row).addEventListener('click', async () => {
        if (!confirm('Delete this chat? This cannot be undone.')) return;
        await api('/v1/conversations/' + c.id, { method: 'DELETE' });
        if (state.convId === c.id) newChat();
        loadConversations();
      });
      el.history.appendChild(row);
    });
  }
  el.search.addEventListener('input', renderHistory);

  function newChat() {
    if (state.generating) stop();
    state.convId = null; state.attached = [];
    el.messages.innerHTML = '';
    el.empty.classList.remove('hidden');
    renderAttachments(); renderHistory();
    el.input.focus();
    el.sidebar.classList.remove('open');
  }
  $('#newChat').addEventListener('click', newChat);

  async function openConversation(id) {
    if (state.generating) stop();
    const conv = await api('/v1/conversations/' + id);
    state.convId = id; state.attached = [];
    renderAttachments();
    el.messages.innerHTML = '';
    el.empty.classList.toggle('hidden', conv.messages.length > 0);
    conv.messages.forEach((m) => {
      if (m.role === 'user') addUser(m.content);
      else if (m.role === 'assistant') {
        const node = addAssistant();
        const meta = m.meta || {};
        if (meta.roblox) renderSavedRoblox(node, m);
        else finishAssistant(node, m.content, { finish: meta.finish_reason, model: m.model, tokens: m.completion_tokens, sources: meta.sources });
      }
    });
    markLastAssistant();
    renderHistory();
    scrollDown(true);
    el.sidebar.classList.remove('open');
  }

  // ------------------------------------------------------------ messages
  function addUser(text) {
    const node = document.createElement('div');
    node.className = 'msg user';
    node.innerHTML = '<div class="bubble"></div>';
    $('.bubble', node).textContent = text;
    el.messages.appendChild(node);
    return node;
  }

  function addAssistant() {
    const node = document.createElement('div');
    node.className = 'msg assistant';
    node.innerHTML = AVATAR + '<div class="body"><div class="notices"></div><div class="md"></div><div class="extra"></div><div class="msg-actions"></div></div>';
    el.messages.appendChild(node);
    return node;
  }

  let renderPending = null;
  function streamRender(node, text) {
    node._text = text;
    if (renderPending) return;
    renderPending = requestAnimationFrame(() => {
      renderPending = null;
      $('.md', node).innerHTML = MD.render(node._text) + '<span class="caret"></span>';
      scrollDown();
    });
  }

  function renderSources(node, sources) {
    const box = $('.extra', node);
    box.innerHTML = '';
    if (!sources || !sources.length) return;
    const wrap = document.createElement('div');
    wrap.className = 'sources';
    wrap.innerHTML = '<h4>From your documents</h4>' + sources.map((s) =>
      `<div class="source"><b>${escapeHtml(s.label)}</b>${escapeHtml(s.filename)}${s.page ? ', page ' + s.page : ''}<p>${escapeHtml(s.excerpt || '')}</p></div>`).join('');
    box.appendChild(wrap);
  }

  function finishAssistant(node, text, info = {}) {
    node.classList.remove('generating');
    const md = $('.md', node);
    if (node._flow) {
      // Roblox build mode: the body already holds prose segments and tool cards in order.
      node._text = (node._parts || []).join('\n\n').replace(/\n{3,}/g, '\n\n');
      const lone = md.querySelector(':scope > .caret');
      if (lone) lone.remove();
      if (!md.children.length && !info.error) md.innerHTML = '<p class="error-text">No text was generated.</p>';
      if (info.error) md.insertAdjacentHTML('beforeend', `<p class="error-text">${escapeHtml(info.error)}</p>`);
      if (info.finish === 'max_steps') {
        const n = document.createElement('p');
        n.className = 'rbx-note';
        n.textContent = info.steps ? `Stopped after ${info.steps} steps` : 'Stopped at the step limit';
        md.appendChild(n);
      }
    } else {
      node._text = text;
      if (text) md.innerHTML = MD.render(text);
      else md.innerHTML = info.error ? '' : '<p class="error-text">No text was generated.</p>';
      if (info.error) md.insertAdjacentHTML('beforeend', `<p class="error-text">${escapeHtml(info.error)}</p>`);
    }
    renderSources(node, info.sources);
    const notes = [];
    if (info.finish === 'cancelled') notes.push('Stopped');
    if (info.finish === 'length') notes.push('Reached the length limit');
    if (info.finish === 'error') notes.push('Ended with an error');
    const actions = $('.msg-actions', node);
    actions.innerHTML = '<button type="button" data-act="copy">Copy</button>' +
      (node._flow ? '' : '<button type="button" data-act="regen" hidden>Regenerate</button>') +
      `<span class="meta">${escapeHtml(info.model || '')}</span>` +
      (info.tokens ? `<span class="meta">${info.tokens} tokens</span>` : '') +
      notes.map((n) => `<span class="meta">${n}</span>`).join('');
    $('[data-act="copy"]', actions).addEventListener('click', () => copyText(node._text || ''));
    const regen = $('[data-act="regen"]', actions);
    if (regen) regen.addEventListener('click', regenerate);
  }

  function markLastAssistant() {
    const all = el.messages.querySelectorAll('.msg.assistant');
    all.forEach((n, i) => { const b = $('[data-act="regen"]', n); if (b) b.hidden = i !== all.length - 1 || !state.convId; });
  }

  function showMemoryActions(node, actions) {
    const box = $('.notices', node);
    (actions || []).forEach((a) => {
      const n = document.createElement('div');
      n.className = 'notice';
      if (a.action === 'remembered') {
        n.innerHTML = `<span>Saved to memory: ${escapeHtml(a.memory)}</span><button type="button">Undo</button>`;
        $('button', n).addEventListener('click', async () => {
          await api('/v1/memories/' + a.id, { method: 'DELETE' });
          n.innerHTML = '<span>Removed from memory.</span>';
        });
      } else if (a.action === 'forgot') {
        n.innerHTML = `<span>Removed from memory: ${escapeHtml(a.memory)}</span>`;
      }
      box.appendChild(n);
    });
  }

  // ------------------------------------------------------------ sending
  function setGenerating(on) {
    state.generating = on;
    el.send.hidden = on; el.stop.hidden = !on;
    el.rbxStudio.disabled = on;               // a build is pinned to the studio it started in
    let bar = $('.generating-arc', el.thread);
    if (on && !bar) { bar = document.createElement('div'); bar.className = 'generating-arc'; el.thread.prepend(bar); }
    if (!on && bar) bar.remove();
  }

  function requestBody(extra) {
    const body = Object.assign({
      model: state.model, stream: true, profile: state.profile,
      use_documents: state.prefs.use_documents !== false,
      document_ids: state.attached.length ? state.attached.map((d) => d.id) : null,
    }, extra);
    if (state.prefs.temperature !== '' && state.prefs.temperature != null) body.temperature = Number(state.prefs.temperature);
    if (state.prefs.max_tokens) body.max_tokens = Number(state.prefs.max_tokens);
    if (state.convId) body.conversation_id = state.convId;
    return body;
  }

  async function send(text) {
    text = (text || '').trim();
    if (!text || state.generating || !state.model) return;
    el.empty.classList.add('hidden');
    addUser(text);
    el.input.value = ''; autosize();
    const node = addAssistant();
    scrollDown(true);
    if (state.rbx.on) await robloxStream(text, node);
    else await stream(requestBody({ message: text }), node);
  }

  async function regenerate() {
    if (state.generating || !state.convId) return;
    const all = el.messages.querySelectorAll('.msg.assistant');
    if (all.length) all[all.length - 1].remove();
    const node = addAssistant();
    await stream(requestBody({ regenerate: true }), node);
  }

  async function stream(body, node) {
    node.classList.add('generating');
    $('.md', node).innerHTML = '<span class="caret"></span>';
    setGenerating(true);
    state.controller = new AbortController();
    let text = '', finish = null, usage = null, meta = null, error = null;
    try {
      const res = await api('/v1/chat/completions', { method: 'POST', json: body, raw: true, signal: state.controller.signal });
      if (!res.ok) throw await responseError(res);
      await readSSE(res, (obj) => {
        if (obj.error) throw new Error(obj.error.message);
        if (obj.zenit) {
          meta = obj.zenit; state.genId = meta.generation_id;
          if (meta.conversation_id && meta.conversation_id !== state.convId) { state.convId = meta.conversation_id; loadConversations(); }
          showMemoryActions(node, meta.memory_actions);
          renderSources(node, meta.sources);
        }
        const ch = obj.choices && obj.choices[0];
        if (ch && ch.delta && ch.delta.content) { text += ch.delta.content; streamRender(node, text); }
        if (ch && ch.finish_reason) finish = ch.finish_reason;
        if (obj.usage) usage = obj.usage;
      });
    } catch (err) {
      if (err.name === 'AbortError') finish = 'cancelled';
      else error = err.message;
    } finally {
      if (renderPending) { cancelAnimationFrame(renderPending); renderPending = null; }
      setGenerating(false);
      state.controller = null; state.genId = null;
      finishAssistant(node, text, { finish, model: meta ? meta.model : state.model, tokens: usage ? usage.completion_tokens : null,
        sources: meta ? meta.sources : null, error });
      markLastAssistant();
      loadConversations().catch(() => {});
      scrollDown();
    }
  }

  async function responseError(res) {
    let d = null; try { d = await res.json(); } catch { /* not json */ }
    if (res.status === 401) {
      if (state.auth.enabled && !state.key) showSignIn(false);
      else showConnect('Your API key was not accepted.');
    }
    const err = new Error((d && d.error && d.error.message) || res.statusText || ('Request failed (' + res.status + ')'));
    err.status = res.status;
    return err;
  }

  // Reads a text/event-stream body and calls onData with each parsed "data: {json}" event.
  async function readSSE(res, onData) {
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6);
          if (payload === '[DONE]') continue;
          onData(JSON.parse(payload));
        }
      }
    }
  }

  // ------------------------------------------------------- roblox build mode
  function setRobloxMode(on) {
    state.rbx.on = !!on;
    store.set('roblox', state.rbx.on);
    el.rbxToggle.setAttribute('aria-pressed', String(state.rbx.on));
    el.rbxToggle.classList.toggle('on', state.rbx.on);
    el.composer.classList.toggle('roblox', state.rbx.on);
    el.input.placeholder = state.rbx.on ? ROBLOX_PLACEHOLDER : DEFAULT_PLACEHOLDER;
    el.attach.disabled = state.rbx.on;
    el.attach.title = state.rbx.on ? 'Files are not used in Roblox build mode' : 'Attach files';
    el.rbxBar.hidden = !state.rbx.on;
    clearInterval(state.rbx.timer); state.rbx.timer = null;
    if (state.rbx.on) { checkRoblox(); state.rbx.timer = setInterval(checkRoblox, 15000); }
    if (state.models.length) describeModel();
  }
  el.rbxToggle.addEventListener('click', () => { setRobloxMode(!state.rbx.on); el.input.focus(); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && state.rbx.on) checkRoblox(); });

  async function checkRoblox() {
    // No polling during a build (it would race the agent's tool calls) or while the tab is hidden
    // (each poll may try to start the Studio MCP proxy).
    if (state.rbx.checking || state.generating || document.hidden) return;
    state.rbx.checking = true;
    try { renderRobloxStatus(await api('/v1/roblox/status')); }
    catch (err) { renderRobloxStatus({ available: false, connected: false, studios: [], studio_id: null, error: err.message }); }
    finally { state.rbx.checking = false; }
  }

  function renderRobloxStatus(s) {
    s = s || {};
    state.rbx.status = s;
    const studios = Array.isArray(s.studios) ? s.studios : [];
    const ok = !!(s.available && s.connected);
    const active = studios.find((x) => x.id === s.studio_id);
    // The request uses the model in the picker; fall back to the server default's flag.
    const m = currentModel();
    const tools = typeof m.supports_tools === 'boolean' ? m.supports_tools : s.model_supports_tools !== false;
    el.rbxPill.classList.toggle('ok', ok && tools);
    el.rbxPill.classList.toggle('warn', ok && !tools);
    $('.label', el.rbxPill).textContent = !ok ? 'Studio not connected' : tools ? 'Studio connected' : "Model can't use tools";
    let tip = ok ? 'Connected to ' + ((active && (active.name || active.id)) || 'Roblox Studio') + '.'
      : (s.error || (s.available === false ? 'Roblox Studio tools are not available on this server.' : 'Open Roblox Studio with the MCP server enabled.'));
    if (!tools) tip += ` ${m.id || 'The selected model'} cannot call tools; pick a model that can.`;
    el.rbxPill.title = tip;
    el.rbxStudio.hidden = studios.length < 2;
    // Do not rebuild the list while the user has it open.
    if (studios.length >= 2 && document.activeElement !== el.rbxStudio) {
      el.rbxStudio.innerHTML = '';
      studios.forEach((st) => {
        const o = document.createElement('option');
        o.value = String(st.id); o.textContent = st.name ? `${st.name} (${st.id})` : String(st.id);
        el.rbxStudio.appendChild(o);
      });
      el.rbxStudio.value = s.studio_id || '';
    }
  }

  el.rbxStudio.addEventListener('change', async () => {
    const id = el.rbxStudio.value;
    try {
      renderRobloxStatus(await api('/v1/roblox/studio', { method: 'POST', json: { studio_id: id } }));
      toast('Building in ' + (el.rbxStudio.selectedOptions[0] ? el.rbxStudio.selectedOptions[0].textContent : id));
    } catch (err) {
      toast('Could not switch Studio: ' + err.message, 'error');
      el.rbxStudio.blur();
      checkRoblox();
    }
  });

  // One-line description of a tool call for the collapsed card.
  function toolSummary(name, args) {
    if (typeof args === 'string') return args;
    if (!args || typeof args !== 'object') return '';
    if (name === 'execute_luau' && typeof args.code === 'string') {
      const line = args.code.split('\n').map((l) => l.trim()).find((l) => l && !l.startsWith('--')) || '';
      return (args.datamodel_type && args.datamodel_type !== 'Edit' ? args.datamodel_type + ': ' : '') + line;
    }
    return Object.entries(args).filter(([k]) => k !== 'studio_id')
      .map(([k, v]) => k + ': ' + (typeof v === 'string' ? v : JSON.stringify(v))).join(', ');
  }

  function codeBox(text, label) {
    const box = document.createElement('div');
    box.className = 'codeblock';
    box.innerHTML = '<div class="code-head"><span></span><button type="button" data-copy-code>Copy code</button></div><pre><code></code></pre>';
    $('.code-head span', box).textContent = label;
    $('code', box).textContent = text;
    return box;
  }

  function toolCard(call) {
    const name = String(call.name || 'tool');
    const args = call.arguments;
    const card = document.createElement('details');
    card.className = 'tool-card running';
    card.innerHTML = '<summary>' + CUBE + '<span class="tool-name"></span><span class="tool-sum"></span><span class="tool-state">Running</span></summary><div class="tool-body"></div>';
    $('.tool-name', card).textContent = name;
    $('.tool-sum', card).textContent = toolSummary(name, args).replace(/\s+/g, ' ').slice(0, 200);
    const body = $('.tool-body', card);
    if (name === 'execute_luau' && args && typeof args.code === 'string') {
      body.appendChild(codeBox(args.code, 'luau' + (args.datamodel_type ? ', ' + args.datamodel_type : '')));
    } else {
      let shown = args;
      if (shown && typeof shown === 'object' && !Array.isArray(shown)) { shown = Object.assign({}, shown); delete shown.studio_id; }
      body.appendChild(codeBox(typeof shown === 'string' ? shown : JSON.stringify(shown == null ? {} : shown, null, 2), 'arguments'));
    }
    return card;
  }

  // ok: true, false, or null when a saved call has no recorded result.
  function toolResult(card, ok, output) {
    card.classList.remove('running', 'ok', 'fail', 'unknown');
    card.classList.add(ok === true ? 'ok' : ok === false ? 'fail' : 'unknown');
    $('.tool-state', card).textContent = ok === true ? 'Done' : ok === false ? 'Failed' : '';
    const prev = $('.tool-out', card);
    if (prev) prev.remove();
    if (ok == null && !output) return;
    const out = document.createElement('details');
    out.className = 'tool-out';
    out.innerHTML = '<summary>Output</summary><pre></pre>';
    $('pre', out).textContent = output ? String(output) : '(no output)';
    $('.tool-body', card).appendChild(out);
  }

  // The assistant body in build mode is a sequence of prose segments and tool cards.
  function flowAppend(node, child) {
    const md = $('.md', node);
    const lone = md.querySelector(':scope > .caret');
    if (lone) lone.remove();
    md.appendChild(child);
  }

  let flowPending = null;
  function flowToken(node, t) {
    if (!node._seg) {
      node._seg = document.createElement('div');
      node._seg.className = 'seg';
      node._segText = '';
      flowAppend(node, node._seg);
    }
    node._segText += t;
    if (flowPending) return;
    flowPending = requestAnimationFrame(() => {
      flowPending = null;
      if (node._seg) node._seg.innerHTML = MD.render(node._segText) + '<span class="caret"></span>';
      scrollDown();
    });
  }

  function flowCloseSeg(node) {
    if (flowPending) { cancelAnimationFrame(flowPending); flowPending = null; }
    if (node._seg) {
      if (node._segText.trim()) { node._seg.innerHTML = MD.render(node._segText); node._parts.push(node._segText.trim()); }
      else node._seg.remove();
    }
    node._seg = null; node._segText = '';
  }

  function flowNote(node, message) {
    flowCloseSeg(node);
    const p = document.createElement('p');
    p.className = 'error-text';
    p.textContent = message;
    flowAppend(node, p);
  }

  async function robloxStream(text, node) {
    node.classList.add('generating', 'roblox');
    Object.assign(node, { _flow: true, _parts: [], _cards: new Map(), _seg: null, _segText: '' });
    $('.md', node).innerHTML = '<span class="caret"></span>';
    setGenerating(true);
    state.controller = new AbortController();
    let finish = null, steps = 0, model = state.model, error = null, preStream = false;
    const body = { message: text, model: state.model || null };
    if (state.convId) body.conversation_id = state.convId;
    try {
      const res = await api('/v1/roblox/chat', { method: 'POST', json: body, raw: true, signal: state.controller.signal });
      if (!res.ok) { preStream = true; throw await responseError(res); }
      await readSSE(res, (ev) => {
        if (!ev || typeof ev !== 'object') return;
        if (!ev.type && ev.error) throw new Error(ev.error.message || 'Request failed');
        switch (ev.type) {
          case 'meta':
            state.genId = ev.generation_id || null;
            if (ev.model) model = ev.model;
            if (ev.conversation_id && ev.conversation_id !== state.convId) { state.convId = ev.conversation_id; loadConversations().catch(() => {}); }
            break;
          case 'token':
            if (ev.text) flowToken(node, String(ev.text));
            break;
          case 'tool_call': {
            flowCloseSeg(node);
            const card = toolCard(ev);
            node._cards.set(ev.id, card);
            flowAppend(node, card);
            scrollDown();
            break;
          }
          case 'tool_result': {
            let card = node._cards.get(ev.id);
            if (!card) { flowCloseSeg(node); card = toolCard({ name: 'tool' }); node._cards.set(ev.id, card); flowAppend(node, card); }
            toolResult(card, ev.ok === true, ev.output);
            scrollDown();
            break;
          }
          case 'error':
            toast(ev.message || 'Something went wrong in build mode', 'error');
            flowNote(node, ev.message || 'Something went wrong.');
            break;
          case 'done':
            finish = ev.finish_reason || null;
            steps = ev.steps || 0;
            break;
          default: break;
        }
      });
    } catch (err) {
      if (err.name === 'AbortError') finish = 'cancelled';
      else { error = err.message; if (!preStream) toast(err.message, 'error'); }
    } finally {
      flowCloseSeg(node);
      setGenerating(false);
      state.controller = null; state.genId = null;
      node._cards.forEach((card) => { if (card.classList.contains('running')) toolResult(card, null, ''); });
      finishAssistant(node, null, { finish, steps, model, error });
      markLastAssistant();
      loadConversations().catch(() => {});
      if (preStream) checkRoblox();
      scrollDown();
    }
  }

  function renderSavedRoblox(node, m) {
    const meta = m.meta || {};
    Object.assign(node, { _flow: true, _parts: [] });
    node.classList.add('roblox');
    const md = $('.md', node);
    md.innerHTML = '';
    (Array.isArray(meta.tool_calls) ? meta.tool_calls : []).forEach((c) => {
      if (!c || typeof c !== 'object') return;
      const card = toolCard({ name: c.name, arguments: c.arguments });
      toolResult(card, c.ok === true ? true : c.ok === false ? false : null, c.output);
      md.appendChild(card);
    });
    if (m.content) {
      const seg = document.createElement('div');
      seg.className = 'seg';
      seg.innerHTML = MD.render(m.content);
      md.appendChild(seg);
      node._parts.push(m.content);
    }
    finishAssistant(node, null, { finish: meta.finish_reason, steps: meta.steps, model: m.model, tokens: m.completion_tokens,
      error: typeof meta.error === 'string' ? meta.error : null });
  }

  function stop() {
    if (state.genId) api('/v1/generations/' + state.genId + '/cancel', { method: 'POST' }).catch(() => {});
    if (state.controller) state.controller.abort();
  }
  el.stop.addEventListener('click', stop);

  el.composer.addEventListener('submit', (e) => { e.preventDefault(); send(el.input.value); });
  el.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(el.input.value); }
  });
  function autosize() { el.input.style.height = 'auto'; el.input.style.height = Math.min(el.input.scrollHeight, 240) + 'px'; }
  el.input.addEventListener('input', autosize);

  el.messages.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-copy-code]');
    if (btn) copyText(btn.closest('.codeblock').querySelector('code').textContent);
  });

  // ------------------------------------------------------------- uploads
  function renderAttachments() {
    el.attachments.innerHTML = '';
    state.attached.forEach((d) => {
      const chip = document.createElement('span');
      chip.className = 'file-chip' + (d.busy ? ' busy' : '');
      chip.innerHTML = `<span>${escapeHtml(d.filename)}${d.busy ? ', reading' : ''}</span>` + (d.busy ? '' : '<button type="button" aria-label="Remove from this chat">✕</button>');
      const b = $('button', chip);
      if (b) b.addEventListener('click', () => { state.attached = state.attached.filter((x) => x !== d); renderAttachments(); });
      el.attachments.appendChild(chip);
    });
  }

  async function upload(files) {
    for (const f of files) {
      const item = { filename: f.name, busy: true };
      state.attached.push(item); renderAttachments();
      try {
        const doc = await api('/v1/documents?filename=' + encodeURIComponent(f.name), {
          method: 'POST', body: f, headers: { 'Content-Type': f.type || 'application/octet-stream' } });
        Object.assign(item, doc, { busy: false });
        toast(`${f.name} is ready (${doc.n_chunks} passage${doc.n_chunks === 1 ? '' : 's'})`);
      } catch (err) {
        state.attached = state.attached.filter((x) => x !== item);
        toast(`Could not read ${f.name}: ${err.message}`, 'error');
      }
      renderAttachments();
    }
  }
  el.attach.addEventListener('click', () => el.file.click());
  el.file.addEventListener('change', () => { upload([...el.file.files]); el.file.value = ''; });
  ['dragenter', 'dragover'].forEach((t) => el.composer.addEventListener(t, (e) => {
    e.preventDefault();
    if (!state.rbx.on) el.composer.classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach((t) => el.composer.addEventListener(t, (e) => { e.preventDefault(); el.composer.classList.remove('dragging'); }));
  el.composer.addEventListener('drop', (e) => {
    if (!e.dataTransfer.files.length) return;
    if (state.rbx.on) { toast('Files are not used in Roblox build mode'); return; }
    upload([...e.dataTransfer.files]);
  });

  // -------------------------------------------------------------- drawer
  const panels = {
    async memory() {
      const data = await api('/v1/memories');
      state.memoryEnabled = data.enabled;
      el.drawerBody.innerHTML = `
        <div class="row between"><label class="toggle"><input type="checkbox" id="memToggle" ${data.enabled ? 'checked' : ''}> Let ZENIT use memory</label></div>
        <p class="hint">ZENIT saves a memory only when you say something like "remember that..." or "my name is...". Say "forget ..." to remove one, or manage them here.</p>
        <form class="row" id="memAdd"><input class="grow" id="memText" placeholder="Add something to remember" style="flex:1;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:8px 10px"><button class="btn">Add</button></form>
        <div id="memList"></div>
        <h3>Your data</h3>
        <div class="row"><button class="btn" id="memExport">Export everything</button><button class="btn danger" id="memWipe">Delete all memories</button></div>`;
      const list = $('#memList');
      if (!data.data.length) list.innerHTML = '<p class="hint">No memories saved.</p>';
      data.data.forEach((m) => {
        const item = document.createElement('div');
        item.className = 'list-item';
        item.innerHTML = `<div class="text"></div><div class="sub">${m.pinned ? 'Pinned, ' : ''}${escapeHtml(m.kind)}, updated ${fmtDate(m.updated)}</div>
          <div class="actions"><button class="btn small" data-a="edit">Edit</button><button class="btn small" data-a="pin">${m.pinned ? 'Unpin' : 'Pin'}</button><button class="btn small danger" data-a="del">Delete</button></div>`;
        $('.text', item).textContent = m.content;
        item.addEventListener('click', async (e) => {
          const a = e.target.dataset.a;
          if (a === 'del') { await api('/v1/memories/' + m.id, { method: 'DELETE' }); panels.memory(); }
          if (a === 'pin') { await api('/v1/memories/' + m.id, { method: 'PATCH', json: { pinned: !m.pinned } }); panels.memory(); }
          if (a === 'edit') {
            const next = prompt('Edit memory', m.content);
            if (next && next.trim()) { await api('/v1/memories/' + m.id, { method: 'PATCH', json: { content: next.trim() } }); panels.memory(); }
          }
        });
        list.appendChild(item);
      });
      $('#memToggle').addEventListener('change', async (e) => { await api('/v1/settings', { method: 'PUT', json: { memory_enabled: e.target.checked } }); toast(e.target.checked ? 'Memory on' : 'Memory off'); });
      $('#memAdd').addEventListener('submit', async (e) => {
        e.preventDefault(); const v = $('#memText').value.trim(); if (!v) return;
        await api('/v1/memories', { method: 'POST', json: { content: v } }); panels.memory();
      });
      $('#memExport').addEventListener('click', async () => {
        const data2 = await api('/v1/export');
        const blob = new Blob([JSON.stringify(data2, null, 2)], { type: 'application/json' });
        const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'zenit-export.json'; a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 2000);
      });
      $('#memWipe').addEventListener('click', async () => {
        if (!confirm('Delete every saved memory? Conversations stay.')) return;
        await api('/v1/memories', { method: 'DELETE' }); panels.memory();
      });
    },

    async documents() {
      const data = await api('/v1/documents');
      el.drawerBody.innerHTML = `<p class="hint">Uploaded files stay on your ZENIT server. Answers that use them cite passages as [S1], [S2], kept apart from what ZENIT learned in training.</p>
        <div class="row"><button class="btn primary" id="docUpload">Upload files</button></div><div id="docList"></div>`;
      $('#docUpload').addEventListener('click', () => el.file.click());
      const list = $('#docList');
      if (!data.data.length) list.innerHTML = '<p class="hint">No documents yet. Supported: PDF, DOCX, TXT, Markdown, HTML, CSV and code files.</p>';
      data.data.forEach((d) => {
        const item = document.createElement('div');
        item.className = 'list-item';
        const inChat = state.attached.some((x) => x.id === d.id);
        item.innerHTML = `<div class="text"></div><div class="sub">${fmtBytes(d.bytes)}, ${d.n_chunks} passage${d.n_chunks === 1 ? '' : 's'}, added ${fmtDate(d.created)}</div>
          <div class="actions"><button class="btn small" data-a="use" ${inChat ? 'disabled' : ''}>${inChat ? 'In this chat' : 'Use in this chat'}</button><button class="btn small danger" data-a="del">Delete</button></div>`;
        $('.text', item).textContent = d.filename;
        item.addEventListener('click', async (e) => {
          const a = e.target.dataset.a;
          if (a === 'use') { state.attached.push(d); renderAttachments(); panels.documents(); }
          if (a === 'del') {
            if (!confirm(`Delete ${d.filename}?`)) return;
            await api('/v1/documents/' + d.id, { method: 'DELETE' });
            state.attached = state.attached.filter((x) => x.id !== d.id); renderAttachments(); panels.documents();
          }
        });
        list.appendChild(item);
      });
    },

    async settings() {
      let health = {};
      try { health = await api('/v1/health'); } catch { /* shown below */ }
      el.drawerBody.innerHTML = `
        <h3>Responses</h3>
        <label class="field">Temperature (blank uses the style default)<input id="sTemp" type="number" min="0" max="2" step="0.05" value="${escapeHtml(String(state.prefs.temperature))}"></label>
        <label class="field">Maximum reply length in tokens (blank uses the default)<input id="sMax" type="number" min="1" max="32768" value="${escapeHtml(String(state.prefs.max_tokens))}"></label>
        <label class="toggle" style="margin-bottom:12px"><input type="checkbox" id="sDocs" ${state.prefs.use_documents !== false ? 'checked' : ''}> Search my documents when relevant</label>
        <h3>Appearance</h3>
        <div class="row"><button class="btn" id="sTheme">Use ${state.theme === 'dark' ? 'light' : 'dark'} theme</button></div>
        <h3>Account</h3>
        <div id="sAccount"></div>
        <h3>Connection</h3>
        <p class="hint">${state.me && state.me.signed_in ? 'Signed in with Google on this PC.' : state.me && state.me.local_mode ? 'Local mode: this PC connects without a key.' : 'Connected with key ' + escapeHtml(state.me ? state.me.key_prefix : '') + '...'}</p>
        <div class="row"><button class="btn" id="sKey">Change API key</button><a class="btn" href="/dashboard">Open training dashboard</a></div>
        <dl class="facts"><dt>Server</dt><dd>${escapeHtml(health.status || 'unreachable')}</dd><dt>Version</dt><dd>${escapeHtml(health.version || '')}</dd><dt>Models</dt><dd>${health.models ?? ''}</dd></dl>`;
      const save = () => {
        state.prefs.temperature = $('#sTemp').value; state.prefs.max_tokens = $('#sMax').value;
        state.prefs.use_documents = $('#sDocs').checked; store.set('prefs', state.prefs);
      };
      ['#sTemp', '#sMax', '#sDocs'].forEach((s) => $(s).addEventListener('change', save));
      $('#sTheme').addEventListener('click', () => { $('#themeToggle').click(); panels.settings(); });
      $('#sKey').addEventListener('click', () => showConnect());
      renderAccountSettings($('#sAccount'));
    },

    async model() {
      const m = currentModel();
      if (!m.id) { el.drawerBody.innerHTML = '<p class="hint">No model selected.</p>'; return; }
      const info = await api('/v1/models/' + encodeURIComponent(m.id));
      const card = info.model_card || {};
      let evals = [];
      try { evals = (await api('/v1/evals')).data.filter((e) => e.model === m.id); } catch { /* no evals yet */ }
      const tiny = card.parameters && card.parameters < 20e6;
      const ds = card.dataset || {};
      el.drawerBody.innerHTML = `
        ${tiny ? '<div class="callout">Experimental build. It proves the training, inference and API pipeline end to end. It does not have broad knowledge.</div>' : ''}
        <dl class="facts">
          <dt>Model</dt><dd>${escapeHtml(info.id)}</dd>
          <dt>Version</dt><dd>${escapeHtml(card.version || info.version || '')}</dd>
          <dt>Stage</dt><dd>${escapeHtml(card.stage || '')}</dd>
          <dt>Parameters</dt><dd>${card.parameters ? card.parameters.toLocaleString() : 'unknown'}</dd>
          <dt>Context</dt><dd>${info.context_window} tokens</dd>
          <dt>Runs on</dt><dd>${escapeHtml((info.backend || info.engine || '') + ' ' + (info.device || ''))}</dd>
          <dt>Architecture</dt><dd>${escapeHtml(card.architecture || info.engine || '')}</dd>
          <dt>Origin</dt><dd>${card.base_model ? 'Fine-tuned from ' + escapeHtml(card.base_model) : card.trained_from_scratch ? 'Trained from scratch by this project' : 'See model card'}</dd>
          <dt>Training data</dt><dd>${ds.tokens ? fmtNum(ds.tokens.train) + ' training tokens' : 'unknown'}</dd>
          <dt>Best val loss</dt><dd>${card.best_val_loss ?? 'not recorded'}</dd>
          <dt>Training run</dt><dd>${escapeHtml(card.run_id || '')}</dd>
        </dl>
        <h3>Evaluations</h3>
        ${evals.length ? evals.map((e) => `<div class="list-item"><div class="sub">${escapeHtml(e.created || '')}, suite ${escapeHtml(e.suite_hash || '')}</div>${
          Object.entries(e.summary || {}).map(([k, v]) => `<div>${escapeHtml(k)}: ${typeof v === 'number' ? (v * 100).toFixed(1) + '%' : escapeHtml(String(v))}</div>`).join('')}</div>`).join('')
          : '<p class="hint">No evaluation results for this model yet. Run python -m zenit.evals.run --model ' + escapeHtml(m.id) + '</p>'}`;
    },
  };
  const titles = { memory: 'Memory', documents: 'Documents', settings: 'Settings', model: 'Model info' };

  async function openPanel(name) {
    el.drawerTitle.textContent = titles[name];
    el.drawerBody.innerHTML = '<p class="hint">Loading</p>';
    el.drawer.classList.add('open'); el.drawer.setAttribute('aria-hidden', 'false');
    try { await panels[name](); } catch (err) { el.drawerBody.innerHTML = `<p class="error-text">${escapeHtml(err.message)}</p>`; }
  }
  function closePanel() { el.drawer.classList.remove('open'); el.drawer.setAttribute('aria-hidden', 'true'); }
  document.querySelectorAll('[data-panel]').forEach((b) => b.addEventListener('click', () => openPanel(b.dataset.panel)));
  $('#closeDrawer').addEventListener('click', closePanel);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { closePanel(); el.sidebar.classList.remove('open'); } });
  $('#openSidebar').addEventListener('click', () => el.sidebar.classList.add('open'));
  $('#closeSidebar').addEventListener('click', () => el.sidebar.classList.remove('open'));

  // ------------------------------------------------- Sign in with Google
  let gsiLoading = null;
  function loadGsi() {
    if (window.google && window.google.accounts && window.google.accounts.id) return Promise.resolve();
    if (!gsiLoading) {
      gsiLoading = new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = 'https://accounts.google.com/gsi/client';
        s.async = true; s.defer = true;
        s.onload = () => resolve();
        s.onerror = () => { gsiLoading = null; s.remove(); reject(new Error('Could not load Google sign-in. Check your internet connection.')); };
        document.head.appendChild(s);
      });
    }
    return gsiLoading;
  }

  async function renderGsiButton(slot) {
    slot.innerHTML = '<span class="hint">Loading Google sign-in</span>';
    try {
      await loadGsi();
      if (!state.gsiReady) {
        window.google.accounts.id.initialize({
          client_id: state.auth.google_client_id, callback: onCredential,
          ux_mode: 'popup', auto_select: false, cancel_on_tap_outside: true, itp_support: true,
        });
        state.gsiReady = true;
      }
      slot.innerHTML = '';
      window.google.accounts.id.renderButton(slot, {
        type: 'standard', theme: state.theme === 'dark' ? 'filled_black' : 'outline', size: 'large',
        text: 'signin_with', shape: 'pill', logo_alignment: 'left', width: 280,
      });
    } catch (err) {
      slot.innerHTML = '';
      const p = document.createElement('p'); p.className = 'form-error'; p.textContent = err.message;
      slot.appendChild(p);
    }
  }

  function showSignIn(canSkip) {
    if (!state.auth.enabled) return;
    $('#signinError').textContent = '';
    $('#signinSkip').hidden = !(canSkip && state.auth.local_access);
    const hostNote = $('#signinHost');
    if (location.hostname !== 'localhost') {
      // Google only accepts page origins registered in Google Cloud; ZENIT documents http://localhost.
      const url = location.protocol + '//localhost' + (location.port ? ':' + location.port : '') + '/';
      hostNote.textContent = 'Google sign-in works on ';
      const a = document.createElement('a'); a.href = url; a.textContent = url;
      hostNote.append(a, '. Open ZENIT there if the button does not work.');
      hostNote.hidden = false;
    } else hostNote.hidden = true;
    el.signin.hidden = false;
    document.body.classList.add('signin-open');
    renderGsiButton($('#gsiButton'));
  }

  function hideSignIn() {
    el.signin.hidden = true;
    document.body.classList.remove('signin-open');
  }

  $('#signinSkip').addEventListener('click', () => {
    store.set('signinSkipped', true);
    hideSignIn();
    if (!state.me) boot();
  });

  function resetThread() {
    if (state.generating) stop();
    state.convId = null; state.attached = []; state.convs = [];
    el.messages.innerHTML = '';
    el.empty.classList.remove('hidden');
    renderAttachments(); renderHistory();
  }

  async function onCredential(resp) {
    const err = $('#signinError');
    err.textContent = '';
    try {
      await api('/v1/auth/google', { method: 'POST', json: { credential: resp && resp.credential } });
    } catch (e) {
      err.textContent = 'Sign-in failed: ' + e.message;
      return;
    }
    state.key = ''; store.set('key', '');      // the Google session replaces any saved API key
    store.set('signinSkipped', false);
    hideSignIn();
    resetThread();
    await boot();
    if (state.me && state.me.signed_in) toast('Signed in as ' + (state.me.name || state.me.email || 'you'));
  }

  async function signOut() {
    try { await api('/v1/auth/logout', { method: 'POST' }); }
    catch (e) { toast('Could not sign out: ' + e.message, 'error'); return; }
    try { if (window.google && window.google.accounts) window.google.accounts.id.disableAutoSelect(); } catch { /* not loaded */ }
    store.set('signinSkipped', false);
    closePanel();
    resetThread();
    state.me = null;
    toast('Signed out');
    await boot();
  }

  function avatarNode(me) {
    const wrap = document.createElement('span');
    wrap.className = 'acct-avatar';
    const initial = (me.given_name || me.name || me.email || '?').trim().charAt(0).toUpperCase() || '?';
    const fallback = () => { wrap.textContent = initial; wrap.classList.add('initial'); };
    if (me.picture && /^https:\/\//.test(me.picture)) {
      const img = document.createElement('img');
      img.alt = ''; img.referrerPolicy = 'no-referrer'; img.decoding = 'async';
      img.addEventListener('error', fallback);
      img.src = me.picture;
      wrap.appendChild(img);
    } else fallback();
    return wrap;
  }

  function renderAccount() {
    const box = el.account;
    box.innerHTML = '';
    const me = state.me;
    if (me && me.signed_in) {
      box.appendChild(avatarNode(me));
      const who = document.createElement('span');
      who.className = 'acct-name';
      who.textContent = me.name || me.email || 'Signed in';
      who.title = me.email || '';
      const out = document.createElement('button');
      out.type = 'button'; out.className = 'acct-out'; out.textContent = 'Sign out';
      out.addEventListener('click', signOut);
      box.append(who, out);
      box.hidden = false;
    } else if (state.auth.enabled) {
      const btn = document.createElement('button');
      btn.type = 'button'; btn.className = 'acct-signin'; btn.textContent = 'Sign in with Google';
      btn.addEventListener('click', () => showSignIn(true));
      box.appendChild(btn);
      box.hidden = false;
    } else box.hidden = true;
  }

  function renderAccountSettings(box) {
    const me = state.me || {};
    box.innerHTML = '';
    const p = document.createElement('p');
    p.className = 'hint';
    const row = document.createElement('div');
    row.className = 'row';
    if (me.signed_in) {
      p.textContent = 'Signed in with Google as ' + (me.name || 'you') + (me.email ? ' (' + me.email + ')' : '') +
        '. ZENIT uses your first name in conversations. Your chats, memory and files belong to this account.';
      const b = document.createElement('button'); b.className = 'btn'; b.textContent = 'Sign out';
      b.addEventListener('click', signOut);
      row.appendChild(b);
      box.append(p, row);
    } else if (state.auth.enabled) {
      p.textContent = 'Sign in with Google so ZENIT knows your name and keeps your chats, memory and files separate.';
      const b = document.createElement('button'); b.className = 'btn primary'; b.textContent = 'Sign in with Google';
      b.addEventListener('click', () => { closePanel(); showSignIn(true); });
      row.appendChild(b);
      box.append(p, row);
    } else {
      p.innerHTML = 'Sign in with Google is off. To turn it on, create an OAuth Web client ID in Google Cloud ' +
        '(authorized JavaScript origin <code>http://localhost:' + escapeHtml(location.port || '8055') + '</code>), ' +
        'put it in <code>configs/auth.json</code> as <code>{"google_client_id": "....apps.googleusercontent.com"}</code> ' +
        'and restart ZENIT. The README section "Sign in with Google" has the steps.';
      box.appendChild(p);
    }
  }

  // ---------------------------------------------------------------- boot
  async function boot() {
    try { state.auth = Object.assign({ enabled: false }, await api('/v1/auth/config')); }
    catch { state.auth = { enabled: false, google_client_id: null, local_access: false }; }
    try {
      state.me = await api('/v1/me');
    } catch (err) {
      state.me = null; renderAccount();
      if (err.status === 401) {
        if (state.auth.enabled && !state.key) showSignIn(false);
        else showConnect();
        return;
      }
      toast('Cannot reach the ZENIT server: ' + err.message, 'error');
      return;
    }
    renderAccount();
    if (state.auth.enabled && !state.me.signed_in && state.me.local_mode && !store.get('signinSkipped', false)) showSignIn(true);
    try { await loadModels(); } catch (err) { toast('Could not load models: ' + err.message, 'error'); }
    try { await loadConversations(); } catch (err) { toast('Could not load chats: ' + err.message, 'error'); }
    setRobloxMode(state.rbx.on);
    el.input.focus();
  }

  applyTheme();
  boot();
})();
