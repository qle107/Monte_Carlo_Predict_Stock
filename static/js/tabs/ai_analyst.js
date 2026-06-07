/* AI Analyst tab.
 * Endpoints: GET /api/ai/status, GET /api/ai/prompt, GET /api/ai/analyze
 * Workflow: generate prompt -> copy into Claude -> paste reply -> render,
 * or run directly via the API when ANTHROPIC_API_KEY is configured. */
(function () {
  'use strict';

  let _aiTicker = '';
  let _aiBusy = false;

  const $ = (id) => document.getElementById(id);

  function _esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  /* ---------- injected styles (animations / effects) ---------- */

  function _injectStyles() {
    if (document.getElementById('ai-analyst-css')) return;
    const st = document.createElement('style');
    st.id = 'ai-analyst-css';
    st.textContent = `
      @keyframes aiBarGrow { from { width: 0; } }
      @keyframes aiFadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
      @keyframes aiPulse { 0%,100% { box-shadow: 0 0 0 0 rgba(188,140,255,.35); } 50% { box-shadow: 0 0 0 6px rgba(188,140,255,0); } }
      @keyframes aiShimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
      #ai-results > div { animation: aiFadeUp .45s ease both; }
      #ai-results > div:nth-child(2) { animation-delay: .05s; }
      #ai-results > div:nth-child(3) { animation-delay: .10s; }
      #ai-results > div:nth-child(4) { animation-delay: .15s; }
      #ai-results > div:nth-child(5) { animation-delay: .20s; }
      #ai-results > div:nth-child(6) { animation-delay: .25s; }
      #ai-results > div:nth-child(7) { animation-delay: .30s; }
      #ai-results > div:nth-child(8) { animation-delay: .35s; }
      .ai-bar-fill {
        height: 100%; border-radius: 8px; position: relative;
        animation: aiBarGrow .9s cubic-bezier(.25,.8,.3,1) both;
      }
      .ai-bar-fill::after {
        content: ''; position: absolute; inset: 0; border-radius: 8px;
        background: linear-gradient(90deg, transparent 30%, rgba(255,255,255,.18) 50%, transparent 70%);
        background-size: 200% 100%; animation: aiShimmer 2.6s linear infinite;
      }
      .ai-card {
        transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
      }
      .ai-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(0,0,0,.4);
      }
      .ai-chip {
        font-size: 9.5px; font-weight: 800; letter-spacing: .6px; padding: 2px 9px;
        border-radius: 10px; text-transform: uppercase;
      }
      .ai-pop-track { height: 5px; border-radius: 3px; background: rgba(255,255,255,.07); overflow: hidden; margin-top: 4px; }
      .ai-pop-fill { height: 100%; border-radius: 3px; animation: aiBarGrow .8s ease both; }
      .ai-ring {
        display: inline-grid; place-items: center; width: 44px; height: 44px;
        border-radius: 50%; animation: aiPulse 2.4s ease infinite;
      }
      .ai-ring > span {
        display: grid; place-items: center; width: 34px; height: 34px; border-radius: 50%;
        background: #11161c; font-size: 11px; font-weight: 800;
      }
      .ai-lvl-marker { position: absolute; transform: translateX(-50%); text-align: center; }
      .ai-lvl-marker .tick { width: 2px; margin: 0 auto; }
      .ai-lvl-label { font-size: 9.5px; font-weight: 700; white-space: nowrap; }
    `;
    document.head.appendChild(st);
  }

  function _ticker() {
    // Prefer the live global config (dashboard inline script declares
    // `let currentConfig`, which is NOT window.currentConfig but is
    // visible across scripts as a global lexical binding).
    let t = '';
    try {
      if (typeof currentConfig === 'object' && currentConfig && currentConfig.ticker) {
        t = currentConfig.ticker;
      }
    } catch (e) { /* not on the dashboard page */ }
    return ((t || _aiTicker) || '').toUpperCase().trim();
  }

  function _show(id, on, disp) {
    const el = $(id);
    if (el) el.style.display = on ? (disp || 'block') : 'none';
  }

  function _setError(msg) {
    const el = $('ai-error-state');
    if (el) { el.textContent = msg || ''; el.style.display = msg ? 'block' : 'none'; }
  }

  function _setStatus(msg) {
    const el = $('ai-status');
    if (el) { el.textContent = msg || ''; el.style.display = msg ? 'block' : 'none'; }
  }

  /* ---------- open hook (called by switchScannerTab) ---------- */

  window.aiAnalystOnOpen = async function (ticker) {
    if (ticker) _aiTicker = ticker.toUpperCase().trim();
    const badge = $('ai-ticker-badge');
    if (badge) badge.textContent = _ticker() || '-';
    try {
      const res = await fetch('/api/ai/status');
      if (res.ok) {
        const st = await res.json();
        _show('ai-run-btn', !!st.configured, 'inline-flex');
        if (st.busy) _setStatus('An AI analysis is already running on the server…');
      }
    } catch (e) { /* best-effort */ }
  };

  /* ---------- step 1: build the prompt ---------- */

  window.generateAiPrompt = async function () {
    if (_aiBusy) return;
    const t = _ticker();
    if (!t) { _setError('No ticker selected - set one in the header first.'); return; }

    _aiBusy = true;
    _setError('');
    _show('ai-empty-state', false);
    _show('ai-spin', true, 'inline-block');
    const label = $('ai-prompt-label');
    if (label) label.textContent = 'Gathering data…';
    _setStatus('Collecting MC signal, GEX, unusual flow, sentiment, news and macro for ' + t + ' (10-40s)…');

    try {
      const res = await fetch('/api/ai/prompt?ticker=' + encodeURIComponent(t));
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status));

      const ta = $('ai-prompt-text');
      if (ta) ta.value = data.prompt || '';
      const meta = $('ai-prompt-meta');
      if (meta) {
        const kb = data.prompt_chars ? (data.prompt_chars / 1000).toFixed(1) + 'k chars' : '';
        const src = (data.sources_used || []).length;
        const fail = (data.sources_failed || []).length;
        meta.textContent = ' ' + [kb, src + ' sources' + (fail ? (', ' + fail + ' failed') : '')].filter(Boolean).join(' · ');
      }
      _setStatus('');
      _show('ai-prompt-section', true);
      _show('ai-paste-section', true);
    } catch (e) {
      _setStatus('');
      _setError('Prompt build failed: ' + e.message);
      _show('ai-empty-state', true);
    } finally {
      _aiBusy = false;
      _show('ai-spin', false);
      if (label) label.textContent = '📋 Generate Prompt';
    }
  };

  window.copyAiPrompt = function () {
    const ta = $('ai-prompt-text');
    if (!ta || !ta.value) return;
    const done = () => {
      const btn = $('ai-copy-btn');
      if (!btn) return;
      const old = btn.textContent;
      btn.textContent = '✓ Copied';
      setTimeout(() => { btn.textContent = old; }, 1600);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(ta.value).then(done).catch(() => { ta.select(); document.execCommand('copy'); done(); });
    } else {
      ta.select(); document.execCommand('copy'); done();
    }
  };

  /* ---------- step 3a: run server-side via the Anthropic API ---------- */

  window.runAiAnalysis = async function () {
    if (_aiBusy) return;
    const t = _ticker();
    if (!t) { _setError('No ticker selected.'); return; }

    _aiBusy = true;
    _setError('');
    _show('ai-empty-state', false);
    _show('ai-api-spin', true, 'inline-block');
    const label = $('ai-run-label');
    if (label) label.textContent = 'Analyzing…';
    _setStatus('Claude is analyzing ' + t + ' - this can take a few minutes…');

    try {
      const res = await fetch('/api/ai/analyze?ticker=' + encodeURIComponent(t));
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status));
      _setStatus('');
      _renderAnalysis(data.analysis, {
        raw: data.raw,
        model: data.model,
        generatedAt: data.generated_at,
        sourcesUsed: data.sources_used,
        sourcesFailed: data.sources_failed,
      });
    } catch (e) {
      _setStatus('');
      _setError('AI analysis failed: ' + e.message);
    } finally {
      _aiBusy = false;
      _show('ai-api-spin', false);
      if (label) label.textContent = '✨ Run via API';
    }
  };

  /* ---------- step 3b: paste Claude's reply ---------- */

  function _dropStrayClosers(str) {
    // Remove stray '}' / ']' that would close the root object before the end
    // (e.g. the model emits `"summary": "..."},` mid-object).
    let out = '', depth = 0, inStr = false, escd = false;
    for (let i = 0; i < str.length; i++) {
      const ch = str[i];
      if (inStr) {
        out += ch;
        if (escd) escd = false;
        else if (ch === '\\') escd = true;
        else if (ch === '"') inStr = false;
        continue;
      }
      if (ch === '"') { inStr = true; out += ch; continue; }
      if (ch === '{' || ch === '[') depth++;
      else if (ch === '}' || ch === ']') {
        if (depth <= 1 && /\S/.test(str.slice(i + 1))) continue; // stray: would end the root early
        depth--;
      }
      out += ch;
    }
    return out;
  }

  function _extractJson(text) {
    const s = (text || '').trim();
    if (!s) return { parsed: null, truncated: false };

    const tryParse = (str) => { try { return JSON.parse(str); } catch (e) { return null; } };

    let parsed = tryParse(s);
    if (parsed) return { parsed, truncated: false };

    const fence = s.match(/```(?:json)?\s*([\s\S]*?)```/i);
    if (fence) {
      parsed = tryParse(fence[1].trim());
      if (parsed) return { parsed, truncated: false };
    }

    const a = s.indexOf('{');
    const b = s.lastIndexOf('}');
    if (a >= 0 && b > a) {
      parsed = tryParse(s.slice(a, b + 1));
      if (parsed) return { parsed, truncated: false };
      parsed = tryParse(_dropStrayClosers(s.slice(a, b + 1)));
      if (parsed) return { parsed, truncated: false };
    }

    if (a >= 0) {
      let frag = s.slice(a).replace(/```\s*$/, '').trim();
      frag = frag.replace(/,?\s*"[^"]*$/, '').replace(/,\s*$/, '');
      let depth = 0, sq = 0, inStr = false, escd = false;
      for (const ch of frag) {
        if (inStr) {
          if (escd) escd = false;
          else if (ch === '\\') escd = true;
          else if (ch === '"') inStr = false;
          continue;
        }
        if (ch === '"') inStr = true;
        else if (ch === '{') depth++;
        else if (ch === '}') depth--;
        else if (ch === '[') sq++;
        else if (ch === ']') sq--;
      }
      if (inStr) frag += '"';
      frag += ']'.repeat(Math.max(0, sq)) + '}'.repeat(Math.max(0, depth));
      parsed = tryParse(frag);
      if (parsed) return { parsed, truncated: true };
    }

    return { parsed: null, truncated: false };
  }

  window.renderPastedAnalysis = function () {
    const ta = $('ai-response-text');
    const text = ta ? ta.value : '';
    if (!text.trim()) { _setError('Paste Claude’s reply first.'); return; }
    _setError('');
    const { parsed, truncated } = _extractJson(text);
    if (!parsed) {
      _renderAnalysis(null, { raw: text, truncated: false, parseFailed: true });
      return;
    }
    _renderAnalysis(parsed, { truncated });
  };

  /* ---------- rendering ---------- */

  const _BAR_STYLES = {
    bullish:  { grad: 'linear-gradient(90deg,#1a6b2e,#3fb950 70%,#56d364)', glow: 'rgba(63,185,80,.45)',  color: 'var(--green)' },
    sideways: { grad: 'linear-gradient(90deg,#8a6d1f,#d29922 70%,#f2c14e)', glow: 'rgba(210,153,34,.40)', color: 'var(--amber)' },
    bearish:  { grad: 'linear-gradient(90deg,#8f1f2e,#f0556d 70%,#ff7b93)', glow: 'rgba(240,85,109,.40)', color: 'var(--red)' },
  };

  function _probBar(label, kind, pct) {
    const s = _BAR_STYLES[kind];
    const p = Math.max(0, Math.min(100, Number(pct) || 0));
    return (
      '<div style="display:flex;align-items:center;gap:10px;margin:7px 0;">' +
        '<span style="width:66px;font-size:10.5px;font-weight:700;letter-spacing:.5px;color:var(--muted);text-transform:uppercase;">' + label + '</span>' +
        '<div style="flex:1;height:16px;background:rgba(255,255,255,.05);border-radius:8px;overflow:hidden;">' +
          '<div class="ai-bar-fill" style="width:' + p + '%;background:' + s.grad + ';box-shadow:0 0 12px ' + s.glow + ';"></div>' +
        '</div>' +
        '<span style="width:52px;text-align:right;font-size:13px;font-weight:800;color:' + s.color + ';text-shadow:0 0 14px ' + s.glow + ';">' + p.toFixed(0) + '%</span>' +
      '</div>'
    );
  }

  function _confidenceRing(pct) {
    const p = Math.max(0, Math.min(100, Number(pct) || 0));
    const col = p >= 65 ? '#3fb950' : p >= 40 ? '#d29922' : '#f0556d';
    return (
      '<span class="ai-ring" style="background:conic-gradient(' + col + ' ' + (p * 3.6) + 'deg, rgba(255,255,255,.07) 0);">' +
        '<span style="color:' + col + ';">' + p.toFixed(0) + '%</span>' +
      '</span>'
    );
  }

  function _levelsLadder(k) {
    const nums = (arr) => (arr || []).map(Number).filter((n) => isFinite(n));
    const sup = nums(k.support);
    const res = nums(k.resistance);
    const gm = (k.gamma_magnet != null && isFinite(Number(k.gamma_magnet))) ? Number(k.gamma_magnet) : null;
    const all = sup.concat(res, gm != null ? [gm] : []);
    if (all.length < 2) return '';

    const lo = Math.min.apply(null, all);
    const hi = Math.max.apply(null, all);
    const pad = (hi - lo) * 0.10 || Math.abs(lo) * 0.02 || 1;
    const a = lo - pad, b = hi + pad;
    const x = (v) => Math.max(2, Math.min(98, ((v - a) / (b - a)) * 100));
    const fmtN = (v) => Number(v) >= 100 ? Number(v).toFixed(0) : Number(v).toFixed(2).replace(/\.?0+$/, '');

    let marks = '';
    res.forEach((v) => {
      marks +=
        '<div class="ai-lvl-marker" style="left:' + x(v) + '%;top:0;">' +
          '<div class="ai-lvl-label" style="color:var(--red);">' + fmtN(v) + '</div>' +
          '<div class="tick" style="height:16px;background:linear-gradient(180deg,var(--red),transparent);box-shadow:0 0 6px rgba(240,85,109,.5);"></div>' +
        '</div>';
    });
    sup.forEach((v) => {
      marks +=
        '<div class="ai-lvl-marker" style="left:' + x(v) + '%;bottom:0;">' +
          '<div class="tick" style="height:16px;background:linear-gradient(0deg,var(--green),transparent);box-shadow:0 0 6px rgba(63,185,80,.5);"></div>' +
          '<div class="ai-lvl-label" style="color:var(--green);">' + fmtN(v) + '</div>' +
        '</div>';
    });
    if (gm != null) {
      marks +=
        '<div class="ai-lvl-marker" style="left:' + x(gm) + '%;top:50%;transform:translate(-50%,-50%);">' +
          '<div title="Gamma magnet" style="width:11px;height:11px;margin:0 auto;background:var(--purple);transform:rotate(45deg);' +
               'box-shadow:0 0 10px rgba(188,140,255,.8);border-radius:2px;"></div>' +
          '<div class="ai-lvl-label" style="color:var(--purple);margin-top:3px;">' + fmtN(gm) + '</div>' +
        '</div>';
    }

    return (
      '<div style="position:relative;height:78px;margin:8px 2px 4px;">' +
        '<div style="position:absolute;left:0;right:0;top:50%;height:2px;transform:translateY(-50%);' +
             'background:linear-gradient(90deg,rgba(63,185,80,.5),rgba(139,148,158,.35),rgba(240,85,109,.5));border-radius:1px;"></div>' +
        marks +
      '</div>'
    );
  }

  function _popMeter(pct) {
    const p = Math.max(0, Math.min(100, Number(pct) || 0));
    const col = p >= 65 ? 'var(--green)' : p >= 45 ? 'var(--amber)' : 'var(--red)';
    return (
      '<div style="font-size:11px;margin-top:6px;"><span style="color:var(--muted);">Est. POP:</span> ' +
        '<strong style="color:' + col + ';">' + p.toFixed(0) + '%</strong>' +
        '<div class="ai-pop-track"><div class="ai-pop-fill" style="width:' + p + '%;background:' + col + ';box-shadow:0 0 8px ' + col + ';"></div></div>' +
      '</div>'
    );
  }

  function _pickCard(p) {
    const sell = (p.action || '').toLowerCase() === 'sell';
    const c = sell ? 'var(--red)' : 'var(--green)';
    const cRgb = sell ? '240,85,109' : '63,185,80';
    return (
      '<div class="ai-card" style="border:1px solid rgba(' + cRgb + ',.3);border-left:3px solid ' + c + ';border-radius:7px;' +
           'padding:10px 12px;background:linear-gradient(135deg,rgba(' + cRgb + ',.08),rgba(255,255,255,.02) 55%);">' +
        '<div style="display:flex;justify-content:space-between;gap:8px;align-items:center;">' +
          '<strong style="font-size:12px;">' + _esc(p.contract) + '</strong>' +
          '<span class="ai-chip" style="background:rgba(' + cRgb + ',.18);color:' + c + ';border:1px solid rgba(' + cRgb + ',.4);">' + (sell ? 'SELL' : 'BUY') + '</span>' +
        '</div>' +
        (p.approx_price ? '<div style="font-size:11px;color:var(--muted);margin-top:3px;">≈ ' + _esc(p.approx_price) + '</div>' : '') +
        (p.size_hint ? '<div style="font-size:10px;color:var(--muted);margin-top:2px;">Size: ' + _esc(p.size_hint) + '</div>' : '') +
        (p.why ? '<div style="font-size:11px;margin-top:5px;line-height:1.5;">' + _esc(p.why) + '</div>' : '') +
      '</div>'
    );
  }

  function _playCard(p) {
    const dirs = {
      bullish: { c: 'var(--green)', rgb: '63,185,80',  icon: '▲' },
      bearish: { c: 'var(--red)',   rgb: '240,85,109', icon: '▼' },
      neutral: { c: 'var(--amber)', rgb: '210,153,34', icon: '◆' },
    };
    const dir = (p.direction || 'neutral').toLowerCase();
    const d = dirs[dir] || dirs.neutral;
    const rows = [
      ['Instrument', p.instrument], ['Details', p.details], ['Entry', p.entry],
      ['Target', p.target], ['Stop', p.stop],
    ].filter((r) => r[1]);
    return (
      '<div class="ai-card" style="border:1px solid rgba(' + d.rgb + ',.25);border-top:3px solid ' + d.c + ';border-radius:8px;' +
           'padding:12px 14px;background:linear-gradient(180deg,rgba(' + d.rgb + ',.07),rgba(255,255,255,.02) 40%);">' +
        '<div style="display:flex;justify-content:space-between;gap:8px;align-items:center;">' +
          '<strong style="font-size:12.5px;">' + _esc(p.name || 'Play') + '</strong>' +
          '<span class="ai-chip" style="background:rgba(' + d.rgb + ',.16);color:' + d.c + ';border:1px solid rgba(' + d.rgb + ',.4);">' +
            d.icon + ' ' + _esc(dir) + '</span>' +
        '</div>' +
        rows.map((r) =>
          '<div style="font-size:11px;margin-top:4px;line-height:1.5;"><span style="color:var(--muted);">' + r[0] + ':</span> ' + _esc(r[1]) + '</div>'
        ).join('') +
        (p.est_pop_pct != null ? _popMeter(p.est_pop_pct) : '') +
        (p.rationale ? '<div style="font-size:11px;color:var(--muted);margin-top:7px;line-height:1.5;font-style:italic;' +
             'border-top:1px dashed rgba(255,255,255,.08);padding-top:6px;">' + _esc(p.rationale) + '</div>' : '') +
      '</div>'
    );
  }

  function _renderAnalysis(a, meta) {
    meta = meta || {};
    _injectStyles();
    _show('ai-empty-state', false);

    // Restart entrance animations on re-render.
    const resEl = $('ai-results');
    if (resEl) {
      resEl.style.display = 'none';
      void resEl.offsetHeight; // reflow
      resEl.style.display = 'block';
    }

    const warn = $('ai-truncation-warning');
    if (warn) {
      if (meta.parseFailed) {
        warn.innerHTML = '⚠ Could not parse a JSON block from the paste - showing the raw text below. ' +
          'Make sure you copy Claude’s <strong>entire</strong> reply including the JSON code block.';
        warn.style.display = 'block';
      } else if (meta.truncated) {
        warn.innerHTML = '⚠ The pasted reply looked cut off - the JSON was repaired automatically, so some trailing fields may be missing. ' +
          'For complete results, copy the full response.';
        warn.style.display = 'block';
      } else {
        warn.style.display = 'none';
      }
    }

    const raw = $('ai-raw');
    if (raw) {
      if (!a && meta.raw) { raw.textContent = meta.raw; raw.style.display = 'block'; }
      else { raw.style.display = 'none'; raw.textContent = ''; }
    }

    if (!a) {
      // Parse failed: clear any previously rendered analysis so stale
      // results from an earlier run/ticker aren't shown as if current.
      ['ai-prob-bars', 'ai-prob-rationale', 'ai-summary', 'ai-crowd', 'ai-levels',
       'ai-picks', 'ai-plays', 'ai-position', 'ai-risks', 'ai-sources',
       'ai-generated-at', 'ai-disclaimer'].forEach((id) => {
        const el = $(id);
        if (el) el.innerHTML = '';
      });
      const confEl = $('ai-confidence');
      if (confEl) confEl.textContent = '-';
      _show('ai-picks-wrap', false);
      return;
    }

    // Probabilities
    const probs = a.probabilities || {};
    const bars = $('ai-prob-bars');
    if (bars) {
      bars.innerHTML =
        _probBar('Bullish', 'bullish', probs.bullish_pct) +
        _probBar('Sideways', 'sideways', probs.sideways_pct) +
        _probBar('Bearish', 'bearish', probs.bearish_pct);
    }
    const conf = $('ai-confidence');
    if (conf) conf.innerHTML = (a.confidence_pct != null) ? _confidenceRing(a.confidence_pct) : '-';
    const rat = $('ai-prob-rationale');
    if (rat) rat.textContent = probs.rationale || '';

    // Summary
    const sum = $('ai-summary');
    if (sum) sum.textContent = a.market_summary || '';

    // Crowd
    const crowd = $('ai-crowd');
    if (crowd) {
      crowd.innerHTML = (a.what_crowd_is_watching || []).map((b) =>
        '<li style="margin:3px 0;">' + _esc(b) + '</li>').join('');
    }

    // Key levels: visual ladder + textual list
    const lv = $('ai-levels');
    if (lv) {
      const k = a.key_levels || {};
      const fmtNums = (arr) => (arr || []).map((n) => '<strong>' + _esc(n) + '</strong>').join(', ') || '-';
      lv.innerHTML =
        _levelsLadder(k) +
        '<div style="margin-top:6px;font-size:11.5px;">' +
          '<div><span style="color:var(--red);">▔ Resistance:</span> ' + fmtNums(k.resistance) + '</div>' +
          '<div><span style="color:var(--green);">▁ Support:</span> ' + fmtNums(k.support) + '</div>' +
          (k.gamma_magnet != null ? '<div><span style="color:var(--purple);">◆ Gamma magnet:</span> <strong>' + _esc(k.gamma_magnet) + '</strong></div>' : '') +
        '</div>';
    }

    // Contract picks
    const picks = a.top_contract_picks || [];
    _show('ai-picks-wrap', picks.length > 0);
    const pickEl = $('ai-picks');
    if (pickEl) pickEl.innerHTML = picks.map(_pickCard).join('');

    // Plays
    const plays = $('ai-plays');
    if (plays) plays.innerHTML = (a.suggested_plays || []).map(_playCard).join('');

    // Position + risks
    const pos = $('ai-position');
    if (pos) pos.textContent = a.next_week_position || '';
    const risks = $('ai-risks');
    if (risks) {
      risks.innerHTML = (a.risks || []).map((b) =>
        '<li style="margin:3px 0;">' + _esc(b) + '</li>').join('');
    }

    // Footer
    const srcEl = $('ai-sources');
    if (srcEl) {
      const used = (meta.sourcesUsed || []).length;
      const failed = (meta.sourcesFailed || []);
      let txt = meta.model ? ('Model: ' + meta.model) : '';
      if (used) txt += (txt ? ' · ' : '') + used + ' data sources';
      if (failed.length) txt += ' (' + failed.length + ' failed)';
      srcEl.textContent = txt;
    }
    const gen = $('ai-generated-at');
    if (gen) gen.textContent = meta.generatedAt ? ('Generated ' + new Date(meta.generatedAt).toLocaleString()) : '';
    const disc = $('ai-disclaimer');
    if (disc) disc.textContent = a.disclaimer || 'Model output for research only - not financial advice.';

    if (resEl && resEl.scrollIntoView) resEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
})();
