/**
 * static/js/tabs/sentiment.js — Live News Feed + Sector Cramer fallback
 *
 * Phase 5 extraction.  Owns:
 *   • /ws/news WebSocket connection with exponential-backoff reconnect
 *   • 50-item ring buffer with client-side dedup (md5-lite via djb2 hash)
 *   • Pause / Resume: items still arrive while paused, stored in _pending[]
 *   • Filter pills: All | Company | Macro | Sector | Positive | Negative
 *   • Live "X seconds ago" timestamps (setInterval 1 s)
 *   • Sector Cramer fallback: when cramer.articles.length === 0 fires
 *     GET /api/sentiment/sector-cramer and renders "Sector Cramer Signal"
 *
 * Public surface (window.*):
 *   window.initSentimentFeed(ticker)  — call when Sentiment tab opens
 *   window.switchSentimentTicker(ticker) — call on ticker change
 *   window.renderSectorCramer(data)   — render sector-cramer card
 *
 * The module monkey-patches itself onto window after DOMContentLoaded, same
 * pattern as market-structure.js and options.js.
 */

(function () {
  'use strict';

  // ── Constants ────────────────────────────────────────────────────────────────
  const BUFFER_MAX   = 50;
  const BACKOFF_INIT = 1000;   // ms
  const BACKOFF_MAX  = 30000;  // ms
  const TIMESTAMP_INTERVAL = 1000; // ms

  // ── Module state ─────────────────────────────────────────────────────────────
  let _ws            = null;
  let _currentTicker = '';
  let _buffer        = [];      // all received items, newest-first, max 50
  let _pending       = [];      // items received while paused
  let _seen          = new Set();
  let _paused        = false;
  let _activeFilter  = 'all';
  let _backoffDelay  = BACKOFF_INIT;
  let _reconnectTimer = null;
  let _tsInterval    = null;
  let _wsOpen        = false;   // true when /ws/news is connected

  // ── djb2 hash — fast enough dedup key (mirrors md5(title[:60]) on server) ──
  function _hashTitle(title) {
    const s = title.slice(0, 60).toLowerCase();
    let h = 5381;
    for (let i = 0; i < s.length; i++) {
      h = (h * 33) ^ s.charCodeAt(i);
    }
    return (h >>> 0).toString(16);
  }

  // ── DOM helpers ──────────────────────────────────────────────────────────────
  function _el(id) { return document.getElementById(id); }

  function _setStatus(state) {
    const dot   = _el('live-news-dot');
    const label = _el('live-news-status');
    if (!dot || !label) return;
    const map = {
      connected:     { color: 'var(--green)',  text: 'Connected'     },
      disconnected:  { color: 'var(--red)',    text: 'Disconnected'  },
      reconnecting:  { color: 'var(--amber)',  text: 'Reconnecting…' },
    };
    const s = map[state] || map.disconnected;
    dot.style.background   = s.color;
    dot.style.boxShadow    = state === 'connected'
      ? `0 0 6px ${s.color}` : 'none';
    label.textContent      = s.text;
    label.style.color      = s.color;
  }

  function _setPauseBtn(paused) {
    const btn = _el('live-news-pause-btn');
    if (btn) btn.textContent = paused ? '▶ Resume' : '⏸ Pause';
  }

  // ── Relative timestamps ───────────────────────────────────────────────────────
  function _ago(isoString) {
    if (!isoString) return '';
    const dt  = new Date(isoString);
    if (isNaN(dt)) return '';
    const sec = Math.floor((Date.now() - dt) / 1000);
    if (sec < 5)   return 'just now';
    if (sec < 60)  return `${sec}s ago`;
    const min = Math.floor(sec / 60);
    if (min < 60)  return `${min}m ago`;
    const hr = Math.floor(min / 60);
    if (hr < 24)   return `${hr}h ago`;
    return `${Math.floor(hr / 24)}d ago`;
  }

  function _startTimestampUpdater() {
    if (_tsInterval) clearInterval(_tsInterval);
    _tsInterval = setInterval(() => {
      const items = document.querySelectorAll('#live-news-list .lnf-ts');
      items.forEach(el => {
        const iso = el.dataset.iso;
        if (iso) el.textContent = _ago(iso);
      });
    }, TIMESTAMP_INTERVAL);
  }

  // ── Single news item HTML ─────────────────────────────────────────────────────
  function _sentimentColor(s) {
    if (s === 'Positive') return 'var(--green)';
    if (s === 'Negative') return 'var(--red)';
    return 'var(--muted)';
  }

  function _categoryBadge(c) {
    const colors = {
      Company: 'rgba(88,166,255,.15)',
      Macro:   'rgba(210,153,34,.15)',
      Sector:  'rgba(63,185,80,.12)',
      General: 'rgba(139,148,158,.12)',
    };
    const bg = colors[c] || colors.General;
    return `<span style="font-size:9px;padding:1px 5px;border-radius:4px;
            background:${bg};color:var(--muted);margin-left:4px;">${c}</span>`;
  }

  function _buildItemHTML(item) {
    const ago = _ago(item.published_iso);
    const sentColor = _sentimentColor(item.sentiment);
    return `
      <div class="lnf-item" data-category="${item.category}"
           data-sentiment="${item.sentiment}"
           style="padding:9px 10px;border-radius:7px;background:var(--surface2);
                  border:1px solid var(--border);display:flex;flex-direction:column;gap:3px;">
        <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
          <span style="font-size:10px;color:var(--muted);">${item.source}</span>
          ${_categoryBadge(item.category)}
          <span style="margin-left:auto;font-size:10px;color:${sentColor};">${item.sentiment}</span>
        </div>
        <a href="${item.url}" target="_blank" rel="noopener"
           style="font-size:12px;font-weight:600;color:var(--text);text-decoration:none;
                  line-height:1.4;">
          ${item.title}
        </a>
        <div style="font-size:10px;color:var(--muted);">
          <span class="lnf-ts" data-iso="${item.published_iso}">${ago}</span>
        </div>
      </div>`;
  }

  // ── Render the visible list ───────────────────────────────────────────────────
  function _renderList() {
    const list = _el('live-news-list');
    if (!list) return;

    let items = _buffer;

    // Apply category / sentiment filter
    if (_activeFilter === 'company')  items = items.filter(i => i.category  === 'Company');
    else if (_activeFilter === 'macro')   items = items.filter(i => i.category  === 'Macro');
    else if (_activeFilter === 'sector')  items = items.filter(i => i.category  === 'Sector');
    else if (_activeFilter === 'positive') items = items.filter(i => i.sentiment === 'Positive');
    else if (_activeFilter === 'negative') items = items.filter(i => i.sentiment === 'Negative');

    if (items.length === 0) {
      list.innerHTML = `<div style="text-align:center;color:var(--muted);
                          font-size:12px;padding:24px;">
                          No headlines yet — waiting for first poll…</div>`;
      return;
    }

    list.innerHTML = items.map(_buildItemHTML).join('');
    _startTimestampUpdater();
  }

  // ── Buffer management ─────────────────────────────────────────────────────────
  function _ingestItems(items) {
    // Items arrive newest-first from server; prepend to buffer maintaining that order
    for (const item of items) {
      const key = _hashTitle(item.title || '');
      if (_seen.has(key)) continue;
      _seen.add(key);
      _buffer.unshift(item);
    }
    // Trim
    if (_buffer.length > BUFFER_MAX) {
      _buffer = _buffer.slice(0, BUFFER_MAX);
    }
  }

  // ── WS message handler ────────────────────────────────────────────────────────
  function _onMessage(evt) {
    let msg;
    try { msg = JSON.parse(evt.data); } catch { return; }

    const type  = msg.type;
    const items = msg.items || [];

    if (type === 'init') {
      // Full buffer for this ticker — reset state
      _buffer = [];
      _seen   = new Set();
      _pending = [];
      _ingestItems(items);
      _renderList();
      return;
    }

    if (type === 'update') {
      if (_paused) {
        // Stash for when user resumes
        _pending.push(...items);
        const btn = _el('live-news-pause-btn');
        if (btn) btn.textContent = `▶ Resume (${_pending.length})`;
        return;
      }
      _ingestItems(items);
      _renderList();
    }
  }

  // ── WebSocket lifecycle ───────────────────────────────────────────────────────
  function _connect() {
    if (_ws && (_ws.readyState === WebSocket.OPEN || _ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    _ws = new WebSocket(`${proto}//${location.host}/ws/news`);
    _setStatus('reconnecting');

    _ws.onopen = () => {
      _wsOpen      = true;
      _backoffDelay = BACKOFF_INIT;
      _setStatus('connected');
      if (_currentTicker) {
        _ws.send(JSON.stringify({ ticker: _currentTicker }));
      }
    };

    _ws.onmessage = _onMessage;

    _ws.onclose = () => {
      _wsOpen = false;
      _setStatus('disconnected');
      _scheduleReconnect();
    };

    _ws.onerror = (err) => {
      console.warn('[sentiment.js] WS error', err);
      // onclose will follow — reconnect happens there
    };
  }

  function _scheduleReconnect() {
    if (_reconnectTimer) return;
    _reconnectTimer = setTimeout(() => {
      _reconnectTimer = null;
      _setStatus('reconnecting');
      _connect();
    }, _backoffDelay);
    _backoffDelay = Math.min(_backoffDelay * 2, BACKOFF_MAX);
  }

  function _disconnect() {
    if (_reconnectTimer) {
      clearTimeout(_reconnectTimer);
      _reconnectTimer = null;
    }
    if (_tsInterval) {
      clearInterval(_tsInterval);
      _tsInterval = null;
    }
    if (_ws) {
      _ws.onclose = null;   // suppress reconnect
      _ws.close();
      _ws = null;
    }
    _wsOpen = false;
    _setStatus('disconnected');
  }

  // ── Filter pills ──────────────────────────────────────────────────────────────
  function _setFilter(filter) {
    _activeFilter = filter;
    document.querySelectorAll('.lnf-filter-btn').forEach(btn => {
      const active = btn.dataset.filter === filter;
      btn.style.background   = active ? 'rgba(88,166,255,.25)' : 'transparent';
      btn.style.borderColor  = active ? 'rgba(88,166,255,.5)'  : 'rgba(139,148,158,.3)';
      btn.style.color        = active ? '#58a6ff'              : 'var(--muted)';
    });
    _renderList();
  }

  // ── Pause / Resume ────────────────────────────────────────────────────────────
  function _togglePause() {
    _paused = !_paused;
    _setPauseBtn(_paused);
    if (!_paused && _pending.length) {
      _ingestItems(_pending);
      _pending = [];
      _renderList();
    }
  }

  // ── Sector Cramer fallback ────────────────────────────────────────────────────
  function _renderSectorCramerCard(data) {
    const inv   = data.inverse_signal  || 'WAIT';
    const sig   = data.cramer_signal   || 'unknown';
    const conf  = data.confidence      || 'low';
    const label = data.source_label    || 'Sector Cramer Signal';
    const arts  = data.articles        || [];

    // Update the existing Cramer elements if they're present
    const invEl   = _el('sp-cramer-inverse-signal');
    const rawEl   = _el('sp-cramer-raw');
    const confEl  = _el('sp-cramer-confidence');
    const hlEl    = _el('sp-cramer-headlines');
    const tickerEl = _el('sp-cramer-ticker-stance');

    const invColor = inv === 'BUY' ? 'var(--green)' : inv === 'SELL' ? 'var(--red)' : 'var(--muted)';

    if (invEl)    { invEl.textContent = inv; invEl.style.color = invColor; }
    if (rawEl)    { rawEl.textContent = sig; rawEl.style.color = invColor; }
    if (confEl)   { confEl.textContent = `${conf} confidence · ${label}`; }
    if (tickerEl) { tickerEl.textContent = '— sector'; tickerEl.style.color = 'var(--muted)'; }

    if (hlEl) {
      hlEl.innerHTML = arts.slice(0, 3).map(a => `
        <div style="font-size:11px;padding:4px 0;border-bottom:1px solid var(--border);">
          <a href="${a.url || '#'}" target="_blank" rel="noopener"
             style="color:var(--text);text-decoration:none;">${a.title || ''}</a>
          <span style="color:var(--muted);margin-left:4px;">${a.date || ''}</span>
          ${a.peer_ticker ? `<span style="color:var(--muted);margin-left:4px;">[${a.peer_ticker}]</span>` : ''}
        </div>`).join('');
    }
  }

  async function _maybeFetchSectorCramer(ticker, articles) {
    if (articles && articles.length > 0) return;  // primary coverage exists
    try {
      const resp = await fetch(`/api/sentiment/sector-cramer?ticker=${encodeURIComponent(ticker)}`);
      if (!resp.ok) return;
      const data = await resp.json();
      if (data.available) {
        _renderSectorCramerCard(data);
        // Add a small amber note below the Cramer card header
        const hdr = _el('sp-cramer-confidence');
        if (hdr && !hdr.dataset.sectorNote) {
          hdr.dataset.sectorNote = '1';
          const note = document.createElement('div');
          note.style.cssText = 'font-size:10px;color:var(--amber);margin-top:4px;';
          note.textContent = `⚠ No recent Cramer coverage for ${ticker} — showing ${data.sector} sector peers`;
          hdr.parentNode && hdr.parentNode.insertBefore(note, hdr.nextSibling);
        }
      }
    } catch (err) {
      console.warn('[sentiment.js] sector-cramer fetch failed', err);
    }
  }

  // ── Public API ────────────────────────────────────────────────────────────────

  /**
   * Called when the Sentiment tab opens (or on first load).
   * Initialises the WS connection for the given ticker.
   */
  function initSentimentFeed(ticker) {
    _currentTicker = (ticker || '').toUpperCase();
    // Reset buffer when initialising for a new page load
    _buffer  = [];
    _seen    = new Set();
    _pending = [];
    _paused  = false;
    _setPauseBtn(false);
    _setFilter('all');
    _connect();
  }

  /**
   * Called whenever the user switches the active ticker.
   * Sends the new subscription message; server replies with "init".
   */
  function switchSentimentTicker(ticker) {
    const t = (ticker || '').toUpperCase();
    if (t === _currentTicker) return;
    _currentTicker = t;
    _buffer  = [];
    _seen    = new Set();
    _pending = [];
    _paused  = false;
    _setPauseBtn(false);
    _renderList();
    if (_ws && _ws.readyState === WebSocket.OPEN) {
      _ws.send(JSON.stringify({ ticker: t }));
    } else {
      _connect();
    }
  }

  /**
   * Triggered from _renderSentimentPanel (inline JS) when sentiment data arrives.
   * Checks cramer.articles and fires the sector fallback if needed.
   */
  function renderSectorCramer(sentimentData) {
    try {
      const cramer   = (sentimentData || {}).cramer || {};
      const articles = cramer.articles || [];
      const ticker   = (sentimentData || {}).ticker || _currentTicker;
      _maybeFetchSectorCramer(ticker, articles);
    } catch (e) {
      console.warn('[sentiment.js] renderSectorCramer error', e);
    }
  }

  // ── Wire pause button + filter pills after DOM ready ─────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    const pauseBtn = _el('live-news-pause-btn');
    if (pauseBtn) {
      pauseBtn.addEventListener('click', _togglePause);
    }

    document.querySelectorAll('.lnf-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => _setFilter(btn.dataset.filter));
    });

    // Close WS on page unload to avoid zombie connections
    window.addEventListener('beforeunload', _disconnect);
    window.addEventListener('pagehide',     _disconnect);
  });

  // ── Expose on window ──────────────────────────────────────────────────────────
  window.initSentimentFeed    = initSentimentFeed;
  window.switchSentimentTicker = switchSentimentTicker;
  window.renderSectorCramer   = renderSectorCramer;

  // Expose for debugging
  window.__sentimentFeed = {
    getBuffer:  () => _buffer,
    getPending: () => _pending,
    isPaused:   () => _paused,
    wsState:    () => _ws ? _ws.readyState : -1,
  };

}());
