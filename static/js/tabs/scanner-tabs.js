// Tab switcher and options-panel fetch orchestration.

function switchScannerTab(tab) {
  const panels = {
    breakout:           document.getElementById('breakout-scan-panel'),
    zone:               document.getElementById('zone-scan-panel'),
    sentiment:          document.getElementById('sentiment-scan-panel'),
    options:            document.getElementById('options-panel'),
    'market-structure': document.getElementById('market-structure-panel'),
    news:               document.getElementById('news-panel'),
    'unusual-opts':     document.getElementById('unusual-opts-panel'),
    pop:                document.getElementById('pop-scan-panel'),
    ai:                 document.getElementById('ai-analyst-panel'),
  };
  const tabs = {
    breakout:           document.getElementById('tab-breakout'),
    zone:               document.getElementById('tab-zone'),
    sentiment:          document.getElementById('tab-sentiment'),
    options:            document.getElementById('tab-options'),
    'market-structure': document.getElementById('tab-market-structure'),
    news:               document.getElementById('tab-news'),
    'unusual-opts':     document.getElementById('tab-unusual-opts'),
    pop:                document.getElementById('tab-pop'),
    ai:                 document.getElementById('tab-ai'),
  };
  const styles = {
    breakout:           { color: 'var(--blue)',   bg: 'rgba(88,166,255,.06)',  border: 'var(--blue)' },
    zone:               { color: 'var(--purple)', bg: 'rgba(188,140,255,.06)', border: 'var(--purple)' },
    sentiment:          { color: 'var(--amber)',  bg: 'rgba(210,153,34,.06)',  border: 'var(--amber)' },
    options:            { color: 'var(--green)',  bg: 'rgba(63,185,80,.06)',   border: 'var(--green)' },
    'market-structure': { color: 'var(--purple)', bg: 'rgba(188,140,255,.06)', border: 'var(--purple)' },
    news:               { color: 'var(--amber)',  bg: 'rgba(210,153,34,.06)',  border: 'var(--amber)' },
    'unusual-opts':     { color: '#f85149',       bg: 'rgba(248,81,73,.06)',   border: '#f85149' },
    pop:                { color: 'var(--blue)',   bg: 'rgba(88,166,255,.06)',  border: 'var(--blue)' },
    ai:                 { color: 'var(--purple)', bg: 'rgba(188,140,255,.06)', border: 'var(--purple)' },
  };

  Object.keys(panels).forEach(key => {
    if (!panels[key]) return;
    panels[key].style.display = (key === tab) ? 'block' : 'none';
    const t = tabs[key];
    if (!t) return;
    if (key === tab) {
      t.classList.add('active');
      t.style.color            = styles[key].color;
      t.style.background       = styles[key].bg;
      t.style.borderBottomColor = styles[key].border;
    } else {
      t.classList.remove('active');
      t.style.color            = '';
      t.style.background       = '';
      t.style.borderBottomColor = '';
    }
  });

  // When the sentiment tab opens, auto-run if the global ticker changed since last fetch.
  if (tab === 'sentiment') {
    const currentVal = (currentConfig.ticker || '').toUpperCase().trim();
    if (currentVal && currentVal !== _lastSentTicker && !_sentLoading) {
      runSentiment(true);
    }
    if (typeof window.switchSentimentTicker === 'function' && currentVal) {
      try { window.switchSentimentTicker(currentVal); } catch (e) { console.warn(e); }
    }
  }
  // When the options tab opens, auto-run if ticker changed or no data yet
  if (tab === 'options') {
    const currentVal = (currentConfig.ticker || '').toUpperCase().trim();
    const optsEmpty  = document.getElementById('opts-empty-state');
    const needsRun   = currentVal && (currentVal !== _lastSentTicker || (optsEmpty && optsEmpty.style.display !== 'none'));
    if (needsRun && !_sentLoading && !_msAnalysisInProgress) {
      runOptionsPanel();
    }
  }
    if (tab === 'news') {
    if (!_macroLoaded) fetchMacroData(false);
    if (!_newsLoaded || (currentConfig.ticker || '').toUpperCase() !== _newsLastTicker) {
      fetchNewsData(false);
    }
  }
  // When the POP scanner opens, sync the global ticker and load expiries.
  if (tab === 'pop') {
    const currentVal = (currentConfig.ticker || '').toUpperCase().trim();
    if (typeof window.popScannerOnOpen === 'function') {
      try { window.popScannerOnOpen(currentVal); } catch (e) { console.warn(e); }
    }
  }
  // When the AI Analyst opens, sync the ticker badge and check configuration.
  if (tab === 'ai' && typeof window.aiAnalystOnOpen === 'function') {
    try { window.aiAnalystOnOpen((currentConfig.ticker || '').toUpperCase().trim()); } catch (e) { console.warn(e); }
  }
}

let _optsLoading = false;
let _optsAbortController = null;

function _setOptsFetchingUI(active) {
  const btn   = document.getElementById('opts-run-btn');
  const stop  = document.getElementById('opts-stop-btn');
  const spin  = document.getElementById('opts-spin');
  const lbl   = document.getElementById('opts-run-label');
  if (btn)  btn.disabled = active;
  if (stop) stop.style.display = active ? 'inline-flex' : 'none';
  if (spin) spin.style.display = active ? 'inline-block' : 'none';
  if (lbl)  lbl.textContent = active ? 'Fetching...' : 'Fetch Options';
}

function stopOptionsPanel() {
  if (_optsAbortController) {
    _optsAbortController.abort();
    _optsAbortController = null;
  }
  _optsLoading = false;
  _setOptsFetchingUI(false);
  const errEl = document.getElementById('opts-error-state');
  if (errEl) {
    errEl.style.display = 'block';
    errEl.textContent = 'Fetch stopped.';
  }
}

async function runOptionsPanel() {
  if (_optsLoading) return;
  _optsLoading = true;
  _optsAbortController = new AbortController();
  const signal = _optsAbortController.signal;
  const empty = document.getElementById('opts-empty-state');
  const errEl = document.getElementById('opts-error-state');
  _setOptsFetchingUI(true);
  if (empty) empty.style.display = 'none';
  if (errEl) errEl.style.display = 'none';
  // Sync ticker badge
  const tb = document.getElementById('opts-ticker-badge');
  if (tb) tb.textContent = (currentConfig.ticker || '-').toUpperCase();
  try {
    // Run both in parallel - sentiment populates sp-*, market-structure populates ms-options-*
    await Promise.all([
      runSentiment(true, { signal, silent: true, force: true }),
      runMarketStructure({ signal, silent: true, force: true }),
    ]);
  } catch(e) {
    if (e.name === 'AbortError') {
      if (errEl) { errEl.style.display = 'block'; errEl.textContent = 'Fetch stopped.'; }
    } else if (errEl) {
      errEl.style.display = 'block';
      errEl.textContent = 'Error: ' + e.message;
    }
  } finally {
    _optsLoading = false;
    _optsAbortController = null;
    _setOptsFetchingUI(false);
  }
}

if (window.__mc_trader_modular__) {
  window.__mc_trader_modular__.extracted.push('tabs/scanner-tabs.js');
}
