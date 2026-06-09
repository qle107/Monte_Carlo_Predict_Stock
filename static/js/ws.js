// WebSocket connect/reconnect and message dispatcher.

(function () {
  'use strict';

  let _skeletonDismissed = false;
  function _dismissSkeleton() {
    if (_skeletonDismissed) return;
    _skeletonDismissed = true;
    const el = document.getElementById('skeleton-overlay');
    if (!el) return;
    el.classList.add('fade-out');
    setTimeout(() => { if (el.parentNode) el.parentNode.removeChild(el); }, 380);
  }

  function updateUI(d) {
    if (d && d.type === 'partial') {
      // Render chart immediately and show "Analyzing..." on signal elements.
      try { _handlePartialUpdate(d); } catch(e) { console.warn('partial UI:', e); }
      return;
    }
    try { _updateUI(d); } catch(e) { log('UI error: '+e.message, 'var(--red)'); console.error(e); }
    try { _updateMarketStructureFromBroadcast(d); } catch(_) {}
    try { _updateMicrostructureCard(d); } catch(e) { console.warn('ms-card:', e); }
    try { _maybeAutoRunBacktest(d.ticker, d.interval); } catch(_) {}
    try { if (typeof initTooltips === 'function') initTooltips(); } catch(_) {}
    try { _updateSignalFromMC(d); } catch(_) {}
    try { if (window.mcAlerts) mcAlerts.check(d.ticker, d.current_price); } catch(_) {}
  }

  // Called when server broadcasts type="partial" after fetch but before analysis.
  function _handlePartialUpdate(d) {
    const el = id => document.getElementById(id);

    // Render chart with available candle data (MC paths will be absent - fine,
    // _setLwcData guards against missing mc fields)
    if (d.candles && d.candles.length) {
      buildCharts(d);
    }

    // Update price and ticker header immediately
    if (d.current_price) {
      const p = el('m-price');
      if (p) p.textContent = '$' + d.current_price.toFixed(2);
    }
    if (d.ticker && d.interval) {
      const h = el('header-title');
      if (h) h.textContent = `MC Trader · ${d.ticker}`;
    }

    // Signal badge -> "Analyzing..."
    const badge = el('signal-badge');
    if (badge) { badge.textContent = 'Analyzing...'; badge.className = 'badge neutral'; }

    // Metric placeholders -> subtle pulse while analysis runs
    ['m-drift','m-target','m-conf','m-cvar'].forEach(id => {
      const e = el(id);
      if (e && e.textContent === '-') {
        e.textContent = '...';
        e.style.color = 'var(--muted)';
      }
    });

    // Regime / verdict -> loading hint
    const verdict = el('regime-verdict');
    if (verdict && (verdict.textContent === 'Loading...' || verdict.textContent === '-')) {
      verdict.textContent = 'Analyzing market conditions...';
      verdict.style.color = 'var(--muted)';
    }
  }

  function connect() {
    const proto=location.protocol==='https:' ? 'wss' : 'ws';
    const ws=new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen   =()=>{ log('Connected'); document.getElementById('status-dot').className='dot live'; };
    ws.onmessage=e =>{
      try {
        const d = JSON.parse(e.data);
        updateUI(d);
        _dismissSkeleton();   // remove skeleton on first message (partial or full)
      } catch(err){ log('Parse error: '+err,'var(--red)'); }
    };
    ws.onerror  =()=>{ log('WS error','var(--red)'); _dismissSkeleton(); };
    ws.onclose  =()=>{
      document.getElementById('status-dot').className='dot';
      log('Disconnected - reconnecting in 5s...','var(--amber)');
      setTimeout(connect, 5000);
    };
  }

  // Safety net: always dismiss skeleton after 15 s even if WS never arrives
  setTimeout(_dismissSkeleton, 15000);

  // Startup: fetch current config, restore any saved settings, then open WS.
  // Deferred so inline script (currentConfig, _restoreSavedSettings) has run.
  setTimeout(() => {
    fetch('/api/config')
      .then(r => r.json())
      .then(c => { currentConfig = c; return _restoreSavedSettings(); })
      .catch(() => {})
      .finally(() => connect());
  }, 0);

  window.updateUI = updateUI;
  window._handlePartialUpdate = _handlePartialUpdate;
  window._dismissSkeleton = _dismissSkeleton;
  window.connect = connect;

  if (window.__mc_trader_modular__) {
    window.__mc_trader_modular__.extracted.push('ws.js');
  }
})();
