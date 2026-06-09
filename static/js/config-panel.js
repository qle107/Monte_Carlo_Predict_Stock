// Settings drawer: form sync, localStorage, POST /api/config.

(function () {
  'use strict';

function toggleSettings() {
  const p=document.getElementById('settings-panel');
  const b=document.getElementById('settings-btn');
  const open=p.classList.toggle('open');
  b.classList.toggle('active',open);
  if (open) loadSettingsIntoForm();
}
function loadSettingsIntoForm() {
  if (!currentConfig.ticker) return;
  document.getElementById('s-ticker').value     =currentConfig.ticker || '';
  document.getElementById('s-interval').value   =currentConfig.interval || '15m';
  document.getElementById('s-model').value      =currentConfig.mc_model || 'microstructure';
  document.getElementById('s-lookback').value   =currentConfig.lookback || 50;
  document.getElementById('s-chart-bars').value =currentConfig.chart_bars || 200;
  document.getElementById('s-nsim').value       =currentConfig.n_sim || 10000;
  document.getElementById('s-nfwd').value       =currentConfig.n_forward || 10;
  document.getElementById('s-poll').value       =currentConfig.poll_seconds || 60;
}
/**
 * Central ticker sync - keeps every ticker input/badge in the UI in lockstep.
 * Call whenever the active ticker changes from any source.
 */
function _syncAllTickerInputs(ticker) {
  if (!ticker) return;
  const t = ticker.toUpperCase().trim();
  // Editable inputs (settings + quick-navbar)
  ['s-ticker', 'quick-ticker'].forEach(id => {
    const el = document.getElementById(id);
    if (el && el.value.toUpperCase().trim() !== t) el.value = t;
  });
  // Read-only ticker badges in feature panels
  ['sent-ticker-badge', 'ms-ticker-badge', 'opts-ticker-badge', 'ai-ticker-badge', 'pop-ticker-badge'].forEach(id => {
    const el = document.getElementById(id);
    if (el && el.textContent.trim() !== t) el.textContent = t;
  });
  // Notify tab modules if their panel is currently open (reloads expiries / status)
  const popPanel = document.getElementById('pop-scan-panel');
  if (popPanel && popPanel.style.display !== 'none' && typeof window.popScannerOnOpen === 'function') {
    try { window.popScannerOnOpen(t); } catch (e) { console.warn(e); }
  }
  const aiPanel = document.getElementById('ai-analyst-panel');
  if (aiPanel && aiPanel.style.display !== 'none' && typeof window.aiAnalystOnOpen === 'function') {
    try { window.aiAnalystOnOpen(t); } catch (e) { console.warn(e); }
  }
}

/**
 * Switch the global ticker from any click target (e.g. trending ticker list).
 * Updates the header input, pushes to backend, and auto-runs sentiment.
 */
function _switchGlobalTicker(ticker) {
  const t = (ticker || '').toUpperCase().trim();
  if (!t) return;
  const qt = document.getElementById('quick-ticker');
  if (qt) qt.value = t;
  applyQuickTicker();
}

async function applySettings() {
  const btn=document.getElementById('apply-btn');
  const msg=document.getElementById('settings-msg');
  btn.disabled=true; msg.style.color='var(--muted)'; msg.textContent='Applying...';
  const payload={};
  const v={
    ticker:      document.getElementById('s-ticker').value.trim(),
    interval:    document.getElementById('s-interval').value,
    mc_model:    document.getElementById('s-model').value,
    lookback:    parseInt(document.getElementById('s-lookback').value),
    chart_bars:  parseInt(document.getElementById('s-chart-bars').value),
    n_sim:       parseInt(document.getElementById('s-nsim').value),
    n_forward:   parseInt(document.getElementById('s-nfwd').value),
    poll_seconds:parseInt(document.getElementById('s-poll').value),
  };
  if (v.ticker)               payload.ticker      =v.ticker;
  if (v.interval)             payload.interval    =v.interval;
  if (v.mc_model)             payload.mc_model    =v.mc_model;
  if (!isNaN(v.lookback))     payload.lookback    =v.lookback;
  if (!isNaN(v.chart_bars))   payload.chart_bars  =v.chart_bars;
  if (!isNaN(v.n_sim))        payload.n_sim       =v.n_sim;
  if (!isNaN(v.n_forward))    payload.n_forward   =v.n_forward;
  if (!isNaN(v.poll_seconds)) payload.poll_seconds=v.poll_seconds;
  try {
    const res =await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await res.json();
    if (!res.ok) {
      msg.style.color='var(--red)';
      msg.textContent=data.detail?.[0]?.msg || 'Error';
    } else {
      currentConfig=data.config;
      _saveSettings();
      _syncAllTickerInputs(currentConfig.ticker);
      msg.style.color='var(--green)';
      msg.textContent='Applied ('+data.changed.join(', ')+')';
      log('Config: '+data.changed.join(', '));
      setTimeout(()=>msg.textContent='', 4000);
            // POST /api/config - don't wait for the next WS broadcast.
      if (data.result) {
        updateUI(data.result);
      } else {
        // Fallback: explicitly fetch /api/signal for the new ticker
        try {
          const sr = await fetch('/api/signal');
          const sd = await sr.json();
          if (!sd.error) updateUI(sd);
        } catch(_) {}
      }
    }
  } catch(e) {
    msg.style.color='var(--red)';
    msg.textContent='Failed: '+e;
  } finally {
    btn.disabled=false;
  }
}
async function applyQuickTicker() {
  const ticker   = (document.getElementById('quick-ticker').value || '').trim().toUpperCase();
  const interval = document.getElementById('quick-interval').value;
  if (!ticker) return;

  const payload = {};
  if (ticker)   payload.ticker   = ticker;
  if (interval) payload.interval = interval;

  const input  = document.getElementById('quick-ticker');
  const goBtn  = document.getElementById('quick-go-btn');
  input.style.color = 'var(--muted)';
  if (goBtn) goBtn.disabled = true;

  try {
    const res  = await fetch('/api/config', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
    const data = await res.json();
    if (res.ok) {
      currentConfig = data.config;
      _saveSettings();
      _syncAllTickerInputs(ticker);
      // Sync interval + model fields so settings panel stays in lockstep
      const si = document.getElementById('s-interval');
      if (si) si.value = interval;
      const sm = document.getElementById('s-model');
      if (sm) sm.value = currentConfig.mc_model || 'microstructure';
      input.style.color = 'var(--green)';
      setTimeout(() => { input.style.color = 'var(--text)'; }, 1500);
      if (data.result) updateUI(data.result);
      else {
        try { const sr = await fetch('/api/signal'); const sd = await sr.json(); if (!sd.error) updateUI(sd); } catch(_){}
      }
    } else {
      input.style.color = 'var(--red)';
      setTimeout(() => { input.style.color = 'var(--text)'; }, 2000);
    }
  } catch(e) {
    input.style.color = 'var(--red)';
    setTimeout(() => { input.style.color = 'var(--text)'; }, 2000);
  } finally {
    if (goBtn) goBtn.disabled = false;
  }
}

function resetSettings() {
  document.getElementById('s-ticker').value     = currentConfig.ticker || 'PLTR';
  document.getElementById('s-interval').value   ='15m';
  document.getElementById('s-model').value      ='microstructure';
  document.getElementById('s-lookback').value   =50;
  document.getElementById('s-chart-bars').value =200;
  document.getElementById('s-nsim').value       =10000;
  document.getElementById('s-nfwd').value       =10;
  document.getElementById('s-poll').value       =60;
  document.getElementById('settings-msg').textContent='Reset - click Apply to save';
  document.getElementById('settings-msg').style.color='var(--muted)';
}

async function reloadData() {
  const btn=document.getElementById('reload-btn');
  const spin=document.getElementById('spin');
  const lbl=document.getElementById('reload-label');
  btn.disabled=true; spin.style.display='block'; lbl.textContent='Loading...';
  log('Manual reload...');
  try {
    const res =await fetch('/api/signal');
    const data=await res.json();
    if (data.error) log('Error: '+data.error,'var(--red)');
    else { updateUI(data); log('Reload complete'); }
  } catch(e) { log('Reload failed: '+e,'var(--red)'); }
  finally { btn.disabled=false; spin.style.display='none'; lbl.textContent='Reload'; }
}

const _STORAGE_KEY = 'mc_trader_settings_v1';

function _saveSettings() {
  try {
    const s = {
      ticker:       currentConfig.ticker       || '',
      interval:     currentConfig.interval     || '15m',
      mc_model:     currentConfig.mc_model     || 'microstructure',
      lookback:     currentConfig.lookback     || 50,
      chart_bars:   currentConfig.chart_bars   || 200,
      n_sim:        currentConfig.n_sim        || 10000,
      n_forward:    currentConfig.n_forward    || 10,
      poll_seconds: currentConfig.poll_seconds || 60,
    };
    localStorage.setItem(_STORAGE_KEY, JSON.stringify(s));
  } catch(e) { /* storage unavailable - silently ignore */ }
}

function _loadSavedSettings() {
  try {
    const raw = localStorage.getItem(_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch(e) { return null; }
}

// Restore saved settings on startup: push to backend so the server reflects them.
async function _restoreSavedSettings() {
  const saved = _loadSavedSettings();
  if (!saved || !saved.ticker) return;
  const payload = {};
  if (saved.ticker)       payload.ticker       = saved.ticker;
  if (saved.interval)     payload.interval     = saved.interval;
  if (saved.mc_model)     payload.mc_model     = saved.mc_model;
  if (saved.lookback)     payload.lookback     = saved.lookback;
  if (saved.chart_bars)   payload.chart_bars   = saved.chart_bars;
  if (saved.n_sim)        payload.n_sim        = saved.n_sim;
  if (saved.n_forward)    payload.n_forward    = saved.n_forward;
  if (saved.poll_seconds) payload.poll_seconds = saved.poll_seconds;
  try {
    const res  = await fetch('/api/config', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
    const data = await res.json();
    if (res.ok) {
      currentConfig = data.config;
      // Sync ALL ticker inputs at once on restore
      _syncAllTickerInputs(saved.ticker);
      const qi = document.getElementById('quick-interval');
      if (qi) qi.value = saved.interval || '15m';
      // Also sync model in settings panel
      const sm = document.getElementById('s-model');
      if (sm) sm.value = currentConfig.mc_model || 'microstructure';
      if (data.result) updateUI(data.result);
      log('Restored settings: ' + saved.ticker + ' ' + (saved.interval || '15m'));
    }
  } catch(e) { /* non-critical - server will use its defaults */ }
}


  window.toggleSettings = toggleSettings;
  window.loadSettingsIntoForm = loadSettingsIntoForm;
  window._syncAllTickerInputs = _syncAllTickerInputs;
  window._switchGlobalTicker = _switchGlobalTicker;
  window.applySettings = applySettings;
  window.applyQuickTicker = applyQuickTicker;
  window.resetSettings = resetSettings;
  window.reloadData = reloadData;
  window._saveSettings = _saveSettings;
  window._restoreSavedSettings = _restoreSavedSettings;

  if (window.__mc_trader_modular__) {
    window.__mc_trader_modular__.extracted.push('config-panel.js');
  }
})();
