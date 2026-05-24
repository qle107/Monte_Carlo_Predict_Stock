/**
 * static/js/alerts.js — Price alert system (Fix 9)
 *
 * Features:
 *  • Add price alerts (above / below a target price for a given ticker)
 *  • Alerts persist in localStorage under key "mc_price_alerts"
 *  • Checks each alert against the live price broadcast from the WebSocket
 *  • Fires a Browser Notification (Notification API) when price crosses target
 *  • Falls back to an in-page banner if Notification permission is denied
 *  • Alert panel is toggled by clicking the 🔔 button in the header
 *
 * Public API (attached to window):
 *   mcAlerts.add(ticker, targetPrice, direction)  — direction: 'above'|'below'
 *   mcAlerts.remove(id)
 *   mcAlerts.check(ticker, price)  — called from updateUI() on every broadcast
 *   mcAlerts.togglePanel()
 *   mcAlerts.openPanel()
 *   mcAlerts.closePanel()
 */

(function () {
  'use strict';

  const LS_KEY = 'mc_price_alerts';

  // ── State ────────────────────────────────────────────────────────────────
  let _alerts = _load();
  let _panelOpen = false;
  let _notifPermission = Notification?.permission || 'default';

  function _load() {
    try { return JSON.parse(localStorage.getItem(LS_KEY) || '[]'); }
    catch (_) { return []; }
  }
  function _save() {
    try { localStorage.setItem(LS_KEY, JSON.stringify(_alerts)); }
    catch (_) {}
  }
  function _uid() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  }

  // ── Notification permission ───────────────────────────────────────────────
  async function _requestPermission() {
    if (!('Notification' in window)) return;
    if (Notification.permission === 'default') {
      _notifPermission = await Notification.requestPermission();
    } else {
      _notifPermission = Notification.permission;
    }
  }

  // ── Fire notification ─────────────────────────────────────────────────────
  function _notify(alert, currentPrice) {
    const msg = `${alert.ticker} hit $${currentPrice.toFixed(2)} — your ${alert.direction} $${alert.target.toFixed(2)} alert triggered`;

    // Browser notification
    if (_notifPermission === 'granted') {
      try {
        new Notification('MC Trader Alert 🔔', {
          body: msg,
          icon: '/static/img/icon.png',
          tag: alert.id,
        });
      } catch (_) {}
    }

    // In-page banner fallback (always shown)
    _showInPageBanner(msg, alert.ticker);
  }

  function _showInPageBanner(msg, ticker) {
    let banner = document.getElementById('mc-alert-banner');
    if (!banner) {
      banner = document.createElement('div');
      banner.id = 'mc-alert-banner';
      Object.assign(banner.style, {
        position: 'fixed', bottom: '16px', right: '16px', zIndex: '99999',
        background: 'rgba(88,166,255,.15)', border: '1px solid rgba(88,166,255,.5)',
        borderRadius: '10px', padding: '12px 14px', maxWidth: '320px',
        color: '#c9d1d9', fontSize: '13px', lineHeight: '1.5',
        boxShadow: '0 4px 18px rgba(0,0,0,.6)',
        display: 'flex', flexDirection: 'column', gap: '8px',
      });
      document.body.appendChild(banner);
    }
    const item = document.createElement('div');
    item.style.cssText = 'display:flex;align-items:flex-start;gap:8px;';
    item.innerHTML = `<span style="font-size:16px;flex-shrink:0;">🔔</span>
      <div style="flex:1;"><strong style="color:#58a6ff;">${ticker}</strong> — ${msg}</div>
      <button onclick="this.parentNode.remove()" style="background:none;border:none;color:#8b949e;cursor:pointer;font-size:14px;flex-shrink:0;padding:0;">✕</button>`;
    banner.appendChild(item);
    // Auto-dismiss after 8 seconds
    setTimeout(() => { try { item.remove(); } catch (_) {} }, 8000);
  }

  // ── Core: add alert ───────────────────────────────────────────────────────
  function add(ticker, targetPrice, direction) {
    if (!ticker || isNaN(targetPrice)) return null;
    const alert = {
      id:        _uid(),
      ticker:    ticker.toUpperCase().trim(),
      target:    parseFloat(targetPrice),
      direction: direction === 'above' ? 'above' : 'below',
      createdAt: new Date().toISOString(),
      triggered: false,
    };
    _alerts.push(alert);
    _save();
    _renderPanel();
    _requestPermission();
    return alert;
  }

  // ── Core: remove alert ────────────────────────────────────────────────────
  function remove(id) {
    _alerts = _alerts.filter(a => a.id !== id);
    _save();
    _renderPanel();
  }

  // ── Core: check price broadcast against all active alerts ─────────────────
  // Called from updateUI() on every full broadcast.
  function check(ticker, price) {
    if (!ticker || price == null) return;
    const sym = ticker.toUpperCase();
    _alerts.forEach(a => {
      if (a.triggered) return;
      if (a.ticker !== sym) return;
      const hit = (a.direction === 'above' && price >= a.target) ||
                  (a.direction === 'below' && price <= a.target);
      if (hit) {
        a.triggered = true;
        _save();
        _notify(a, price);
        _renderPanel();   // refresh to show triggered state
      }
    });
  }

  // ── Panel HTML ────────────────────────────────────────────────────────────
  function _renderPanel() {
    const panel = document.getElementById('mc-alerts-panel');
    if (!panel) return;

    const active    = _alerts.filter(a => !a.triggered);
    const triggered = _alerts.filter(a => a.triggered);

    const _row = (a) => `
      <div class="mc-alert-row ${a.triggered ? 'mc-alert-triggered' : ''}" data-id="${a.id}">
        <div class="mc-alert-row-main">
          <span class="mc-alert-ticker">${a.ticker}</span>
          <span class="mc-alert-dir ${a.direction === 'above' ? 'above' : 'below'}">
            ${a.direction === 'above' ? '▲' : '▼'} $${a.target.toFixed(2)}
          </span>
          ${a.triggered ? '<span class="mc-alert-fired">✓ Fired</span>' : ''}
        </div>
        <button class="mc-alert-del" onclick="mcAlerts.remove('${a.id}')" title="Remove alert">✕</button>
      </div>`;

    const notifNote = _notifPermission === 'denied'
      ? `<div class="mc-alert-notif-note">⚠ Browser notifications blocked — in-page banners only.</div>`
      : _notifPermission === 'default'
      ? `<div class="mc-alert-notif-note">Click "Add" to enable browser notifications.</div>`
      : `<div class="mc-alert-notif-note mc-alert-notif-ok">✓ Browser notifications enabled.</div>`;

    panel.innerHTML = `
      <div class="mc-alerts-hdr">
        <span>🔔 Price Alerts</span>
        <button class="mc-alerts-close" onclick="mcAlerts.closePanel()">✕</button>
      </div>
      ${notifNote}
      <!-- Add form -->
      <div class="mc-alerts-form">
        <input id="mc-alert-ticker" class="mc-alerts-input" type="text"
               placeholder="Ticker" maxlength="10"
               oninput="this.value=this.value.toUpperCase()"
               value="${(window.currentConfig && window.currentConfig.ticker) || ''}" />
        <select id="mc-alert-dir" class="mc-alerts-input">
          <option value="above">▲ Above</option>
          <option value="below">▼ Below</option>
        </select>
        <input id="mc-alert-price" class="mc-alerts-input" type="number"
               placeholder="$0.00" step="0.01" min="0" style="width:80px;" />
        <button class="mc-alerts-add-btn" onclick="mcAlerts._submitForm()">Add</button>
      </div>
      <!-- Active alerts -->
      <div class="mc-alerts-section-label">Active (${active.length})</div>
      <div class="mc-alerts-list" id="mc-alerts-active-list">
        ${active.length ? active.map(_row).join('') : '<div class="mc-alerts-empty">No active alerts</div>'}
      </div>
      ${triggered.length ? `
        <div class="mc-alerts-section-label" style="margin-top:10px;">
          Fired (${triggered.length})
          <button class="mc-alerts-clear-fired" onclick="mcAlerts.clearFired()">Clear all</button>
        </div>
        <div class="mc-alerts-list">${triggered.map(_row).join('')}</div>
      ` : ''}`;
  }

  function _submitForm() {
    const ticker = (document.getElementById('mc-alert-ticker')?.value || '').trim().toUpperCase();
    const dir    = document.getElementById('mc-alert-dir')?.value || 'above';
    const price  = parseFloat(document.getElementById('mc-alert-price')?.value || '');
    if (!ticker || isNaN(price) || price <= 0) {
      _showInPageBanner('Please enter a valid ticker and price.', '⚠');
      return;
    }
    add(ticker, price, dir);
    // Clear price input, keep ticker
    const pi = document.getElementById('mc-alert-price');
    if (pi) pi.value = '';
  }

  function clearFired() {
    _alerts = _alerts.filter(a => !a.triggered);
    _save();
    _renderPanel();
  }

  // ── Panel toggle / open / close ───────────────────────────────────────────
  function togglePanel() {
    _panelOpen ? closePanel() : openPanel();
  }

  function openPanel() {
    _panelOpen = true;
    let panel = document.getElementById('mc-alerts-panel');
    if (!panel) {
      panel = document.createElement('div');
      panel.id = 'mc-alerts-panel';
      document.body.appendChild(panel);
    }
    panel.style.display = 'flex';
    panel.style.flexDirection = 'column';
    _renderPanel();
    // Update bell badge
    _updateBell();
  }

  function closePanel() {
    _panelOpen = false;
    const panel = document.getElementById('mc-alerts-panel');
    if (panel) panel.style.display = 'none';
  }

  function _updateBell() {
    const btn = document.getElementById('mc-alerts-bell-btn');
    if (!btn) return;
    const active = _alerts.filter(a => !a.triggered).length;
    const badge = document.getElementById('mc-alerts-bell-badge');
    if (badge) {
      badge.textContent = active || '';
      badge.style.display = active ? 'flex' : 'none';
    }
  }

  // ── Expose public API ─────────────────────────────────────────────────────
  window.mcAlerts = { add, remove, check, togglePanel, openPanel, closePanel,
                      clearFired, _submitForm };

  // Init bell badge on load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _updateBell);
  } else {
    setTimeout(_updateBell, 200);
  }

}());
