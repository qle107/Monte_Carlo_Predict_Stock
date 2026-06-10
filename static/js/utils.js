// Pure helpers shared across dashboard modules.

(function () {
  'use strict';

  function log(msg, color) {
    const el = document.getElementById('log');
    if (!el) return;
    const t = new Date().toLocaleTimeString();
    const c = color || 'var(--blue)';
    const line = document.createElement('div');
    line.style.cssText = 'border-bottom:1px solid rgba(255,255,255,.04);padding:3px 0;word-break:break-word;';
    line.innerHTML = `<span style="color:${c};font-weight:600;">[${t}]</span> ${msg}`;
    el.insertBefore(line, el.firstChild);
    while (el.children.length > 60) el.removeChild(el.lastChild);
  }

  function fmt(n, d = 2) { return typeof n === 'number' && isFinite(n) ? n.toFixed(d) : String(n); }
  function fmtPct(n, d = 2) { return (n >= 0 ? '+' : '') + fmt(n, d) + '%'; }
  function colorOf(n) { return n > 0 ? 'var(--green)' : n < 0 ? 'var(--red)' : 'var(--muted)'; }

  function _fmtVol(n) {
    if (n == null || isNaN(n)) return '-';
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
    if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
    return n.toLocaleString();
  }

  function _fmtDollar(n) {
    if (n == null || isNaN(n)) return '-';
    if (n >= 1_000_000_000) return '$' + (n / 1_000_000_000).toFixed(2) + 'B';
    if (n >= 1_000_000) return '$' + (n / 1_000_000).toFixed(2) + 'M';
    if (n >= 1_000) return '$' + (n / 1_000).toFixed(1) + 'K';
    return '$' + n.toFixed(0);
  }

  function _fmtCompact(n) {
    if (!n && n !== 0) return '-';
    const abs = Math.abs(n);
    const sign = n < 0 ? '-' : '';
    if (abs >= 1e9) return sign + (abs / 1e9).toFixed(1) + 'B';
    if (abs >= 1e6) return sign + (abs / 1e6).toFixed(1) + 'M';
    if (abs >= 1e3) return sign + (abs / 1e3).toFixed(1) + 'K';
    return sign + abs.toFixed(0);
  }

  function _scoreColor(s) {
    if (s > 0.1) return 'var(--green)';
    if (s < -0.1) return 'var(--red)';
    return 'var(--muted)';
  }
  function _scoreStr(s) { return (s >= 0 ? '+' : '') + s.toFixed(3); }

  function _toLwcSec(iso) { return Math.floor(new Date(iso).getTime() / 1000); }

  const $ = id => document.getElementById(id);

  window.log = log;
  window.fmt = fmt;
  window.fmtPct = fmtPct;
  window.colorOf = colorOf;
  window._fmtVol = _fmtVol;
  window._fmtDollar = _fmtDollar;
  window._fmtCompact = _fmtCompact;
  window._scoreColor = _scoreColor;
  window._scoreStr = _scoreStr;
  window._toLwcSec = _toLwcSec;
  window.$ = $;
})();
