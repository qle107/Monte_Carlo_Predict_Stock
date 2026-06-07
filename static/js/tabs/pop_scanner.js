/* Probability-of-Profit scanner tab.
 * Endpoints: GET /api/options/expiries, GET /api/options/pop_scan,
 *            GET /api/options/research_scan */
(function () {
  'use strict';

  let _popTicker = '';
  let _popMode = 'my_view';          // 'my_view' | 'market'
  let _popFilter = 'best';           // 'best' | 'all' | 'top_return' | 'top_pop'
  let _popSort = { key: 'exp_return_pct', dir: -1 };
  let _popData = null;               // last scan response
  let _popLoading = false;
  let _payoffChart = null;

  const $ = (id) => document.getElementById(id);

  function _esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function _ticker() {
    // Prefer the live global config (`let currentConfig` in the dashboard
    // inline script - a global lexical binding, not window.currentConfig).
    let t = '';
    try {
      if (typeof currentConfig === 'object' && currentConfig && currentConfig.ticker) {
        t = currentConfig.ticker;
      }
    } catch (e) { /* not on the dashboard page */ }
    return ((t || _popTicker) || '').toUpperCase().trim();
  }

  function _show(id, on, disp) {
    const el = $(id);
    if (el) el.style.display = on ? (disp || 'block') : 'none';
  }

  function _setError(msg) {
    const el = $('pop-error');
    if (el) { el.textContent = msg || ''; el.style.display = msg ? 'block' : 'none'; }
  }

  /* ---------- open hook ---------- */

  window.popScannerOnOpen = function (ticker) {
    const t = (ticker || '').toUpperCase().trim();
    const changed = t && t !== _popTicker;
    if (t) _popTicker = t;
    const badge = $('pop-ticker-badge');
    if (badge) badge.textContent = _ticker() || '-';
    const sel = $('pop-expiry');
    if (changed || (sel && (!sel.options.length || sel.options[0].value === ''))) {
      _loadExpiries();
    }
  };

  async function _loadExpiries() {
    const t = _ticker();
    const sel = $('pop-expiry');
    if (!t || !sel) return;
    sel.innerHTML = '<option value="">loading…</option>';
    try {
      const res = await fetch('/api/options/expiries?ticker=' + encodeURIComponent(t));
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status));
      const exps = data.expiries || [];
      if (!exps.length) { sel.innerHTML = '<option value="">no expiries</option>'; return; }
      const today = new Date();
      sel.innerHTML = exps.map((e) => {
        const dte = Math.round((new Date(e + 'T00:00:00Z') - today) / 86400000);
        return '<option value="' + e + '">' + e + ' (' + dte + 'd)</option>';
      }).join('');
      // Default to the first expiry with >= 5 DTE.
      for (let i = 0; i < exps.length; i++) {
        const dte = Math.round((new Date(exps[i] + 'T00:00:00Z') - today) / 86400000);
        if (dte >= 5) { sel.selectedIndex = i; break; }
      }
    } catch (e) {
      sel.innerHTML = '<option value="">expiries failed</option>';
      _setError('Could not load expiries: ' + e.message);
    }
  }

  /* ---------- mode toggle / checkboxes ---------- */

  function _setMode(mode) {
    _popMode = mode === 'market' ? 'market' : 'my_view';
    document.querySelectorAll('#pop-mode-toggle .pop-mode-btn').forEach((b) => {
      const on = b.dataset.popMode === _popMode;
      b.style.background = on ? 'rgba(88,166,255,.18)' : 'transparent';
      b.style.color = on ? 'var(--blue)' : 'var(--muted)';
      b.style.fontWeight = on ? '700' : '400';
    });
    if (_popData) _renderTable();
  }

  document.addEventListener('click', (ev) => {
    const btn = ev.target.closest && ev.target.closest('#pop-mode-toggle .pop-mode-btn');
    if (btn) _setMode(btn.dataset.popMode);
    const chip = ev.target.closest && ev.target.closest('.pop-filter');
    if (chip) _setFilter(chip.dataset.popFilter);
    const th = ev.target.closest && ev.target.closest('#pop-table th[data-pop-sort]');
    if (th) _setSort(th.dataset.popSort);
  });

  window.popToggleSingles = function () {
    const singles = $('pop-singles-only');
    const bearLabel = $('pop-bear-label');
    if (bearLabel) bearLabel.style.display = (singles && singles.checked) ? 'none' : 'flex';
  };

  /* ---------- scans ---------- */

  function _params() {
    const t = _ticker();
    const expiry = ($('pop-expiry') || {}).value || '';
    const p = new URLSearchParams({ ticker: t, expiry });
    const num = (id) => { const v = parseFloat(($(id) || {}).value); return isFinite(v) ? v : null; };
    const target = num('pop-target');
    const drift = num('pop-drift');
    const vol = num('pop-vol');
    const rf = num('pop-rf');
    const comm = num('pop-comm');
    if (target != null) p.set('target', target);
    if (drift != null) p.set('drift', drift);
    if (vol != null && vol > 0) p.set('vol', vol);
    if (rf != null) p.set('risk_free', rf);
    if (comm != null) p.set('commission', comm);
    const singles = $('pop-singles-only');
    p.set('verticals', singles && singles.checked ? 'false' : 'true');
    const bear = $('pop-show-bear');
    p.set('show_bear', bear && bear.checked ? 'true' : 'false');
    return p;
  }

  function _startLoading(btnSpin, btnLabel, labelText) {
    _popLoading = true;
    _setError('');
    _show('pop-empty', false);
    _show('pop-loading', true);
    _show(btnSpin, true, 'inline-block');
    const l = $(btnLabel);
    if (l) { l.dataset.old = l.textContent; l.textContent = labelText; }
  }

  function _stopLoading(btnSpin, btnLabel) {
    _popLoading = false;
    _show('pop-loading', false);
    _show(btnSpin, false);
    const l = $(btnLabel);
    if (l && l.dataset.old) l.textContent = l.dataset.old;
  }

  window.runPopScan = async function () {
    if (_popLoading) return;
    const t = _ticker();
    const expiry = ($('pop-expiry') || {}).value;
    if (!t) { _setError('No ticker selected.'); return; }
    if (!expiry) { _setError('Pick an expiry first.'); return; }

    _startLoading('pop-spin', 'pop-run-label', 'Scanning…');
    _show('pop-forecast', false);
    try {
      const res = await fetch('/api/options/pop_scan?' + _params().toString());
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status));
      _popData = data;
      _popFilter = 'best';
      _renderAssumptions(data);
      _syncFilterChips();
      _renderTable();
      _show('pop-filters', true, 'flex');
      _show('pop-results', true);
    } catch (e) {
      _setError('Scan failed: ' + e.message);
      _show('pop-empty', true);
    } finally {
      _stopLoading('pop-spin', 'pop-run-label');
    }
  };

  window.runResearchScan = async function () {
    if (_popLoading) return;
    const t = _ticker();
    if (!t) { _setError('No ticker selected.'); return; }

    _startLoading('pop-research-spin', 'pop-research-label', 'Forecasting…');
    try {
      const num = (id) => { const v = parseFloat(($(id) || {}).value); return isFinite(v) ? v : null; };
      const p = new URLSearchParams({ ticker: t });
      const rf = num('pop-rf'); const comm = num('pop-comm');
      if (rf != null) p.set('risk_free', rf);
      if (comm != null) p.set('commission', comm);
      const res = await fetch('/api/options/research_scan?' + p.toString());
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status));
      _popData = data;
      _popFilter = 'top_return';
      _renderForecast(data);
      _renderAssumptions(data);
      _syncFilterChips();
      _renderTable();
      _show('pop-filters', true, 'flex');
      _show('pop-results', true);
    } catch (e) {
      _setError('Research scan failed: ' + e.message);
      if (!_popData) _show('pop-empty', true);
    } finally {
      _stopLoading('pop-research-spin', 'pop-research-label');
    }
  };

  /* ---------- rendering ---------- */

  function _renderForecast(data) {
    const el = $('pop-forecast');
    if (!el) return;
    const f = data.forecast || {};
    const dir = (f.direction || 'neutral');
    const dirColor = dir === 'bullish' ? 'var(--green)' : dir === 'bearish' ? 'var(--red)' : 'var(--muted)';
    const factors = (f.factors || []).map((x) =>
      '<span style="margin-right:12px;white-space:nowrap;">' + _esc(x.label || x.name) + ': ' +
      '<strong style="color:' + (x.direction === 'bullish' ? 'var(--green)' : x.direction === 'bearish' ? 'var(--red)' : 'var(--muted)') + ';">' +
      (x.contribution_pct > 0 ? '+' : '') + _esc(x.contribution_pct) + '%/yr</strong></span>'
    ).join('');
    el.innerHTML =
      '<div style="display:flex;flex-wrap:wrap;gap:14px;align-items:baseline;">' +
        '<strong style="color:var(--purple);">Research forecast</strong>' +
        '<span>Drift: <strong style="color:' + dirColor + ';">' + _esc(f.mu_annual_pct) + '%/yr (' + _esc(dir) + ')</strong></span>' +
        (f.expected_monthly_return_pct != null ? '<span>≈ ' + _esc(f.expected_monthly_return_pct) + '%/mo</span>' : '') +
        (f.confidence != null ? '<span>Signal alignment: <strong>' + _esc(f.confidence) + '</strong></span>' : '') +
        (data.best_expiry ? '<span>Best expiry: <strong style="color:var(--blue);">' + _esc(data.best_expiry) + '</strong></span>' : '') +
      '</div>' +
      (factors ? '<div style="margin-top:5px;color:var(--muted);">' + factors + '</div>' : '') +
      (f.disclaimer ? '<div style="margin-top:4px;font-size:10px;color:var(--muted);font-style:italic;">' + _esc(f.disclaimer) + '</div>' : '');
    el.style.display = 'block';
  }

  function _renderAssumptions(data) {
    const el = $('pop-assumptions');
    if (!el) return;
    const a = data.assumptions || {};
    const mv = a.my_view || {};
    const parts = [];
    if (data.spot != null) parts.push('Spot <strong>$' + data.spot + '</strong>');
    if (a.horizon_days != null) parts.push('Horizon <strong>' + a.horizon_days + 'd</strong>');
    if (mv.sigma_pct != null) parts.push('My-view σ <strong>' + mv.sigma_pct + '%</strong> (' + _esc(mv.sigma_source || '') + ')');
    if (mv.mu_pct != null) parts.push('My-view µ <strong>' + mv.mu_pct + '%/yr</strong> (' + _esc(mv.mu_source || '') + ')');
    if (data.rv_forecast_pct != null) parts.push('RV forecast <strong>' + data.rv_forecast_pct + '%</strong>');
    if (a.risk_free_pct != null) parts.push('r <strong>' + a.risk_free_pct + '%</strong>');
    if (a.commission_per_leg != null) parts.push('Comm <strong>$' + a.commission_per_leg + '/leg</strong>');
    el.innerHTML = parts.join(' · ') +
      '<span style="float:right;color:var(--muted);">Mode: <strong style="color:var(--blue);">' +
      (_popMode === 'market' ? 'MARKET (µ=r, σ=IV)' : 'MY VIEW') + '</strong></span>';
    el.style.display = 'block';
  }

  function _setFilter(f) {
    if (!f) return;
    _popFilter = f;
    _syncFilterChips();
    _renderTable();
  }

  function _syncFilterChips() {
    document.querySelectorAll('.pop-filter').forEach((c) => {
      const on = c.dataset.popFilter === _popFilter;
      c.classList.toggle('active', on);
      c.style.background = on ? 'rgba(63,185,80,.15)' : 'transparent';
      c.style.color = on ? 'var(--green)' : 'var(--muted)';
      c.style.borderColor = on ? 'rgba(63,185,80,.4)' : 'rgba(139,148,158,.3)';
    });
  }

  function _setSort(key) {
    if (_popSort.key === key) _popSort.dir *= -1;
    else _popSort = { key, dir: -1 };
    _renderTable();
  }

  function _rows() {
    if (!_popData) return [];
    let rows;
    switch (_popFilter) {
      case 'all':        rows = _popData.all || []; break;
      case 'top_return': rows = _popData.top_by_expected_return || []; break;
      case 'top_pop':    rows = _popData.top_by_pop || (_popData.all || []).slice().sort((a, b) => _val(b, 'pop_pct') - _val(a, 'pop_pct')).slice(0, 8); break;
      default:           rows = _popData.top_by_kelly || _popData.top_by_expected_return || _popData.all || [];
    }
    return rows.slice();
  }

  // Value respecting the MARKET/MY-VIEW mode for mode-dependent metrics.
  function _val(row, key) {
    const modeKeys = { pop_pct: 1, prob_itm_pct: 1, exp_pnl: 1, exp_return_pct: 1, kelly_pct: 1 };
    if (_popMode === 'market' && modeKeys[key] && row.market && row.market[key] != null) {
      return row.market[key];
    }
    return row[key];
  }

  function _sortVal(row, key) {
    switch (key) {
      case 'label':    return row.label || '';
      case 'strikes':  return (row.strikes || [0])[0];
      case 'dte':      return row.dte != null ? row.dte : ((row.assumptions || {}).horizon_days || 0);
      case 'breakeven': return (row.breakevens || [0])[0];
      default:         return _val(row, key) != null ? _val(row, key) : -Infinity;
    }
  }

  const _fmt = (v, dec) => (v == null || !isFinite(v)) ? '-' : Number(v).toFixed(dec != null ? dec : 2);
  const _fmtD = (v) => (v == null || !isFinite(v)) ? '-' : ('$' + Number(v).toFixed(2));
  const _retColor = (v) => v == null ? 'var(--muted)' : v >= 0 ? 'var(--green)' : 'var(--red)';

  function _renderTable() {
    const tb = $('pop-tbody');
    if (!tb) return;
    const rows = _rows().sort((a, b) => {
      const x = _sortVal(a, _popSort.key), y = _sortVal(b, _popSort.key);
      if (typeof x === 'string') return _popSort.dir * x.localeCompare(y);
      return _popSort.dir * ((x || 0) - (y || 0));
    });

    if (!rows.length) {
      tb.innerHTML = '<tr><td colspan="12" style="text-align:center;padding:24px;color:var(--muted);">No structures matched.</td></tr>';
      const cnt = $('pop-count'); if (cnt) cnt.textContent = '';
      return;
    }

    tb.innerHTML = rows.map((r, i) => {
      const ret = _val(r, 'exp_return_pct');
      const pnl = _val(r, 'exp_pnl');
      const pop = _val(r, 'pop_pct');
      const itm = _val(r, 'prob_itm_pct');
      const mktRet = (r.market || {}).exp_return_pct;
      const expTxt = r.expiry ? (r.expiry + (r.dte != null ? ' (' + r.dte + 'd)' : '')) :
        ((r.assumptions || {}).horizon_days != null ? (r.assumptions.horizon_days + 'd') : '-');
      const strikes = (r.strikes || []).join(' / ');
      const kindColor = /put/i.test(r.label || '') ? 'var(--red)' : 'var(--green)';
      return '<tr data-pop-row="' + i + '" style="cursor:pointer;border-bottom:1px solid rgba(255,255,255,.05);" ' +
             'onmouseover="this.style.background=\'rgba(88,166,255,.05)\'" onmouseout="this.style.background=\'\'">' +
        '<td style="padding:6px 8px;color:' + kindColor + ';font-weight:600;">' + _esc(r.label) + '</td>' +
        '<td style="padding:6px 8px;">' + _esc(strikes) + '</td>' +
        '<td style="padding:6px 8px;color:var(--muted);">' + _esc(expTxt) + '</td>' +
        '<td style="padding:6px 8px;text-align:right;">' + _fmtD(r.mid_per_contract != null ? r.mid_per_contract : r.net_premium) + '</td>' +
        '<td style="padding:6px 8px;text-align:right;">' + _esc((r.breakevens || []).map((b) => Number(b).toFixed(2)).join(' / ') || '-') + '</td>' +
        '<td style="padding:6px 8px;text-align:right;font-weight:700;">' + _fmt(pop, 1) + '%</td>' +
        '<td style="padding:6px 8px;text-align:right;color:var(--muted);">' + _fmt(itm, 1) + '%</td>' +
        '<td style="padding:6px 8px;text-align:right;font-weight:700;color:' + _retColor(ret) + ';">' + (ret == null ? '-' : ((ret > 0 ? '+' : '') + _fmt(ret, 1) + '%')) + '</td>' +
        '<td style="padding:6px 8px;text-align:right;color:' + _retColor(pnl) + ';">' + _fmtD(pnl) + '</td>' +
        '<td style="padding:6px 8px;text-align:right;color:var(--red);">' + _fmtD(r.max_loss) + '</td>' +
        '<td style="padding:6px 8px;text-align:right;">' + (r.rr == null ? '-' : _fmt(r.rr, 2)) + '</td>' +
        '<td style="padding:6px 8px;text-align:right;color:' + _retColor(mktRet) + ';">' + (mktRet == null ? '-' : ((mktRet > 0 ? '+' : '') + _fmt(mktRet, 1) + '%')) + '</td>' +
      '</tr>';
    }).join('');

    const cnt = $('pop-count');
    if (cnt) {
      const total = ((_popData || {}).counts || {}).structures;
      cnt.textContent = rows.length + ' shown' + (total ? (' of ' + total + ' structures scored') : '') +
        ' · mode: ' + (_popMode === 'market' ? 'MARKET' : 'MY VIEW') + ' · click a row for the payoff chart';
    }

    // Row click -> payoff chart.
    tb.querySelectorAll('tr[data-pop-row]').forEach((tr) => {
      tr.addEventListener('click', () => _drawPayoff(rows[parseInt(tr.dataset.popRow, 10)]));
    });
  }

  /* ---------- payoff chart ---------- */

  function _drawPayoff(row) {
    if (!row || !window.Chart) return;
    const legs = row.legs || [];
    if (!legs.length) return;

    const spot = (_popData || {}).spot || (row.strikes || [100])[0];
    const ks = row.strikes && row.strikes.length ? row.strikes : [spot];
    const lo = Math.min(spot, Math.min.apply(null, ks)) * 0.82;
    const hi = Math.max(spot, Math.max.apply(null, ks)) * 1.18;
    const n = 121;
    const xs = [], ys = [];
    const netDebit = row.net_debit != null ? row.net_debit : legs.reduce((s, l) => s + l.qty * l.mid, 0);
    for (let i = 0; i < n; i++) {
      const S = lo + (hi - lo) * i / (n - 1);
      let pay = 0;
      for (const l of legs) {
        const intr = l.type === 'call' ? Math.max(S - l.strike, 0) : Math.max(l.strike - S, 0);
        pay += l.qty * intr;
      }
      xs.push(S.toFixed(2));
      ys.push(((pay - netDebit) * 100));
    }

    _show('pop-payoff-wrap', true);
    const title = $('pop-payoff-title');
    if (title) title.textContent = 'Payoff at expiry - ' + (row.label || '');
    const sub = $('pop-payoff-sub');
    if (sub) {
      sub.textContent = 'Cost ' + _fmtD(row.net_premium) + ' · POP ' + _fmt(_val(row, 'pop_pct'), 1) +
        '% · max loss ' + _fmtD(row.max_loss) + (row.max_gain != null ? (' · max gain ' + _fmtD(row.max_gain)) : '');
    }

    const ctx = $('pop-payoff-chart');
    if (!ctx) return;
    if (_payoffChart) { _payoffChart.destroy(); _payoffChart = null; }
    _payoffChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: xs,
        datasets: [{
          label: 'P&L $/contract',
          data: ys,
          borderWidth: 2,
          pointRadius: 0,
          borderColor: '#58a6ff',
          fill: {
            target: { value: 0 },
            above: 'rgba(63,185,80,.15)',
            below: 'rgba(248,81,73,.12)',
          },
          tension: 0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => 'Price $' + items[0].label,
              label: (item) => 'P&L: $' + Number(item.raw).toFixed(0),
            },
          },
        },
        scales: {
          x: {
            ticks: { color: '#7d8b99', maxTicksLimit: 9 },
            grid: { color: 'rgba(255,255,255,.04)' },
          },
          y: {
            ticks: { color: '#7d8b99', callback: (v) => '$' + v },
            grid: { color: (c) => c.tick.value === 0 ? 'rgba(255,255,255,.35)' : 'rgba(255,255,255,.04)' },
          },
        },
      },
    });
  }
})();
