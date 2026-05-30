

//
// Owns:
//   • Unusual Activity table (DTE filter pills, sortable columns, sentiment tag)
//
// Hooks into the existing sentiment-panel flow: when the inline _renderSentimentPanel()
// runs in dashboard.html it calls window.renderUnusualActivity(data) - see the
// thin shim at the bottom of this file.
//
// The volume tooltips and the "Flow direction unavailable" disclaimer are pure
// HTML in dashboard.html (no JS needed).
//
// Public globals (used by inline onclick=... attributes):
//   • setUnusualFilter(filter)

(function () {
  'use strict';

  let _unusualData = [];     // last unusual_activity[] from /api/sentiment
  let _uaFilter    = 'all';  // 'all' | '0-7' | '8-30' | '31-60' | 'leaps'
  let _uaSortKey   = 'vol_oi';
  let _uaSortAsc   = false;

  function _fmtVol(n) {
    if (n == null || isNaN(n)) return '-';
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
    if (n >= 1_000)     return (n / 1_000).toFixed(1) + 'K';
    return Number(n).toLocaleString();
  }

  function _fmtDollar(n) {
    if (n == null || isNaN(n)) return '-';
    if (n >= 1_000_000_000) return '$' + (n / 1_000_000_000).toFixed(2) + 'B';
    if (n >= 1_000_000)     return '$' + (n / 1_000_000).toFixed(2) + 'M';
    if (n >= 1_000)         return '$' + (n / 1_000).toFixed(1) + 'K';
    return '$' + Number(n).toFixed(0);
  }

  function _inBucket(dte, bucket) {
    if (bucket === 'all')   return true;
    if (bucket === '0-7')   return dte >= 0  && dte <= 7;
    if (bucket === '8-30')  return dte >= 8  && dte <= 30;
    if (bucket === '31-60') return dte >= 31 && dte <= 60;
    if (bucket === 'leaps') return dte > 60;
    return true;
  }

  // IMPORTANT: yfinance does not expose aggressor side. The "tag" below is
  // descriptive only - it tells the user what TYPE of unusual flow it is, not
  // whether the flow is buyer- or seller-initiated. The disclaimer banner at
  // the top of the Options tab makes this explicit.
  function _sentimentTag(c) {
    const ratio = c.vol_oi || 0;
    const itm   = c.in_money ? ' · ITM' : '';
    // Pure type-based labels with intensity from vol/OI ratio.
    if (c.type === 'call') {
      if (ratio >= 5)  return { label: '📞 Call · Extreme',     color: 'var(--green)', weight: 800 };
      if (ratio >= 3)  return { label: '📞 Call · Heavy' + itm, color: 'var(--green)', weight: 700 };
      return                  { label: '📞 Call' + itm,         color: 'var(--green)', weight: 600 };
    }
    if (ratio >= 5)    return { label: '🔻 Put · Extreme',      color: 'var(--red)',   weight: 800 };
    if (ratio >= 3)    return { label: '🔻 Put · Heavy' + itm,  color: 'var(--red)',   weight: 700 };
    return                    { label: '🔻 Put' + itm,          color: 'var(--red)',   weight: 600 };
  }

  function renderUnusualActivity(data) {
    const ua = (data && data.options_flow && data.options_flow.unusual_activity) || [];
    _unusualData = Array.isArray(ua) ? ua : [];
    _renderTable();
  }

  function _renderTable() {
    const tbody    = document.getElementById('ua-tbody');
    const emptyEl  = document.getElementById('ua-empty');
    const countEl  = document.getElementById('ua-count');
    if (!tbody) return;

    // Filter by DTE bucket
    let rows = _unusualData.filter(c => _inBucket(c.dte, _uaFilter));

    // Sort
    rows.sort((a, b) => {
      let av = a[_uaSortKey];
      let bv = b[_uaSortKey];
      if (typeof av === 'string') { av = av.toLowerCase(); bv = (bv || '').toLowerCase(); }
      if (av == null) av = -Infinity;
      if (bv == null) bv = -Infinity;
      if (av < bv) return _uaSortAsc ? -1 : 1;
      if (av > bv) return _uaSortAsc ? 1 : -1;
      return 0;
    });

    if (countEl) {
      const total = _unusualData.length;
      countEl.textContent = `${rows.length} of ${total} contract${total === 1 ? '' : 's'}`;
    }

    if (rows.length === 0) {
      tbody.innerHTML = '';
      if (emptyEl) emptyEl.style.display = 'block';
      const tbl = document.getElementById('ua-table');
      if (tbl) tbl.style.display = 'none';
      return;
    }

    if (emptyEl) emptyEl.style.display = 'none';
    const tbl = document.getElementById('ua-table');
    if (tbl) tbl.style.display = 'table';

    tbody.innerHTML = rows.map(c => {
      const typeColor = c.type === 'call' ? 'var(--green)' : 'var(--red)';
      const typeIcon  = c.type === 'call' ? '📞 Call' : '🔻 Put';
      const itmStr    = c.in_money ? ' ✅' : '';
      const ratioColor =
        c.vol_oi >= 5 ? 'var(--amber)' :
        c.vol_oi >= 3 ? 'var(--blue)'  :
        'var(--text)';
      const ratioWeight = c.vol_oi >= 3 ? 700 : 500;
      const chgColor =
        c.pct_change == null ? 'var(--muted)' :
        c.pct_change > 0 ? 'var(--green)' :
        c.pct_change < 0 ? 'var(--red)' : 'var(--muted)';
      const chgStr   = c.pct_change == null ? '-'
        : (c.pct_change >= 0 ? '+' : '') + c.pct_change.toFixed(1) + '%';
      const flowColor =
        c.flow == null ? 'var(--muted)' :
        c.flow >= 1_000_000 ? 'var(--amber)' :
        c.flow >= 100_000   ? 'var(--text)'  :
        'var(--muted)';
      const dteBadge = c.leaps
        ? `<span style="display:inline-block;font-size:9px;font-weight:700;padding:1px 6px;border-radius:8px;background:rgba(188,140,255,.15);color:var(--purple);margin-left:6px;">LEAPS</span>`
        : '';
      const tag = _sentimentTag(c);

      return `<tr style="border-bottom:1px solid rgba(255,255,255,.04);">
        <td style="font-weight:700;padding:6px 8px;">$${c.strike}${itmStr}</td>
        <td style="color:${typeColor};padding:6px 8px;font-weight:600;">${typeIcon}</td>
        <td style="color:var(--muted);padding:6px 8px;font-size:11px;">${c.expiry}</td>
        <td style="text-align:right;padding:6px 8px;">${c.dte}d${dteBadge}</td>
        <td style="text-align:right;padding:6px 8px;font-weight:600;">${_fmtVol(c.volume)}</td>
        <td style="text-align:right;padding:6px 8px;color:var(--muted);">${_fmtVol(c.oi)}</td>
        <td style="text-align:right;padding:6px 8px;color:${ratioColor};font-weight:${ratioWeight};">${c.vol_oi.toFixed(2)}×</td>
        <td style="text-align:right;padding:6px 8px;color:var(--muted);">${_fmtDollar(c.premium)}</td>
        <td style="text-align:right;padding:6px 8px;color:${flowColor};font-weight:600;">${_fmtDollar(c.flow)}</td>
        <td style="text-align:right;padding:6px 8px;color:${chgColor};font-weight:600;">${chgStr}</td>
        <td style="padding:6px 8px;color:${tag.color};font-weight:${tag.weight};font-size:11px;">${tag.label}</td>
      </tr>`;
    }).join('');
  }

  function setUnusualFilter(filter) {
    _uaFilter = filter;
    document.querySelectorAll('[data-ua-filter]').forEach(b => {
      if (b.getAttribute('data-ua-filter') === filter) b.classList.add('active');
      else                                              b.classList.remove('active');
    });
    _renderTable();
  }

  document.addEventListener('click', (e) => {
    const th = e.target.closest('[data-ua-sort]');
    if (!th) return;
    const key = th.getAttribute('data-ua-sort');
    if (_uaSortKey === key) {
      _uaSortAsc = !_uaSortAsc;
    } else {
      _uaSortKey = key;
      // Numeric columns default to descending on first click; text columns ascending.
      _uaSortAsc = (key === 'type' || key === 'expiry');
    }
    _renderTable();
  });

  window.setUnusualFilter      = setUnusualFilter;
  window.renderUnusualActivity = renderUnusualActivity;

  // Update the index.js marker so the dashboard's "what's loaded" flag is accurate.
  if (window.__mc_trader_modular__) {
    window.__mc_trader_modular__.extracted.push('tabs/options.js');
  }
})();
