// Unusual options flow scanner tab.

(function () {
  'use strict';

  let _uoData      = [];   // raw hits from API
  let _uoFiltered  = [];   // after filters applied
  let _uoSortKey   = 'ticker';   // default: alphabetical by ticker
  let _uoSortAsc   = true;       // A -> Z
  let _uoSentFilter = 'all';
  let _uoSectFilter = 'all';     // sector filter - 'all' or a sector string
  let _uoNewOnly    = false;     // New Only toggle state (scan-time param)
  let _uoRunning   = false;
  let _uoAbortController = null;
  let _uoProgressTimer = null;

  function _setUoFetchingUI(active) {
    const btn   = document.getElementById('uo-run-btn');
    const stop  = document.getElementById('uo-stop-btn');
    const spin  = document.getElementById('uo-spin');
    const label = document.getElementById('uo-run-label');
    if (btn)   btn.disabled = active;
    if (stop)  stop.style.display = active ? 'inline-flex' : 'none';
    if (spin)  spin.style.display = active ? 'inline-block' : 'none';
    if (label) label.textContent = active ? 'Scanning...' : 'Run Scan';
  }

  function stopUoScan() {
    if (_uoAbortController) {
      _uoAbortController.abort();
      _uoAbortController = null;
    }
    if (_uoProgressTimer) {
      clearInterval(_uoProgressTimer);
      _uoProgressTimer = null;
    }
    fetch('/api/options/unusual/cancel', { method: 'POST' }).catch(() => {});
    _uoRunning = false;
    _setUoFetchingUI(false);
    const progWrap  = document.getElementById('uo-progress-wrap');
    const statusTxt = document.getElementById('uo-status-text');
    const errEl     = document.getElementById('uo-error-state');
    if (progWrap) progWrap.style.display = 'none';
    if (statusTxt) {
      statusTxt.textContent = 'Scan stopped.';
      statusTxt.style.display = 'block';
    }
    if (errEl) {
      errEl.style.display = 'block';
      errEl.textContent = 'Scan stopped. Server workers will skip remaining tickers.';
    }
  }
  window.stopUoScan = stopUoScan;

  function _uoFmt(n) {
    if (n === null || n === undefined || isNaN(n)) return '-';
    if (n >= 1e6)  return '$' + (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3)  return '$' + (n / 1e3).toFixed(1) + 'K';
    return '$' + n.toFixed(0);
  }

  function _uoFlag(flags) {
    const MAP = {
      high_vol_oi:   { label: 'Vol/OI',    color: '#58a6ff' },
      iv_spike:      { label: 'IV Spike',  color: '#f85149' },
      otm_sweep:     { label: 'OTM',       color: '#d29922' },
      large_premium: { label: ' Big $',     color: '#bc8cff' },
      cp_divergence: { label: 'CP Skew',   color: '#3fb950' },
    };
    return (flags || []).map(f => {
      const m = MAP[f] || { label: f, color: 'var(--muted)' };
      return `<span style="display:inline-block;padding:2px 7px;border-radius:10px;font-size:9px;
                            font-weight:700;letter-spacing:.2px;white-space:nowrap;
                            background:${m.color}18;color:${m.color};
                            border:1px solid ${m.color}40;margin:1px 1px;">${m.label}</span>`;
    }).join('');
  }

  function _uoSentBadge(s) {
    if (s === 'bullish') return `<span style="display:inline-block;padding:2px 8px;border-radius:10px;
      font-size:10px;font-weight:700;background:rgba(63,185,80,.12);color:var(--green);
      border:1px solid rgba(63,185,80,.3);">^ Bull</span>`;
    if (s === 'bearish') return `<span style="display:inline-block;padding:2px 8px;border-radius:10px;
      font-size:10px;font-weight:700;background:rgba(248,81,73,.12);color:#f85149;
      border:1px solid rgba(248,81,73,.3);">v Bear</span>`;
    return `<span style="display:inline-block;padding:2px 8px;border-radius:10px;
      font-size:10px;font-weight:700;background:rgba(210,153,34,.12);color:var(--amber);
      border:1px solid rgba(210,153,34,.3);">◆ Mix</span>`;
  }

  // Score tier definitions - used by both badge and row stripe
  const _UO_TIERS = [
    { min: 0.65, col: '#f85149', bg: 'rgba(248,81,73,.18)',   border: 'rgba(248,81,73,.50)',  label: 'HOT'  },
    { min: 0.45, col: '#d29922', bg: 'rgba(210,153,34,.16)',  border: 'rgba(210,153,34,.45)', label: 'HIGH' },
    { min: 0.20, col: '#58a6ff', bg: 'rgba(88,166,255,.14)',  border: 'rgba(88,166,255,.40)', label: 'MOD'  },
    { min: 0.00, col: '#8b949e', bg: 'rgba(139,148,158,.10)', border: 'rgba(139,148,158,.25)',label: 'LOW'  },
  ];
  function _uoTier(score) {
    return _UO_TIERS.find(t => (score || 0) >= t.min) || _UO_TIERS[3];
  }

  function _uoScoreBadge(score) {
    const pct = Math.round((score || 0) * 100);
    const t   = _uoTier(score);
    return `<span class="uo-col-score" style="display:inline-flex;align-items:center;gap:4px;
                padding:4px 10px;border-radius:20px;
                background:${t.bg};border:1px solid ${t.border};
                font-size:11px;font-weight:800;color:${t.col};
                letter-spacing:.2px;white-space:nowrap;">
              ${pct}% <span style="font-size:9px;font-weight:600;opacity:.8;">${t.label}</span>
            </span>`;
  }

  const _KEY_TO_BTN = {
    unusual_score:  'uo-srt-score', total_premium:  'uo-srt-prem',
    vol_oi_ratio:   'uo-srt-voloi', implied_vol:    'uo-srt-iv',
    volume:         'uo-srt-vol',   days_to_expiry: 'uo-srt-dte',
    ticker:         'uo-srt-ticker',percent_change: 'uo-srt-pctchg',
    sector:         'uo-srt-sector',
  };
  const _BTN_BASE_LABEL = {
    'uo-srt-score':  'Score',   'uo-srt-prem':   'Premium $',
    'uo-srt-voloi':  'Vol/OI',  'uo-srt-iv':     'IV%',
    'uo-srt-vol':    'Volume',  'uo-srt-dte':    'DTE',
    'uo-srt-ticker': 'A-Z',     'uo-srt-pctchg': '%Chg',
  };

  function uoSetSort(key) {
    if (_uoSortKey === key) {
      _uoSortAsc = !_uoSortAsc;
    } else {
      _uoSortKey = key;
      // Default direction: ascending for ticker, descending for everything else
      _uoSortAsc = (key === 'ticker');
    }

    // Update button states and labels
    document.querySelectorAll('[id^="uo-srt-"]').forEach(b => {
      b.classList.remove('active');
      const base = _BTN_BASE_LABEL[b.id];
      if (base) b.textContent = base;   // reset arrow
    });
    const btnId = _KEY_TO_BTN[key];
    const btn   = document.getElementById(btnId);
    if (btn) {
      btn.classList.add('active');
      const base  = _BTN_BASE_LABEL[btnId] || key;
      const arrow = _uoSortAsc ? ' ↑' : ' ↓';
      btn.textContent = base + arrow;
    }
    uoRenderTable();
  }
  window.uoSetSort = uoSetSort;

  function uoFilter(sentiment) {
    _uoSentFilter = sentiment;
    // button ids: uo-flt-all / uo-flt-bull / uo-flt-bear / uo-flt-mixed
    // sentiments : 'all'     / 'bullish'   / 'bearish'   / 'mixed'
    const _S2ID = { all:'all', bullish:'bull', bearish:'bear', mixed:'mixed' };
    Object.entries(_S2ID).forEach(([s, id]) => {
      const b = document.getElementById('uo-flt-' + id);
      if (b) b.classList.toggle('active', s === sentiment);
    });
    uoRenderTable();
  }
  window.uoFilter = uoFilter;

  function uoToggleNewOnly() {
    _uoNewOnly = !_uoNewOnly;
    const btn = document.getElementById('uo-new-only-btn');
    if (btn) {
      btn.classList.toggle('active', _uoNewOnly);
      btn.style.color = _uoNewOnly ? '#f85149' : '';
    }
  }
  window.uoToggleNewOnly = uoToggleNewOnly;

  function uoSectFilter(sector) {
    _uoSectFilter = sector;
    // Update chip button active states
    document.querySelectorAll('[id^="uo-sect-"]').forEach(b => {
      b.classList.toggle('active', b.dataset.sect === sector);
    });
    uoRenderTable();
  }
  window.uoSectFilter = uoSectFilter;

  function uoRenderTable() {
    const search = (document.getElementById('uo-search')?.value || '').toUpperCase().trim();

    let rows = _uoData.slice();

    // sentiment filter
    if (_uoSentFilter !== 'all') {
      rows = rows.filter(r => r.sentiment === _uoSentFilter);
    }
    // sector filter
    if (_uoSectFilter !== 'all') {
      rows = rows.filter(r => (r.sector || 'Other') === _uoSectFilter);
    }
    // search
    if (search) {
      rows = rows.filter(r => r.ticker.includes(search));
    }

    // sort
    rows.sort((a, b) => {
      let av = a[_uoSortKey], bv = b[_uoSortKey];
      if (typeof av === 'string') av = av.toLowerCase();
      if (typeof bv === 'string') bv = bv.toLowerCase();
      if (av < bv) return _uoSortAsc ? -1 : 1;
      if (av > bv) return _uoSortAsc ? 1 : -1;
      return 0;
    });

    _uoFiltered = rows;
    const cnt = document.getElementById('uo-row-count');
    if (cnt) cnt.textContent = rows.length + ' contracts';

    const tbody = document.getElementById('uo-tbody');
    if (!tbody) return;

    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="17" style="text-align:center;padding:32px;color:var(--muted);font-size:13px;">No contracts match the current filters.</td></tr>';
      return;
    }

    tbody.innerHTML = rows.map(r => {
      const tier = _uoTier(r.unusual_score);

      const itm       = r.in_the_money
        ? `<span style="color:var(--green);font-size:10px;font-weight:700;
                         padding:2px 6px;background:rgba(63,185,80,.12);
                         border:1px solid rgba(63,185,80,.25);border-radius:5px;">ITM</span>`
        : `<span style="color:var(--muted);font-size:10px;
                         padding:2px 6px;background:rgba(139,148,158,.08);
                         border:1px solid rgba(139,148,158,.18);border-radius:5px;">OTM</span>`;

      const typeColor = r.option_type === 'call' ? '#3fb950' : '#f85149';
      const typeBg    = r.option_type === 'call' ? 'rgba(63,185,80,.10)'  : 'rgba(248,81,73,.10)';
      const typeBrd   = r.option_type === 'call' ? 'rgba(63,185,80,.30)'  : 'rgba(248,81,73,.30)';
      const typeLbl   = r.option_type === 'call' ? '^ CALL' : 'v PUT';

      // Vol/OI color tiers
      const voiColor = r.vol_oi_ratio >= 20 ? '#f85149'
                     : r.vol_oi_ratio >= 10 ? '#d29922'
                     : r.vol_oi_ratio >=  5 ? '#58a6ff'
                     :                        'var(--text)';

      // IV color (high IV = red)
      const ivColor = r.implied_vol >= 150 ? '#f85149'
                    : r.implied_vol >= 80  ? '#d29922'
                    :                        'var(--text)';

      // Premium color
      const premColor = r.total_premium >= 1e6 ? '#f85149'
                      : r.total_premium >= 2e5 ? '#d29922'
                      : r.total_premium >= 5e4 ? '#58a6ff'
                      :                          'var(--text)';

      // Expiry date short form + DTE badge
      const expShort  = r.expiry.slice(5);   // "05-29" from "2026-05-29"
      const dte       = Math.round(r.days_to_expiry);
      const dteColor  = dte <= 7 ? '#f85149' : dte <= 14 ? '#d29922' : 'var(--muted)';

      // Vol spike badge (Hot mode only)
      const volSpikeBadge = r.vol_spike
        ? `<div style="font-size:9px;color:var(--amber);font-weight:700;margin-top:2px;white-space:nowrap;"
                title="Today's volume is ${r.vol_spike.vol_ratio}× the 20-day average"> ${r.vol_spike.vol_ratio}× vol</div>`
        : '';

      return `<tr style="border-left:3px solid ${tier.border};cursor:pointer;"
                  title="Open contract tracker in a new tab" data-contract="1"
                  data-ticker="${r.ticker}" data-expiry="${r.expiry}" data-strike="${r.strike}" data-type="${r.option_type}">
        <td style="border-left:none;">
          <span style="cursor:pointer;color:#58a6ff;font-weight:800;font-size:13px;"
                onclick="uoLoadTicker('${r.ticker}')"
                title="Load ${r.ticker}">${r.ticker}</span>
          ${volSpikeBadge}
        </td>
        <td>
          <span style="color:${typeColor};font-weight:700;font-size:10px;
                       background:${typeBg};padding:3px 8px;border-radius:5px;
                       border:1px solid ${typeBrd};letter-spacing:.4px;">${typeLbl}</span>
        </td>
        <td>
          <div style="font-size:11px;font-weight:600;color:var(--text);">${expShort}</div>
          <div style="font-size:10px;font-weight:700;color:${dteColor};margin-top:1px;">${dte}d</div>
        </td>
        <td style="font-weight:700;font-size:12px;">$${r.strike.toFixed(2)}</td>
        <td style="color:var(--muted);font-size:11px;">$${r.spot.toFixed(2)}</td>
        <td>${itm}</td>
        <td style="font-weight:700;">${(r.volume||0).toLocaleString()}</td>
        <td style="color:var(--muted);">${(r.open_interest||0).toLocaleString()}</td>
        <td style="font-weight:800;font-size:12px;color:${voiColor};">${r.vol_oi_ratio.toFixed(1)}×</td>
        <td style="font-weight:600;color:${ivColor};">${r.implied_vol.toFixed(1)}%</td>
        <td style="${(()=>{const v=r.percent_change||0;const c=v>20?'#3fb950':v>5?'#58a6ff':v<-20?'#f85149':v<-5?'#d29922':'var(--muted)';return 'font-weight:700;color:'+c+';';})()}">${(r.percent_change>=0?'+':'')+(r.percent_change||0).toFixed(1)}%</td>
        <td style="font-weight:700;color:${premColor};font-size:12px;">${_uoFmt(r.total_premium)}</td>
        <td>${_uoScoreBadge(r.unusual_score)}</td>
        <td class="uo-col-flags">${_uoFlag(r.flags)}</td>
        <td>${_uoSentBadge(r.sentiment)}</td>
        <td>
          <span onclick="uoSectFilter('${(r.sector||'Other').replace(/'/g,'\\\'')}')"
                style="cursor:pointer;display:inline-block;padding:2px 7px;border-radius:10px;
                       font-size:9px;font-weight:700;letter-spacing:.3px;white-space:nowrap;
                       background:rgba(139,148,158,.10);color:var(--muted);
                       border:1px solid rgba(139,148,158,.20);transition:background .12s;"
                onmouseover="this.style.background='rgba(88,166,255,.15)';this.style.color='#58a6ff'"
                onmouseout="this.style.background='rgba(139,148,158,.10)';this.style.color='var(--muted)'"
                title="Filter by ${r.sector||'Other'}">${r.sector||'Other'}</span>
        </td>
        <td>
          <button onclick="uoLoadTicker('${r.ticker}')"
                  style="background:rgba(88,166,255,.08);border:1px solid rgba(88,166,255,.25);
                         color:#58a6ff;border-radius:5px;padding:4px 10px;font-size:11px;
                         cursor:pointer;font-weight:600;transition:background .12s;"
                  onmouseover="this.style.background='rgba(88,166,255,.18)'"
                  onmouseout="this.style.background='rgba(88,166,255,.08)'"
                  title="Load in main chart">↗ Load</button>
        </td>
      </tr>`;
    }).join('');
  }
  window.uoRenderTable = uoRenderTable;

  function uoLoadTicker(ticker) {
    const inp = document.getElementById('quick-ticker');
    if (inp) { inp.value = ticker; }
    if (typeof applyQuickTicker === 'function') applyQuickTicker();
  }
  window.uoLoadTicker = uoLoadTicker;

  async function uoRunScan() {
    if (_uoRunning) return;
    _uoRunning = true;
    _uoAbortController = new AbortController();

    const progWrap  = document.getElementById('uo-progress-wrap');
    const progBar   = document.getElementById('uo-progress-bar');
    const statusTxt = document.getElementById('uo-status-text');
    const emptyEl   = document.getElementById('uo-empty-state');
    const errEl     = document.getElementById('uo-error-state');
    const resEl     = document.getElementById('uo-results');

    // UI: loading state
    _setUoFetchingUI(true);
    if (progWrap) progWrap.style.display = 'block';
    if (progBar)  progBar.style.width = '5%';
    if (emptyEl)  emptyEl.style.display = 'none';
    if (errEl)    errEl.style.display = 'none';
    if (resEl)    resEl.style.display = 'none';

    // Hide any previous vol-spike panel
    const spikePanel = document.getElementById('uo-vol-spike-panel');
    if (spikePanel) spikePanel.style.display = 'none';

    // Read params
    const watchlist  = document.getElementById('uo-watchlist')?.value || '';
    const maxDte     = parseInt(document.getElementById('uo-dte')?.value) || 45;
    const volOi      = parseFloat(document.getElementById('uo-voloi')?.value) || 3;
    const minPrem    = parseFloat(document.getElementById('uo-minprem')?.value) || 200000;
    const isHot      = watchlist === 'hot';

        let fetchUrl;
    if (isHot) {
      const hp = new URLSearchParams({
        max_dte: maxDte, vol_oi_threshold: volOi, min_premium: minPrem,
        top_n: 60
      });
      fetchUrl = '/api/options/hot?' + hp.toString();
    } else {
      const params = new URLSearchParams({
        max_dte: maxDte, vol_oi_threshold: volOi, min_premium: minPrem,
        new_positions_only: _uoNewOnly ? 'true' : 'false', top_n: 100
      });
      if (watchlist) params.set('watchlist', watchlist);
      fetchUrl = '/api/options/unusual?' + params.toString();
    }

    // Fake progress animation while waiting
    let pct = 5;
    _uoProgressTimer = setInterval(() => {
      pct = Math.min(pct + (Math.random() * (isHot ? 2 : 3)), 88);
      if (progBar) progBar.style.width = pct + '%';
      if (statusTxt) {
        statusTxt.textContent = isHot
          ? `Hot scan: scanning ~500 stocks for volume spikes, then checking options...`
          : `Scanning ${watchlist || 'all_optionable (~500)'} - fetching options chains... (this takes a few minutes)`;
      }
    }, 1200);

    try {
      const resp = await fetch(fetchUrl, { signal: _uoAbortController.signal });
      clearInterval(_uoProgressTimer);
      _uoProgressTimer = null;
      if (!resp.ok) {
        const errBody = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(errBody.detail || resp.statusText);
      }
      const data = await resp.json();

      // Complete progress
      if (progBar) progBar.style.width = '100%';
      setTimeout(() => { if (progWrap) progWrap.style.display = 'none'; }, 600);

      _uoData = data.hits || [];
      _uoSentFilter = 'all';
      _uoSectFilter = 'all';
      // Reset to default sort (ticker A->Z) on each fresh scan
      _uoSortKey = 'ticker';
      _uoSortAsc = true;
      document.querySelectorAll('[id^="uo-srt-"]').forEach(b => {
        b.classList.remove('active');
        const base = _BTN_BASE_LABEL[b.id]; if (base) b.textContent = base;
      });
      const tickerBtn = document.getElementById('uo-srt-ticker');
      if (tickerBtn) { tickerBtn.classList.add('active'); tickerBtn.textContent = 'A-Z ↑'; }

      // Build sector filter chips from scan results
      const sectChipsEl = document.getElementById('uo-sect-chips');
      if (sectChipsEl && _uoData.length) {
        const sectCounts = {};
        _uoData.forEach(r => { const s = r.sector || 'Other'; sectCounts[s] = (sectCounts[s]||0)+1; });
        const sectors = Object.entries(sectCounts).sort((a,b) => b[1]-a[1]);
        sectChipsEl.innerHTML = '<span style="font-size:10px;color:var(--muted);margin-right:2px;white-space:nowrap;">Sector:</span>';
        const allChip = document.createElement('button');
        allChip.id = 'uo-sect-all';
        allChip.dataset.sect = 'all';
        allChip.className = 'scan-sort-btn active';
        allChip.style.cssText = 'font-size:10px;padding:3px 8px;';
        allChip.textContent = `All (${_uoData.length})`;
        allChip.onclick = () => uoSectFilter('all');
        sectChipsEl.appendChild(allChip);
        sectors.forEach(([sect, cnt]) => {
          const chip = document.createElement('button');
          chip.id = 'uo-sect-' + sect.replace(/[^a-z0-9]/gi,'_');
          chip.dataset.sect = sect;
          chip.className = 'scan-sort-btn';
          chip.style.cssText = 'font-size:10px;padding:3px 8px;';
          chip.textContent = `${sect} (${cnt})`;
          chip.onclick = () => uoSectFilter(sect);
          sectChipsEl.appendChild(chip);
        });
        sectChipsEl.style.display = 'flex';
      } else if (sectChipsEl) {
        sectChipsEl.style.display = 'none';
      }

      uoFilter('all');  // resets sentiment filter buttons + re-renders

      if (isHot && data.volume_spikes && data.volume_spikes.length) {
        _uoRenderVolSpikePanel(data.volume_spikes, data.summary || {});
      }

      // Summary pills
      const s = data.summary || {};
      const setTxt = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };
      if (isHot) {
        setTxt('uo-pill-total',   `${s.total_hits || 0} unusual contracts`);
        setTxt('uo-pill-bull',    `^ ${s.bullish_count || 0} bullish`);
        setTxt('uo-pill-bear',    `v ${s.bearish_count || 0} bearish`);
        setTxt('uo-pill-mixed',   `◆ ${s.mixed_count || 0} mixed`);
        setTxt('uo-pill-scanned',
          ` ${s.vol_spike_found || 0} vol spikes found from ${s.universe_scanned || 500} stocks, ${s.tickers_with_hits || 0} with unusual options`);
      } else {
        setTxt('uo-pill-total',   `${s.total_hits || 0} unusual contracts`);
        setTxt('uo-pill-bull',    `^ ${s.bullish_count || 0} bullish`);
        setTxt('uo-pill-bear',    `v ${s.bearish_count || 0} bearish`);
        setTxt('uo-pill-mixed',   `◆ ${s.mixed_count || 0} mixed`);
        const delistedNote = s.delisted_count ? ` ,  ${s.delisted_count} delisted removed` : '';
        setTxt('uo-pill-scanned', `${s.tickers_scanned || 0} tickers scanned, ${s.tickers_with_hits || 0} with hits${delistedNote}`);
      }
      const ts = data.scanned_at ? new Date(data.scanned_at).toLocaleTimeString() : '';
      setTxt('uo-scanned-at', ts ? `Updated ${ts}` : '');

      // Tab badge
      const badge = document.getElementById('tab-unusual-opts-badge');
      if (badge) { badge.textContent = (s.total_hits || 0) + ' hits'; }

      if (resEl) resEl.style.display = 'block';
      if (s.cancelled && statusTxt) {
        statusTxt.textContent = `Scan stopped early - ${s.total_hits || 0} contracts collected so far.`;
        statusTxt.style.display = 'block';
      } else if (!_uoData.length) {
        if (statusTxt) { statusTxt.textContent = 'No unusual activity found with current filters.'; statusTxt.style.display = 'block'; }
      }

    } catch (err) {
      if (_uoProgressTimer) {
        clearInterval(_uoProgressTimer);
        _uoProgressTimer = null;
      }
      if (progWrap) progWrap.style.display = 'none';
      if (err.name === 'AbortError') {
        fetch('/api/options/unusual/cancel', { method: 'POST' }).catch(() => {});
        if (statusTxt) {
          statusTxt.textContent = 'Scan stopped.';
          statusTxt.style.display = 'block';
        }
        if (errEl) {
          errEl.style.display = 'block';
          errEl.textContent = 'Scan stopped. Server workers will skip remaining tickers.';
        }
      } else if (errEl) {
        errEl.style.display = 'block';
        errEl.textContent = 'Warning: Scan failed: ' + err.message;
      }
      console.error('[uoScan]', err);
    } finally {
      _uoRunning = false;
      _uoAbortController = null;
      _setUoFetchingUI(false);
    }
  }
  window.uoRunScan = uoRunScan;

  function _uoRenderVolSpikePanel(spikes, summary) {
    // Find or create the panel element (injected just before uo-results)
    let panel = document.getElementById('uo-vol-spike-panel');
    if (!panel) {
      panel = document.createElement('div');
      panel.id = 'uo-vol-spike-panel';
      panel.style.cssText = 'margin-bottom:14px;padding:12px 16px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;';
      const resEl = document.getElementById('uo-results');
      if (resEl && resEl.parentNode) resEl.parentNode.insertBefore(panel, resEl);
    }

    const fmtVol = v => v >= 1e6 ? (v/1e6).toFixed(1)+'M' : v >= 1e3 ? (v/1e3).toFixed(0)+'K' : v;
    const rows = spikes.slice(0, 20).map(s => `
      <tr style="border-bottom:1px solid var(--border);">
        <td style="padding:5px 8px;font-weight:700;color:var(--blue);cursor:pointer;" onclick="uoLoadTicker('${s.ticker}')" title="Load ${s.ticker} in chart">${s.ticker}</td>
        <td style="padding:5px 8px;text-align:right;">${s.price ? '$'+s.price.toFixed(2) : '-'}</td>
        <td style="padding:5px 8px;text-align:right;color:var(--amber);font-weight:700;">${s.vol_ratio.toFixed(1)}×</td>
        <td style="padding:5px 8px;text-align:right;">${fmtVol(s.today_volume)}</td>
        <td style="padding:5px 8px;text-align:right;color:var(--muted);">${fmtVol(s.avg_volume)}</td>
        <td style="padding:5px 8px;font-size:10px;color:var(--muted);">${s.sector || '-'}</td>
      </tr>`).join('');

    panel.innerHTML = `
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
        <span style="font-size:13px;font-weight:700;"> Volume Spikes Detected</span>
        <span style="font-size:11px;color:var(--muted);">${summary.vol_spike_found || spikes.length} stocks with unusual volume from ${summary.universe_scanned || 500} scanned</span>
      </div>
      <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:12px;">
          <thead>
            <tr style="border-bottom:2px solid var(--border);">
              <th style="padding:4px 8px;text-align:left;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.3px;">Ticker</th>
              <th style="padding:4px 8px;text-align:right;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.3px;">Price</th>
              <th style="padding:4px 8px;text-align:right;color:var(--amber);font-size:10px;text-transform:uppercase;letter-spacing:.3px;">Vol Spike</th>
              <th style="padding:4px 8px;text-align:right;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.3px;">Today Vol</th>
              <th style="padding:4px 8px;text-align:right;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.3px;">Avg Vol</th>
              <th style="padding:4px 8px;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.3px;">Sector</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
    panel.style.display = 'block';
  }

})();

if (window.__mc_trader_modular__) {
  window.__mc_trader_modular__.extracted.push('tabs/unusual-options.js');
}
