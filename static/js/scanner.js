
// Scanner logic

// Global state: _scanData, _scanSortKey, _scanSortAsc, _scanFilter, _scanFakeProgress
// Public functions (used via inline onclick in dashboard.html):
//   runScanner, setScanSort, filterScan, renderScanTable, loadTicker
// Depends on globals defined later in main app JS:
//   currentConfig, applySettings, log

let _scanData    = [];
let _scanSortKey = 'score';
let _scanSortAsc = false;
let _scanFilter  = 'all';
let _scanFakeProgress = null;

document.getElementById('scan-watchlist').addEventListener('change', function() {
  document.getElementById('scan-custom').style.display =
    this.value === 'custom' ? 'inline-block' : 'none';
});

function _startFakeProgress() {
  const bar = document.getElementById('scan-progress-bar');
  const wrap = document.getElementById('scan-progress-wrap');
  const txt  = document.getElementById('scan-status-text');
  wrap.style.display = 'block'; txt.style.display = 'block';
  bar.style.width = '0%';
  let pct = 0;
  const msgs = ['Fetching price data...', 'Computing indicators...', 'Running regime detection...', 'Scoring signals...'];
  let mi = 0;
  txt.textContent = msgs[0];
  _scanFakeProgress = setInterval(() => {
    pct = Math.min(pct + (Math.random() * 4 + 1), 88);
    bar.style.width = pct + '%';
    const newMi = Math.min(Math.floor(pct / 25), msgs.length - 1);
    if (newMi !== mi) { mi = newMi; txt.textContent = msgs[mi]; }
  }, 300);
}

function _stopFakeProgress(success) {
  if (_scanFakeProgress) { clearInterval(_scanFakeProgress); _scanFakeProgress = null; }
  const bar  = document.getElementById('scan-progress-bar');
  const wrap = document.getElementById('scan-progress-wrap');
  const txt  = document.getElementById('scan-status-text');
  bar.style.width = '100%';
  bar.style.background = success ? 'linear-gradient(90deg,#1a7f37,#3fb950)' : 'var(--red)';
  setTimeout(() => {
    wrap.style.display = 'none'; txt.style.display = 'none';
    bar.style.width = '0%';
    bar.style.background = 'linear-gradient(90deg,#1f6feb,#58a6ff)';
  }, 700);
}

async function runScanner() {
  const btn = document.getElementById('scan-btn');
  const spin = document.getElementById('scan-spin');
  const lbl  = document.getElementById('scan-label');

  btn.disabled = true; spin.style.display = 'block'; lbl.textContent = 'Scanning...';

  document.getElementById('scan-empty-state').style.display = 'none';
  document.getElementById('scan-error-state').style.display = 'none';
  document.getElementById('scan-results').style.display = 'none';
  _startFakeProgress();

  const wl       = document.getElementById('scan-watchlist').value;
  const interval = document.getElementById('scan-interval').value;
  const payload  = { interval, lookback: 60, max_concurrent: 8 };

  if (wl === 'custom') {
    const raw = document.getElementById('scan-custom').value;
    const tickers = raw.split(/[,\s]+/).map(t => t.trim().toUpperCase()).filter(Boolean);
    if (!tickers.length) {
      _stopFakeProgress(false);
      document.getElementById('scan-error-state').style.display = 'block';
      document.getElementById('scan-error-state').textContent = 'Warning: Enter at least one ticker in the custom field.';
      btn.disabled=false; spin.style.display='none'; lbl.textContent='▶ Run Scanner';
      return;
    }
    payload.tickers = tickers;
  } else {
    payload.watchlist = wl;
  }

  try {
    const res  = await fetch('/api/scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
    _stopFakeProgress(true);
    processScanResult(data);
    log(`Scanner: ${data.scanned} scanned, ${data.breakouts.length} breakouts, ${data.breakdowns.length} breakdowns, ${data.elapsed_ms}ms`, 'var(--green)');
  } catch(e) {
    _stopFakeProgress(false);
    const errEl = document.getElementById('scan-error-state');
    errEl.style.display = 'block';
    errEl.textContent = 'Warning: Scanner failed: ' + e.message;
    log('Scanner error: ' + e, 'var(--red)');
  } finally {
    btn.disabled = false; spin.style.display = 'none'; lbl.textContent = '▶ Run Scanner';
  }
}

function processScanResult(data) {
  _scanData    = data.all || [];
  _scanSortKey = 'score';
  _scanSortAsc = false;
  _scanFilter  = 'all';

  // Pills
  const info = `${data.scanned} scanned, ${data.succeeded} OK, ${data.elapsed_ms}ms`;
  document.getElementById('sc-info').textContent   = info;
  document.getElementById('sc-up').textContent     = `↑ ${data.breakouts.length} bullish`;
  document.getElementById('sc-down').textContent   = `↓ ${data.breakdowns.length} bearish`;
  document.getElementById('sc-neut').textContent   = `-> ${data.neutral.length} neutral`;
  document.getElementById('sc-all').classList.add('active');

  const errEl2 = document.getElementById('sc-errors');
  if (data.failed > 0) {
    errEl2.style.display = 'inline-block';
    errEl2.textContent = `Warning: ${data.failed} failed`;
  } else {
    errEl2.style.display = 'none';
  }

  // Top cards
  buildTopCards(data);

  document.getElementById('scan-results').style.display = 'block';
  renderScanTable();
}

function buildTopCards(data) {
  const wrap = document.getElementById('scan-top-cards');
  wrap.innerHTML = '';
  const topUp   = (data.breakouts   || []).slice(0, 4);
  const topDown = (data.breakdowns  || []).slice(0, 4);
  const all     = [...topUp, ...topDown];
  if (!all.length) { wrap.style.display = 'none'; return; }
  wrap.style.display = 'flex';

  all.forEach(r => {
    const isUp = r.score >= 0;
    const card = document.createElement('div');
    card.className = 'scan-top-card ' + (isUp ? 'up-card' : 'down-card');
    card.onclick = () => loadTicker(r.ticker);
    card.title = r.reasoning || '';
    const scoreStr = (r.score >= 0 ? '+' : '') + (r.score||0).toFixed(3);
    const regDisp  = (r.regime||'').replace(/_/g,' ');
    card.innerHTML = `
      <div class="stc-ticker" style="color:${isUp ? 'var(--green)' : 'var(--red)'}">
        ${isUp ? '↑' : '↓'} ${r.ticker}
      </div>
      <div class="stc-score"  style="color:${isUp ? 'var(--green)' : 'var(--red)'}">Score ${scoreStr}</div>
      <div class="stc-regime">${regDisp}</div>
      <div class="stc-price">$${typeof r.price === 'number' ? r.price.toFixed(2) : '-'} , RSI ${(r.rsi||0).toFixed(0)}</div>
    `;
    wrap.appendChild(card);
  });
}

function setScanSort(key) {
  if (_scanSortKey === key) { _scanSortAsc = !_scanSortAsc; }
  else { _scanSortKey = key; _scanSortAsc = key === 'ticker'; }
  document.querySelectorAll('.scan-sort-btn').forEach(b => b.classList.remove('active'));
  const id = { score:'srt-score', rsi:'srt-rsi', adx:'srt-adx',
                price_vs_52w:'srt-52w', confidence:'srt-conf', ticker:'srt-ticker' }[key];
  if (id) document.getElementById(id)?.classList.add('active');
  renderScanTable();
}

function filterScan(type) {
  _scanFilter = type;
  ['sc-up','sc-down','sc-neut','sc-all'].forEach(id => {
    document.getElementById(id)?.classList.remove('active');
  });
  const activeMap = { breakout:'sc-up', breakdown:'sc-down', neutral:'sc-neut', all:'sc-all' };
  document.getElementById(activeMap[type])?.classList.add('active');
  renderScanTable();
}

function renderScanTable() {
  let rows = [..._scanData];

  // Filter by direction
  if (_scanFilter === 'breakout')  rows = rows.filter(r => r.score >  0.10);
  if (_scanFilter === 'breakdown') rows = rows.filter(r => r.score < -0.10);
  if (_scanFilter === 'neutral')   rows = rows.filter(r => Math.abs(r.score) <= 0.10);

  // Search filter
  const q = (document.getElementById('scan-search')?.value || '').toUpperCase().trim();
  if (q) rows = rows.filter(r => r.ticker.includes(q));

  // Sort - supports nested trade_setup fields via adapter map
  rows.sort((a, b) => {
    const adapter = _tsSortAdapters[_scanSortKey];
    let av = adapter ? adapter(a) : a[_scanSortKey];
    let bv = adapter ? adapter(b) : b[_scanSortKey];
    if (typeof av === 'string') { av = av.toLowerCase(); bv = (bv||'').toLowerCase(); }
    if (av == null) av = -9999; if (bv == null) bv = -9999;
    return _scanSortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
  });

  const tbody = document.getElementById('scan-tbody');
  const table = document.getElementById('scan-table');

  document.getElementById('scan-row-count').textContent = `${rows.length} row${rows.length !== 1 ? 's' : ''}`;

  if (!rows.length) {
    table.style.display = 'none';
    return;
  }
  table.style.display = 'table';
  tbody.innerHTML = '';

  // Color helpers
  const sc  = s => s > 0.40 ? '#3fb950' : s > 0.20 ? '#56d364' : s > 0 ? '#7ee787'
                 : s < -0.40 ? '#f85149' : s < -0.20 ? '#ff7b72' : s < 0 ? '#ffa198' : '#8b949e';
  const rsiC = v => v > 70 ? 'var(--red)' : v < 30 ? 'var(--green)' : 'var(--text)';
  const pctC = (v, upGood=true) => {
    if (v == null) return 'var(--muted)';
    return upGood ? (v > 0 ? 'var(--green)' : v < 0 ? 'var(--red)' : 'var(--muted)')
                  : (v < 0 ? 'var(--green)' : v > 0 ? 'var(--red)' : 'var(--muted)');
  };
  const regC = {
    breakout_up:'var(--amber)', breakout_down:'var(--amber)',
    strong_uptrend:'var(--green)', weak_uptrend:'#56d364',
    strong_downtrend:'var(--red)', weak_downtrend:'#ff7b72',
    range_bound:'var(--blue)', choppy:'var(--muted)',
  };
  const dirLabel = {
    breakout_up:  '🚀 Breakout ↑', trending_up:  '📈 Trending ↑', bullish: '↑ Bullish',
    breakdown:    '💥 Breakdown ↓', trending_down:'📉 Trending ↓', bearish: '↓ Bearish',
    neutral: '-> Neutral', error: 'Warning: Error',
  };

  rows.forEach(r => {
    const score = r.score || 0;
    const barW  = Math.round(Math.min(Math.abs(score) * 100, 100));
    const barC  = score >= 0 ? 'rgba(63,185,80,.7)' : 'rgba(248,81,73,.7)';

    const ts = r.trade_setup || {};
    const tsValid = ts.valid === true;

    let tsSideHtml, tsEntryHtml, tsSlHtml, tsTp1Html, tsTp2Html, tsRrHtml, tsProbHtml;

    if (!tsValid) {
      // No entry - show reason as tooltip
      const reason = ts.reason || 'Criteria not met';
      const shortReason = reason.length > 28 ? reason.slice(0, 28) + '...' : reason;
      tsSideHtml  = `<span title="${reason}" style="color:var(--muted);font-size:10px;cursor:help;">⊘ No Entry</span>`;
      tsEntryHtml = `<span style="color:var(--muted);">-</span>`;
      tsSlHtml    = `<span style="color:var(--muted);">-</span>`;
      tsTp1Html   = `<span style="color:var(--muted);">-</span>`;
      tsTp2Html   = `<span style="color:var(--muted);">-</span>`;
      tsRrHtml    = `<span style="color:var(--muted);">-</span>`;
      tsProbHtml  = `<span style="color:var(--muted);" title="${shortReason}">-</span>`;
    } else {
      const side   = ts.side || 'long';
      const isLong = side === 'long';
      const sideColor = isLong ? 'var(--green)' : 'var(--red)';
      const sideIcon  = isLong ? '^ LONG' : 'v SHORT';

      const fmt = v => v != null ? '$' + Number(v).toFixed(2) : '-';
      const fmtPct = v => v != null ? (v >= 0 ? '+' : '') + Number(v).toFixed(1) + '%' : '-';

      // Use best R:R (tighter of ATR vs fixed stop) - same logic as backend sl_recommended
      const rr_atr = ts.rr_atr ?? 0;
      const rr_pct = ts.rr_pct ?? 0;
      const rr     = Math.max(rr_atr, rr_pct);                      // best R:R
      const rrC    = rr >= 2.0 ? 'var(--green)' : rr >= 1.3 ? 'var(--amber)' : 'var(--red)';
      const prob1  = ts.prob_tp1 != null ? Math.round(ts.prob_tp1 * 100) + '%' : '-';
      const probSlAtr = ts.prob_sl_atr != null ? Math.round(ts.prob_sl_atr * 100) + '%' : '-';

      // Show the recommended (tighter) SL in the table cell
      const recSL   = ts.sl_recommended === 'pct' ? ts.sl_pct : ts.sl_atr;
      const recDist = ts.sl_recommended === 'pct' ? ts.sl_pct_dist : ts.sl_atr_dist;
      const recType = ts.sl_recommended === 'pct' ? `Fixed ${ts.sl_pct_used}%` : `ATR ×${ts.atr_mult?.toFixed(1)}`;
      const capWarn = ts.atr_capped ? ' Warning:' : '';
      const slDistStr = recDist != null ? ` (${recDist.toFixed(1)}%)` : '';
      const tp1Dist = ts.tp1_dist != null ? ` (+${ts.tp1_dist.toFixed(1)}%)` : '';
      const tp2Dist = ts.tp2_dist != null ? ` (+${ts.tp2_dist.toFixed(1)}%)` : '';

      tsSideHtml  = `<span style="color:${sideColor}; font-weight:700; font-size:11px;">${sideIcon}</span>`;
      tsEntryHtml = `<span style="color:var(--blue); font-weight:600;">${fmt(ts.entry)}</span>`;
      tsSlHtml    = `<span style="color:var(--red);" title="${recType} stop | ATR: ${fmt(ts.sl_atr)} (${ts.sl_atr_dist?.toFixed(1)}%) | Fixed: ${fmt(ts.sl_pct)} (${ts.sl_pct_dist?.toFixed(1)}%)${ts.atr_capped?' | Warning: ATR was capped':''}">${fmt(recSL)}<span style="font-size:9px; color:var(--muted);">${slDistStr}${capWarn}</span></span>`;
      tsTp1Html   = `<span style="color:var(--green);" title="P75 target">${fmt(ts.tp1)}<span style="font-size:9px; color:var(--muted);">${tp1Dist}</span></span>`;
      tsTp2Html   = `<span style="color:rgba(63,185,80,.7);" title="P90 runner">${fmt(ts.tp2)}<span style="font-size:9px; color:var(--muted);">${tp2Dist}</span></span>`;
      tsRrHtml    = `<span style="color:${rrC}; font-weight:700;" title="Best R:R using tighter stop | ATR R:R: ${rr_atr.toFixed(1)}R | Fixed R:R: ${rr_pct.toFixed(1)}R">${rr.toFixed(1)}R</span>`;
      tsProbHtml  = `<span style="color:var(--green); font-weight:600;" title="P(TP1)=${prob1} | P(SL)=${probSlAtr}">${prob1}</span>`;
    }

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="font-weight:700; color:var(--blue); font-size:13px;">${r.ticker}</td>
      <td style="color:var(--text);">$${typeof r.price==='number' ? r.price.toFixed(2) : '-'}</td>
      <td style="white-space:nowrap;">
        <span style="display:inline-block;width:36px;height:4px;background:var(--surface2);border-radius:2px;vertical-align:middle;margin-right:6px;overflow:hidden;">
          <span style="display:block;height:100%;width:${barW}%;background:${barC};border-radius:2px;"></span>
        </span>
        <span style="font-weight:700; color:${sc(score)};">${score>=0?'+':''}${score.toFixed(3)}</span>
      </td>
      <td style="color:${score>0.1?'var(--green)':score<-0.1?'var(--red)':'var(--muted)'}; font-size:11px;">
        ${dirLabel[r.direction] || r.direction}</td>
      <td style="color:${regC[r.regime]||'var(--muted)'}; font-size:11px; text-transform:uppercase; letter-spacing:.3px;">
        ${(r.regime||'-').replace(/_/g,' ')}</td>
      <td style="font-size:11px; color:var(--muted);">${r.signal_label||'-'}</td>
      <td style="color:${r.confidence>0.6?'var(--green)':r.confidence>0.35?'var(--text)':'var(--muted)'};">
        ${r.confidence!=null ? Math.round(r.confidence*100)+'%' : '-'}</td>
      <td style="color:${rsiC(r.rsi)}; font-weight:${(r.rsi>70||r.rsi<30)?700:400};">
        ${r.rsi!=null ? r.rsi.toFixed(1) : '-'}</td>
      <td style="color:${r.adx>25?'var(--amber)':'var(--muted)'}; font-weight:${r.adx>25?700:400};">
        ${r.adx!=null ? r.adx.toFixed(1) : '-'}</td>
      <td style="color:${pctC(r.bb_position, false)};">
        ${r.bb_position!=null ? (r.bb_position>=0?'+':'')+r.bb_position.toFixed(2) : '-'}</td>
      <td style="color:${pctC(r.obv_slope)};">
        ${r.obv_slope!=null ? (r.obv_slope>=0?'+':'')+r.obv_slope.toFixed(2) : '-'}</td>
      <td style="color:${r.price_vs_52w>=-2?'var(--green)':r.price_vs_52w>-10?'var(--text)':'var(--red)'};">
        ${r.price_vs_52w!=null ? (r.price_vs_52w>=0?'+':'')+r.price_vs_52w.toFixed(1)+'%' : '-'}</td>
      <td style="color:${pctC(r.ema200_dist, false)};">
        ${r.ema200_dist!=null ? (r.ema200_dist>=0?'+':'')+r.ema200_dist.toFixed(1)+'%' : '-'}</td>
      <td style="color:${r.hurst>0.6?'var(--green)':r.hurst<0.4?'var(--amber)':'var(--muted)'};">
        ${r.hurst!=null ? r.hurst.toFixed(2) : '-'}</td>
      <!-- Trade setup columns -->
      <td style="border-left:2px solid rgba(88,166,255,.15);">${tsSideHtml}</td>
      <td>${tsEntryHtml}</td>
      <td>${tsSlHtml}</td>
      <td>${tsTp1Html}</td>
      <td>${tsTp2Html}</td>
      <td>${tsRrHtml}</td>
      <td>${tsProbHtml}</td>
      <td>
        <button onclick="loadTicker('${r.ticker}')"
          style="background:rgba(88,166,255,.1); border:1px solid rgba(88,166,255,.3); color:var(--blue);
                 border-radius:5px; padding:4px 10px; font-size:11px; cursor:pointer; font-weight:600;
                 transition:background .12s;"
          onmouseover="this.style.background='rgba(88,166,255,.2)'"
          onmouseout="this.style.background='rgba(88,166,255,.1)'">
          Load ↗
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

const _tsSortAdapters = {
  ts_side:    r => (r.trade_setup?.side    || 'none'),
  ts_entry:   r => (r.trade_setup?.entry   ?? -9999),
  ts_sl:      r => {
    const ts = r.trade_setup;
    if (!ts) return -9999;
    return ts.sl_recommended === 'pct' ? (ts.sl_pct ?? -9999) : (ts.sl_atr ?? -9999);
  },
  ts_tp1:     r => (r.trade_setup?.tp1     ?? -9999),
  ts_tp2:     r => (r.trade_setup?.tp2     ?? -9999),
  ts_rr:      r => Math.max(r.trade_setup?.rr_atr ?? -9999, r.trade_setup?.rr_pct ?? -9999),
  ts_prob_tp1:r => (r.trade_setup?.prob_tp1 ?? -9999),
};

async function loadTicker(ticker) {
  // Sync ticker + interval from scanner
  const scanInterval = document.getElementById('scan-interval').value;
  document.getElementById('s-ticker').value   = ticker;
  document.getElementById('s-interval').value = scanInterval;
  // Also update the quick-ticker in the header so it stays consistent
  const qt = document.getElementById('quick-ticker');
  const qi = document.getElementById('quick-interval');
  if (qt) qt.value = ticker;
  if (qi) qi.value = scanInterval;

  // Also sync MC settings from currentConfig so the sidebar analysis
  // runs with the same parameters as the user has configured.
  if (currentConfig.n_sim)        document.getElementById('s-nsim').value    = currentConfig.n_sim;
  if (currentConfig.n_forward)    document.getElementById('s-nfwd').value    = currentConfig.n_forward;
  if (currentConfig.lookback)     document.getElementById('s-lookback').value= currentConfig.lookback;
  if (currentConfig.mc_model)     document.getElementById('s-model').value   = currentConfig.mc_model || 'microstructure';

  log(`Loading ${ticker} @ ${scanInterval} (n_sim=${currentConfig.n_sim||'?'}, fwd=${currentConfig.n_forward||'?'})...`);
  await applySettings();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
