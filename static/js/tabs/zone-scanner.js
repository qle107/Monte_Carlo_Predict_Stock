// Zone + EMA scanner table and watchlist UI.

let _zoneData     = [];
let _zoneSortKey  = 'setup_score';
let _zoneSortAsc  = false;
let _zoneFilter   = 'all';
let _zoneFakeProg = null;

// Custom ticker toggle for zone watchlist
document.getElementById('zone-watchlist').addEventListener('change', function() {
  document.getElementById('zone-custom').style.display =
    this.value === 'custom' ? 'inline-block' : 'none';
});

function _startZoneProgress() {
  const wrap = document.getElementById('zone-progress-wrap');
  const bar  = document.getElementById('zone-progress-bar');
  wrap.style.display = 'block';
  bar.style.width = '20%';
  let pct = 20;
  _zoneFakeProg = setInterval(() => {
    pct = Math.min(pct + (Math.random() * 3 + 0.5), 85);
    bar.style.width = pct + '%';
  }, 400);
}
function _stopZoneProgress(success) {
  if (_zoneFakeProg) { clearInterval(_zoneFakeProg); _zoneFakeProg = null; }
  const wrap = document.getElementById('zone-progress-wrap');
  const bar  = document.getElementById('zone-progress-bar');
  bar.style.width = '100%';
  bar.style.background = success
    ? 'linear-gradient(90deg,#4a1ea5,#bc8cff)'
    : 'var(--red)';
  setTimeout(() => {
    wrap.style.display = 'none';
    bar.style.width = '20%';
    bar.style.background = 'linear-gradient(90deg,#6e40c9,#bc8cff)';
  }, 700);
}

async function runZoneScanner() {
  const btn   = document.getElementById('zone-scan-btn');
  const spin  = document.getElementById('zone-spin');
  const lbl   = document.getElementById('zone-scan-label');

  btn.disabled = true;
  spin.style.display = 'block';
  lbl.textContent = 'Scanning zones...';

  document.getElementById('zone-empty-state').style.display = 'none';
  document.getElementById('zone-error-state').style.display = 'none';
  document.getElementById('zone-results').style.display     = 'none';
  _startZoneProgress();

  const wl       = document.getElementById('zone-watchlist').value;
  const interval = document.getElementById('zone-interval').value;
  const payload  = { interval, lookback: 120, max_concurrent: 8 };

  if (wl === 'custom') {
    const raw     = document.getElementById('zone-custom').value;
    const tickers = raw.split(/[,\s]+/).map(t => t.trim().toUpperCase()).filter(Boolean);
    if (!tickers.length) {
      _stopZoneProgress(false);
      const errEl = document.getElementById('zone-error-state');
      errEl.style.display = 'block';
      errEl.textContent   = 'Warning: Enter at least one ticker in the custom field.';
      btn.disabled = false; spin.style.display = 'none'; lbl.textContent = 'Run Zone Scan';
      return;
    }
    payload.tickers = tickers;
  } else {
    payload.watchlist = wl;
  }

  try {
    const res  = await fetch('/api/zone-scan', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
    _stopZoneProgress(true);
    processZoneResult(data);
    const meta = data.meta || {};
    log(
      `Zone scan: ${meta.scanned||'?'} scanned, ${(data.longs||[]).length} longs, ${(data.shorts||[]).length} shorts, ${meta.elapsed_ms||'?'}ms`,
      'var(--purple)'
    );
  } catch(e) {
    _stopZoneProgress(false);
    const errEl = document.getElementById('zone-error-state');
    errEl.style.display = 'block';
    errEl.textContent   = 'Warning: Zone scan failed: ' + e.message;
    log('Zone scan error: ' + e, 'var(--red)');
  } finally {
    btn.disabled = false;
    spin.style.display = 'none';
    lbl.textContent = 'Run Zone Scan';
  }
}

function processZoneResult(data) {
  _zoneData    = data.all || [];
  _zoneSortKey = 'setup_score';
  _zoneSortAsc = false;
  _zoneFilter  = 'all';

  const longs  = data.longs  || [];
  const shorts = data.shorts || [];
  const meta   = data.meta   || {};

  // Summary pills
  document.getElementById('zc-info').textContent  =
    `${meta.scanned||'?'} scanned, ${meta.succeeded||'?'} OK, ${meta.elapsed_ms||'?'}ms`;
  document.getElementById('zc-long').textContent  = `^ ${longs.length} longs`;
  document.getElementById('zc-short').textContent = `v ${shorts.length} shorts`;

  // Tab badge
  const total = longs.length + shorts.length;
  document.getElementById('tab-zone-badge').textContent = total > 0 ? total : '-';

  document.getElementById('zone-results').style.display = 'block';
  renderZoneTable();
}

function _emaStackPill(stack) {
  const map = {
    bull_stack:  ['ema-stack-pill ema-bull',   ' Bull Stack'],
    above_200:   ['ema-stack-pill ema-above200','↑ Above 200'],
    bear_stack:  ['ema-stack-pill ema-bear',   ' Bear Stack'],
    below_200:   ['ema-stack-pill ema-below200','↓ Below 200'],
    mixed:       ['ema-stack-pill ema-mixed',  '~ Mixed'],
  };
  const [cls, label] = map[stack] || ['ema-stack-pill ema-mixed', stack || '-'];
  return `<span class="${cls}">${label}</span>`;
}

function _setupTypePill(type) {
  const map = {
    demand_bounce:  ['zone-type-pill zone-demand-bounce',  '↑ Demand Bounce'],
    demand_support: ['zone-type-pill zone-demand-support', '↑ Demand Support'],
    supply_break:   ['zone-type-pill zone-supply-break',   '↑ Supply Break'],
    supply_bounce:  ['zone-type-pill zone-supply-bounce',  '↓ Supply Reject'],
    demand_break:   ['zone-type-pill zone-demand-break',   '↓ Demand Break'],
    none:           ['zone-type-pill zone-none',           '- None'],
  };
  const [cls, label] = map[type] || ['zone-type-pill zone-none', type || '-'];
  return `<span class="${cls}">${label}</span>`;
}

// (zone sort uses direct field access - no nested adapters needed)

function setZoneSort(key) {
  if (_zoneSortKey === key) { _zoneSortAsc = !_zoneSortAsc; }
  else { _zoneSortKey = key; _zoneSortAsc = key === 'ticker'; }

  document.querySelectorAll('[id^="zsrt-"]').forEach(b => b.classList.remove('active'));
  const idMap = {
    setup_score:  'zsrt-score',
    rsi:          'zsrt-rsi',
    adx:          'zsrt-adx',
    dist_ema20:   'zsrt-e20',
    dist_ema200:  'zsrt-e200',
    ticker:       'zsrt-ticker',
  };
  if (idMap[key]) document.getElementById(idMap[key])?.classList.add('active');
  renderZoneTable();
}

function filterZone(type) {
  _zoneFilter = type;
  ['zc-long','zc-short','zc-all'].forEach(id =>
    document.getElementById(id)?.classList.remove('active')
  );
  const activeMap = { long:'zc-long', short:'zc-short', all:'zc-all' };
  document.getElementById(activeMap[type])?.classList.add('active');
  renderZoneTable();
}

function _emaCellHtml(emaVal, price, crossSignal, label) {
  if (!emaVal) return '<td>-</td>';
  const dist    = price ? ((price - emaVal) / price * 100) : 0;
  const abv     = price > emaVal;
  const distStr = (dist >= 0 ? '+' : '') + dist.toFixed(2) + '%';
  const distC   = abv ? 'var(--green)' : 'var(--red)';

  let crossHtml = '';
  if (crossSignal === 'cross_up') {
    crossHtml = `<span style="color:var(--green);font-weight:700;font-size:11px;margin-left:4px;"
      title="${label} crossed UP (Golden cross)">↑</span>`;
  } else if (crossSignal === 'cross_down') {
    crossHtml = `<span style="color:var(--red);font-weight:700;font-size:11px;margin-left:4px;"
      title="${label} crossed DOWN (Death cross)">↓</span>`;
  }

  return `<td style="border-left:2px solid rgba(88,166,255,.1);">
    <div style="font-weight:600;font-size:12px;color:var(--text);">$${Number(emaVal).toFixed(2)}</div>
    <div style="font-size:10px;color:${distC};">${distStr}${crossHtml}</div>
  </td>`;
}

function _zoneCellHtml(zone, kind) {
  if (!zone) return `<td style="color:var(--muted);font-size:11px;">-</td>`;
  const isStrong  = zone.label && zone.label.startsWith('Strong');
  const isSupport = kind === 'demand';
  const baseColor = isSupport ? 'var(--green)' : 'var(--red)';
  const bgColor   = isSupport ? 'rgba(63,185,80,' : 'rgba(248,81,73,';
  const strPct    = Math.round((zone.strength ?? 0) * 100);
  const freshTag  = zone.fresh
    ? `<span style="font-size:8px;padding:1px 4px;border-radius:3px;
         background:${bgColor}.15);color:${baseColor};margin-left:3px;">FRESH</span>` : '';
  const strongTag = isStrong
    ? `<span style="font-size:8px;padding:1px 4px;border-radius:3px;
         background:${bgColor}.2);color:${baseColor};font-weight:700;margin-left:3px;"></span>` : '';
  const distStr = zone.dist_pct != null
    ? `<span style="font-size:9px;color:var(--muted);margin-left:2px;">(${zone.dist_pct.toFixed(1)}%)</span>`
    : '';

  return `<td style="border-left:2px solid ${bgColor}.18);">
    <div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.3px;margin-bottom:1px;">
      ${zone.label}${strongTag}${freshTag}
    </div>
    <div style="font-weight:700;color:${baseColor};font-size:12px;">
      $${Number(zone.level).toFixed(2)}${distStr}
    </div>
    <div style="font-size:9px;color:var(--muted);">str ${strPct}%, ${zone.touches} touch${zone.touches !== 1 ? 'es' : ''}</div>
  </td>`;
}

function renderZoneTable() {
  let rows = [..._zoneData];

  if (_zoneFilter === 'long')  rows = rows.filter(r => r.side === 'long');
  if (_zoneFilter === 'short') rows = rows.filter(r => r.side === 'short');

  const q = (document.getElementById('zone-search')?.value || '').toUpperCase().trim();
  if (q) rows = rows.filter(r => (r.ticker || '').includes(q));

  rows.sort((a, b) => {
    let av = a[_zoneSortKey] ?? -9999;
    let bv = b[_zoneSortKey] ?? -9999;
    if (typeof av === 'string') { av = av.toLowerCase(); bv = (bv || '').toLowerCase(); }
    return _zoneSortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
  });

  const tbody = document.getElementById('zone-tbody');
  const table = document.getElementById('zone-scan-table');
  document.getElementById('zone-row-count').textContent =
    `${rows.length} row${rows.length !== 1 ? 's' : ''}`;

  if (!rows.length) { table.style.display = 'none'; return; }
  table.style.display = 'table';
  tbody.innerHTML = '';

  const f2   = v => v != null ? '$' + Number(v).toFixed(2) : '-';
  const rsiC = v => v > 70 ? 'var(--red)' : v < 30 ? 'var(--green)' : 'var(--text)';
  const adxC = v => v > 25 ? 'var(--amber)' : 'var(--muted)';

  rows.forEach(r => {
    const score  = r.setup_score ?? 0;
    const scoreW = Math.round(Math.min(score * 100, 100));

    // EMA cross signals
    const c2050  = r.cross_20_50  || 'none';
    const c50200 = r.cross_50_200 || 'none';
    const c20200 = r.cross_20_200 || 'none';

    // Pick the most significant active cross for each EMA (20->50 for EMA20, 50->200 for EMA50, etc.)
    const cross20  = c2050  !== 'none' ? c2050  : (c20200 !== 'none' ? c20200 : 'none');
    const cross50  = c2050  !== 'none' ? c2050  : (c50200 !== 'none' ? c50200 : 'none');
    const cross200 = c50200 !== 'none' ? c50200 : (c20200 !== 'none' ? c20200 : 'none');

    // Demand and supply zone arrays
    const dz = r.demand_zones || [];
    const sz = r.supply_zones || [];

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="font-weight:700;color:var(--purple);font-size:13px;cursor:pointer;"
          onclick="loadZoneTicker('${r.ticker}')">${r.ticker}</td>
      <td style="color:var(--text);font-weight:600;">$${typeof r.price==='number'?r.price.toFixed(2):'-'}</td>
      <td style="white-space:nowrap;">
        <div style="display:flex;align-items:center;gap:5px;">
          <span style="display:inline-block;width:${scoreW}px;max-width:48px;height:4px;border-radius:2px;
            background:linear-gradient(90deg,var(--blue),var(--purple));"></span>
          <span style="font-weight:700;color:${score>0.6?'var(--green)':score>0.35?'var(--amber)':'var(--muted)'};">
            ${score.toFixed(2)}
          </span>
        </div>
      </td>
      <td>${_setupTypePill(r.setup_type)}</td>
      <td style="color:${rsiC(r.rsi)};font-weight:${(r.rsi>70||r.rsi<30)?700:400};">
        ${r.rsi!=null?r.rsi.toFixed(1):'-'}</td>
      <td style="color:${adxC(r.adx)};font-weight:${r.adx>25?700:400};">
        ${r.adx!=null?r.adx.toFixed(1):'-'}</td>
      ${_emaCellHtml(r.ema20,  r.price, cross20,  'EMA20/50')}
      ${_emaCellHtml(r.ema50,  r.price, cross50,  'EMA50/200')}
      ${_emaCellHtml(r.ema200, r.price, cross200, 'EMA200')}
      ${_zoneCellHtml(dz[0] || null, 'demand')}
      ${_zoneCellHtml(dz[1] || null, 'demand')}
      ${_zoneCellHtml(sz[0] || null, 'supply')}
      ${_zoneCellHtml(sz[1] || null, 'supply')}
      <td>
        <button onclick="loadZoneTicker('${r.ticker}')"
          style="background:rgba(188,140,255,.1);border:1px solid rgba(188,140,255,.35);color:var(--purple);
                 border-radius:5px;padding:4px 10px;font-size:11px;cursor:pointer;font-weight:600;transition:background .12s;"
          onmouseover="this.style.background='rgba(188,140,255,.2)'"
          onmouseout="this.style.background='rgba(188,140,255,.1)'">
          Load ↗
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadZoneTicker(ticker) {
  const interval = document.getElementById('zone-interval').value;
  document.getElementById('s-ticker').value   = ticker;
  document.getElementById('s-interval').value = interval;
  // Keep quick-ticker header in sync
  const qt = document.getElementById('quick-ticker');
  const qi = document.getElementById('quick-interval');
  if (qt) qt.value = ticker;
  if (qi) qi.value = interval;

  // Sync MC settings
  if (currentConfig.n_sim)     document.getElementById('s-nsim').value    = currentConfig.n_sim;
  if (currentConfig.n_forward) document.getElementById('s-nfwd').value    = currentConfig.n_forward;
  if (currentConfig.lookback)  document.getElementById('s-lookback').value= currentConfig.lookback;
  if (currentConfig.mc_model)  document.getElementById('s-model').value   = currentConfig.mc_model;

  log(`Loading ${ticker} @ ${interval} from Zone Scanner...`, 'var(--purple)');
  await applySettings();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
