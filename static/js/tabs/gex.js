// GEX heatmap tab: net gamma exposure per strike x expiration.

let _gexLoading = false;
let _gexLastTicker = '';

function gexOnOpen(globalTicker) {
  const input = document.getElementById('gex-ticker');
  if (input && !input.value && globalTicker) input.value = globalTicker;
  const tkr = (input && input.value || '').toUpperCase().trim();
  if (tkr && tkr !== _gexLastTicker && !_gexLoading) runGexHeatmap();
}

function _gexCellColor(v, maxAbs) {
  if (v == null || !maxAbs) return 'transparent';
  const a = 0.10 + 0.80 * Math.sqrt(Math.min(Math.abs(v) / maxAbs, 1));
  return v >= 0 ? `rgba(46,160,67,${a.toFixed(3)})` : `rgba(218,54,51,${a.toFixed(3)})`;
}

function _gexExpLabel(iso) {
  const d = new Date(iso + 'T00:00:00');
  return d.toLocaleDateString('en-US', { month: 'short', day: '2-digit' }).toUpperCase();
}

async function runGexHeatmap() {
  if (_gexLoading) return;
  const input = document.getElementById('gex-ticker');
  const tkr = (input && input.value || '').toUpperCase().trim();
  const errEl = document.getElementById('gex-error');
  const emptyEl = document.getElementById('gex-empty');
  const loadEl = document.getElementById('gex-loading');
  const resEl = document.getElementById('gex-results');
  if (!tkr) {
    if (errEl) { errEl.style.display = 'block'; errEl.textContent = 'Enter a ticker.'; }
    return;
  }
  _gexLoading = true;
  const btn = document.getElementById('gex-run-btn');
  const spin = document.getElementById('gex-spin');
  if (btn) btn.disabled = true;
  if (spin) spin.style.display = 'inline-block';
  if (errEl) errEl.style.display = 'none';
  if (emptyEl) emptyEl.style.display = 'none';
  if (resEl) resEl.style.display = 'none';
  if (loadEl) loadEl.style.display = 'block';
  try {
    const r = await fetch(`/api/options/gex_heatmap?ticker=${encodeURIComponent(tkr)}`);
    if (!r.ok) {
      let detail = `HTTP ${r.status}`;
      try { detail = (await r.json()).detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    const data = await r.json();
    _gexLastTicker = tkr;
    _renderGexHeatmap(data);
    if (resEl) resEl.style.display = 'block';
  } catch (e) {
    if (errEl) { errEl.style.display = 'block'; errEl.textContent = 'Error: ' + e.message; }
  } finally {
    _gexLoading = false;
    if (btn) btn.disabled = false;
    if (spin) spin.style.display = 'none';
    if (loadEl) loadEl.style.display = 'none';
  }
}

function _renderGexHeatmap(data) {
  // Header stats
  const spot = data.spot || 0;
  const setTxt = (id, txt, color) => {
    const el = document.getElementById(id);
    if (el) { el.textContent = txt; if (color) el.style.color = color; }
  };
  setTxt('gex-stat-spot', '$' + fmt(spot));
  setTxt('gex-stat-flip', '$' + fmt(data.gamma_flip));
  setTxt('gex-stat-net', _fmtCompact(data.net_gex), colorOf(data.net_gex));
  if (data.max_pos) setTxt('gex-stat-maxpos', `${_fmtCompact(data.max_pos.gex)} @ ${fmt(data.max_pos.strike)}`, 'var(--green)');
  if (data.max_neg) setTxt('gex-stat-maxneg', `${_fmtCompact(data.max_neg.gex)} @ ${fmt(data.max_neg.strike)}`, '#f85149');
  const az = data.accel_zone;
  setTxt('gex-stat-accel', az ? `$${fmt(az.low)} – $${fmt(az.high)}` : 'none');
  setTxt('gex-heatmap-title', `${data.ticker} GEX Heatmap`);

  // Max |GEX| for color scaling; locate extreme cells for highlight.
  let maxAbs = 0;
  data.rows.forEach(row => row.cells.forEach(v => { if (v != null) maxAbs = Math.max(maxAbs, Math.abs(v)); }));

  // Strike row closest to spot gets a marker.
  let spotRow = -1, spotDist = Infinity;
  data.rows.forEach((row, i) => {
    const d = Math.abs(row.strike - spot);
    if (d < spotDist) { spotDist = d; spotRow = i; }
  });

  // Price anchor: the cell with the highest ABSOLUTE net GEX gets the
  // ocean-blue box (most impactful level, per the GEX guide convention).
  const OCEAN = '#0077be';
  let anchorI = -1, anchorJ = -1, anchorV = 0;
  data.rows.forEach((row, i) => row.cells.forEach((v, j) => {
    if (v != null && Math.abs(v) > anchorV) { anchorV = Math.abs(v); anchorI = i; anchorJ = j; }
  }));

  const thStyle = 'position:sticky;top:0;background:#0d1117;padding:6px 8px;font-size:10px;color:var(--muted);' +
                  'border-bottom:1px solid rgba(255,255,255,.14);z-index:2;text-align:center;white-space:nowrap;';
  let html = '<table style="width:100%;border-collapse:collapse;font-size:11px;">';
  html += `<thead><tr><th style="${thStyle}text-align:right;position:sticky;left:0;z-index:3;width:1%;">STRIKE</th>`;
  data.expiries.forEach(e => { html += `<th style="${thStyle}" title="${e}">${_gexExpLabel(e)}</th>`; });
  html += '</tr></thead><tbody>';

  data.rows.forEach((row, i) => {
    const isSpot = i === spotRow;
    const strikeStyle = 'text-align:right;padding:3px 8px;font-weight:600;position:sticky;left:0;background:#0d1117;z-index:1;white-space:nowrap;width:1%;' +
      (isSpot ? 'color:var(--amber);border:1px solid var(--amber);border-radius:3px;' : 'color:var(--muted);');
    html += `<tr><td style="${strikeStyle}" ${isSpot ? `title="spot $${fmt(spot)}"` : ''}>${fmt(row.strike, row.strike % 1 ? 1 : 0)}</td>`;
    row.cells.forEach((v, j) => {
      if (v == null) {
        html += '<td style="padding:3px 6px;"></td>';
      } else {
        const isAnchor = i === anchorI && j === anchorJ;
        const bg = isAnchor ? OCEAN : _gexCellColor(v, maxAbs);
        const txtColor = isAnchor ? '#fff' : (Math.abs(v) / maxAbs > 0.04 ? '#e6edf3' : 'var(--muted)');
        html += `<td title="${data.expiries[j]}  strike ${row.strike}  GEX ${_fmtCompact(v)}${isAnchor ? '  (highest absolute GEX - most impactful level)' : ''}"` +
                ` style="padding:3px 6px;text-align:center;background:${bg};color:${txtColor};` +
                `font-weight:${isAnchor || Math.abs(v) === maxAbs ? 700 : 400};white-space:nowrap;">${_fmtCompact(v)}</td>`;
      }
    });
    html += '</tr>';
  });
  html += '</tbody></table>';

  const wrap = document.getElementById('gex-table-wrap');
  if (wrap) wrap.innerHTML = html;
}
