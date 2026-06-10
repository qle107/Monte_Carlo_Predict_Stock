// Macro strip + tab cards; background fetch on DOMContentLoaded.

let _macroLoaded     = false;
let _macroLoading    = false;
let _macroData       = null;   // last fetched payload from /api/macro

// Colour helpers
const _impactColor = { bullish: 'var(--green)', bearish: 'var(--red)', neutral: 'var(--muted)' };
const _arrowColor  = { '↑': 'var(--green)', '↓': 'var(--red)', '->': 'var(--muted)' };

/**
 * Fetch macro indicators from the server and render both locations.
 * @param {boolean} force  Pass true to bypass server-side 4h cache.
 */
async function fetchMacroData(force = false) {
  if (_macroLoading) return;
  _macroLoading = true;

  // Disable refresh buttons while loading
  ['macro-refresh-btn', 'macro-tab-refresh-btn'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = true;
  });

  try {
    const url  = force ? '/api/macro?force=1' : '/api/macro';
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    _macroData   = await resp.json();
    _macroLoaded = true;
    renderMacroStrip(_macroData);
    renderMacroCards(_macroData);
    try { _updateSignalFromMacro(_macroData); } catch(_) {}
  } catch (err) {
    console.warn('[macro] fetch failed:', err);
    document.getElementById('macro-chips').innerHTML =
      '<span style="font-size:11px;color:var(--muted);padding:0 12px;">Macro data unavailable</span>';
  } finally {
    _macroLoading = false;
    ['macro-refresh-btn', 'macro-tab-refresh-btn'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.disabled = false;
    });
  }
}

/**
 * Render the compact horizontal macro strip (#macro-chips).
 */
function renderMacroStrip(data) {
  const container = document.getElementById('macro-chips');
  if (!container || !data?.indicators) return;

  container.innerHTML = data.indicators.map(ind => {
    const hasVal = ind.current !== null && ind.current !== undefined;
    const valStr = hasVal ? `${ind.current}${ind.unit === '%' || ind.unit.includes('%') ? '%' : ''}` : 'N/A';
    const color  = hasVal ? (_impactColor[ind.impact] || 'var(--muted)') : 'var(--muted)';
    const arrow  = hasVal ? ind.arrow : '->';
    const arrowC = _arrowColor[arrow] || 'var(--muted)';

    return `
      <div class="macro-chip" title="${ind.full_name}: ${ind.description}">
        <div class="macro-chip-name">${ind.name}</div>
        <div class="macro-chip-val" style="color:${color}">
          ${hasVal
            ? `<span>${valStr}</span><span class="mc-arrow" style="color:${arrowC}">${arrow}</span>`
            : `<span class="mc-na">N/A</span>`
          }
        </div>
        <div class="macro-badge ${ind.impact}">${ind.impact}</div>
      </div>`;
  }).join('');
}

/**
 * Render the larger macro indicator cards in the News & Macro tab.
 */
function renderMacroCards(data) {
  const grid = document.getElementById('macro-cards-grid');
  if (!grid || !data?.indicators) return;

  grid.innerHTML = data.indicators.map(ind => {
    const hasVal  = ind.current !== null && ind.current !== undefined;
    const unitStr = (ind.unit === '%' || ind.unit.includes('%')) ? '%' : (ind.unit ? ` ${ind.unit}` : '');
    const valStr  = hasVal ? `${ind.current}${unitStr}` : 'N/A';
    const prevStr = (ind.previous !== null && ind.previous !== undefined)
                  ? `Prev: ${ind.previous}${unitStr}`
                  : 'Prev: N/A';
    const color   = hasVal ? (_impactColor[ind.impact] || 'var(--muted)') : 'var(--muted)';
    const arrow   = hasVal ? ind.arrow : '->';
    const arrowC  = _arrowColor[arrow] || 'var(--muted)';

    return `
      <div class="macro-card">
        <div class="macro-card-name">${ind.full_name || ind.name}</div>
        <div class="macro-card-val" style="color:${color}">
          <span>${valStr}</span>
          <span style="color:${arrowC};font-size:16px;">${arrow}</span>
          <span class="macro-badge ${ind.impact}" style="margin-left:2px;">${ind.impact}</span>
        </div>
        <div class="macro-card-prev" style="color:var(--muted);">${prevStr}</div>
        <div class="macro-card-desc">${ind.description}</div>
      </div>`;
  }).join('');

  // Show FRED hint if data is partial
  const hint = document.getElementById('macro-fred-hint');
  if (hint) hint.style.display = data.fred_active ? 'none' : 'block';

  // Show data note
  const note = document.getElementById('macro-data-note');
  if (note && data.data_note) note.textContent = data.data_note;
}


// Runs after DOMContentLoaded so #macro-chips is ready.
document.addEventListener('DOMContentLoaded', () => {
  // Small delay to let the WS connect first, then fetch macro in background.
  setTimeout(() => fetchMacroData(false), 800);
});
