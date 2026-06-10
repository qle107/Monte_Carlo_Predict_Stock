// News & Macro tab feed: fetch, filter, render.

let _newsLoaded      = false;
// _newsLoading already declared earlier in the file - do not re-declare
let _newsLastTicker  = '';
let _newsData        = [];   // raw articles array
let _newsFilter      = 'All'; // current filter selection

/**
 * Fetch news from the server and render the list.
 * @param {boolean} force  Bypass any in-flight guard and re-fetch.
 */
async function fetchNewsData(force = false) {
  if (_newsLoading && !force) return;
  _newsLoading = true;

  const ticker = (currentConfig.ticker || '').toUpperCase().trim();
  const btn    = document.getElementById('news-refresh-btn');
  if (btn) btn.disabled = true;

  // Update ticker label
  const lbl = document.getElementById('news-ticker-label');
  if (lbl) lbl.textContent = ticker || 'Market';

  // Show loading state
  const list = document.getElementById('news-feed-list');
  if (list) list.innerHTML = '<div class="news-empty">Loading news...</div>';

  try {
    const url  = `/api/news?ticker=${encodeURIComponent(ticker)}&limit=30`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    _newsData       = data.articles || [];
    _newsLastTicker = ticker;
    _newsLoaded     = true;

    // VIX badge
    if (data.vix !== null && data.vix !== undefined) {
      const vixBadge = document.getElementById('news-vix-badge');
      const vixVal   = document.getElementById('news-vix-val');
      if (vixBadge) vixBadge.style.display = 'inline-flex';
      if (vixVal) {
        const vixColor = data.vix > 30 ? 'var(--red)' : data.vix > 20 ? 'var(--amber)' : 'var(--green)';
        vixVal.textContent = data.vix.toFixed(1);
        vixVal.style.color = vixColor;
      }
    }

    // Fetched-at timestamp
    const ts = document.getElementById('news-fetched-at');
    if (ts && data.fetched_at) {
      const d = new Date(data.fetched_at);
      ts.textContent = `Last updated: ${d.toLocaleTimeString()}`;
    }

    renderNewsFeed();

  } catch (err) {
    console.warn('[news] fetch failed:', err);
    if (list) list.innerHTML =
      `<div class="news-empty" style="color:var(--red);">Failed to load news. ${err.message}</div>`;
  } finally {
    _newsLoading = false;
    if (btn) btn.disabled = false;
  }
}

/**
 * Set the active news filter and re-render the list.
 * @param {string} filter  'All' | 'Company' | 'Macro' | 'Sector' | 'General' | 'Positive' | 'Negative'
 */
function setNewsFilter(filter) {
  _newsFilter = filter;

  // Update button states
  const filterMap = {
    'All':     'nf-all',
    'Company': 'nf-company',
    'Macro':   'nf-macro',
    'Sector':  'nf-sector',
    'General': 'nf-general',
    'Positive':'nf-positive',
    'Negative':'nf-negative',
  };
  Object.entries(filterMap).forEach(([key, id]) => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('active', key === filter);
  });

  renderNewsFeed();
}

/**
 * Render the news feed list, applying the current filter.
 */
function renderNewsFeed() {
  const list = document.getElementById('news-feed-list');
  if (!list) return;

  // Apply filter
  const filtered = _newsData.filter(a => {
    if (_newsFilter === 'All')      return true;
    if (_newsFilter === 'Positive') return a.sentiment === 'Positive';
    if (_newsFilter === 'Negative') return a.sentiment === 'Negative';
    return a.category === _newsFilter;
  });

  // Update count badge
  const badge = document.getElementById('news-count-badge');
  if (badge) {
    badge.textContent = filtered.length
      ? `${filtered.length} article${filtered.length !== 1 ? 's' : ''}`
      : '';
  }

  if (!filtered.length) {
    list.innerHTML = `<div class="news-empty">No ${_newsFilter === 'All' ? '' : _newsFilter + ' '}articles found.</div>`;
    return;
  }

  list.innerHTML = filtered.map(a => {
    // Format timestamp
    let timeStr = '';
    if (a.published) {
      try {
        const d = new Date(a.published);
        const now = Date.now();
        const diff = now - d.getTime();
        const hrs  = Math.floor(diff / 3600000);
        const mins = Math.floor(diff / 60000);
        if (hrs > 48)      timeStr = d.toLocaleDateString();
        else if (hrs >= 1) timeStr = `${hrs}h ago`;
        else if (mins >= 1)timeStr = `${mins}m ago`;
        else                timeStr = 'Just now';
      } catch (_) {}
    }

    const sentClass = a.sentiment || 'Neutral';
    const catLabel  = a.category  || 'General';
    const safeTitle = (a.title || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const safeSrc   = (a.source || '').replace(/</g, '&lt;');
    const safeUrl   = (a.url || '#');

    return `
      <div class="news-item">
        <a class="news-item-title" href="${safeUrl}" target="_blank" rel="noopener noreferrer">
          ${safeTitle}
        </a>
        <div class="news-item-meta">
          <span class="news-sent-tag ${sentClass}">${sentClass}</span>
          <span class="news-cat-tag">${catLabel}</span>
          ${safeSrc ? `<span>${safeSrc}</span>` : ''}
          ${timeStr ? `<span>${timeStr}</span>` : ''}
        </div>
      </div>`;
  }).join('');
}
