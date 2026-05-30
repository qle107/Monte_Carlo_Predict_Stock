

//
// The full happy-path renderers (`_renderHMM`, `_renderHawkes`,
// `_renderBlendedZones`, `_renderVolumeProfile`, `_renderOptionsFlow`,
// `_renderMarketStructure`, `runMarketStructure`) still live inline in
// dashboard.html - extracting all 470 lines wholesale would consume the
// context budget for later phases.
//
// What THIS file owns:
//
//  • Error-state HTML for the three previously-empty sections (HMM regime,
//    Hawkes excitation, Blended Zone Reaction Probabilities) - driven by the

//        state ∈ {"ok", "error", "insufficient_data", "no_zones"}
//        error_reason, min_bars_required, bars_available
//
//  • A retry button that re-invokes `runMarketStructure()`.
//
//  • Console.debug logs at each pipeline step:
//        request → response → render-hmm → render-hawkes → render-zones
//
//  • Monkey-patch override of `_renderHMM` / `_renderHawkes` /
//    `_renderBlendedZones` / `_renderMarketStructure` to wrap them in
//    try/catch and route the new state contract to the error renderers.

(function () {
  'use strict';

  const _DBG = (label, payload) => {
    try { console.debug(`[ms] ${label}`, payload); } catch (_e) { /* no-op */ }
  };

  // Produces the *inside* of a card body - caller decides which IDs to wipe.
  function _emptyStateHtml(opts) {
    const {
      icon       = '⚠',
      color      = 'var(--amber)',
      title      = 'Unavailable',
      message    = '',
      showRetry  = true,
      retryLabel = '↻ Retry',
    } = opts || {};
    const retryBtn = showRetry
      ? `<button onclick="retryMarketStructure()"
             style="margin-top:10px;background:rgba(88,166,255,.1);border:1px solid rgba(88,166,255,.3);
                    color:var(--blue);border-radius:6px;padding:5px 14px;font-size:11px;font-weight:600;
                    cursor:pointer;transition:background .12s;"
             onmouseover="this.style.background='rgba(88,166,255,.2)'"
             onmouseout="this.style.background='rgba(88,166,255,.1)'"
        >${retryLabel}</button>`
      : '';
    return `<div style="text-align:center;padding:14px 8px;color:var(--muted);">
      <div style="font-size:22px;margin-bottom:6px;">${icon}</div>
      <div style="font-size:13px;font-weight:700;color:${color};margin-bottom:4px;">${title}</div>
      <div style="font-size:11px;line-height:1.5;max-width:280px;margin:0 auto;">${message}</div>
      ${retryBtn}
    </div>`;
  }

  // Replaces the empty-when-disabled rendering with a clear status message.
  // Falls through to the original inline _renderHMM() for the happy path.
  function _renderHMMSection(hmm) {
    _DBG('render-hmm', hmm);

    // Find the card body (the parent of #ms-hmm-state holds the whole card)
    const stateEl = document.getElementById('ms-hmm-state');
    if (!stateEl) return;
    const cardBody = stateEl.parentElement;
    if (!cardBody) return;

    // Cached original markup so a successful render can restore the layout.
    if (!cardBody.dataset.originalHtml) cardBody.dataset.originalHtml = cardBody.innerHTML;

    const state = (hmm && hmm.state) || (hmm ? 'ok' : 'error');
    if (state === 'ok') {
      // Restore layout if we'd previously rendered an error state into the body
      if (cardBody.dataset.lastState && cardBody.dataset.lastState !== 'ok') {
        cardBody.innerHTML = cardBody.dataset.originalHtml;
      }
      cardBody.dataset.lastState = 'ok';
      // Delegate to the inline happy-path renderer (preserved in dashboard.html).
      if (typeof window._renderHMM_original === 'function') {
        try { window._renderHMM_original(hmm); }
        catch (e) { console.warn('[ms] _renderHMM happy path failed:', e); }
      }
      return;
    }

    cardBody.dataset.lastState = state;

    if (state === 'insufficient_data') {
      cardBody.innerHTML = _emptyStateHtml({
        icon: '📊',
        color: 'var(--amber)',
        title: 'Market Regime - Need more candles',
        message: hmm.error_reason || `HMM requires at least ${hmm.min_bars_required || 40} bars; ${hmm.bars_available || 0} available. Switch to a longer timeframe.`,
      });
    } else if (state === 'error') {
      cardBody.innerHTML = _emptyStateHtml({
        icon: '⚠',
        color: 'var(--red)',
        title: 'Market Regime - Error',
        message: `Baum-Welch fit failed: <code>${(hmm.error || 'unknown')}</code>`,
      });
    } else {
      // Unknown state - fall back to neutral
      cardBody.innerHTML = _emptyStateHtml({
        icon: '❓',
        color: 'var(--muted)',
        title: 'Market Regime - Unavailable',
        message: 'No regime data returned from the server.',
      });
    }
  }

  function _renderHawkesSection(hk) {
    _DBG('render-hawkes', hk);

    const labelEl = document.getElementById('ms-hawkes-label');
    if (!labelEl) return;
    const cardBody = labelEl.parentElement;
    if (!cardBody) return;
    if (!cardBody.dataset.originalHtml) cardBody.dataset.originalHtml = cardBody.innerHTML;

    const state = (hk && hk.state) || (hk && hk.excitement_label ? 'ok' : 'error');
    if (state === 'ok') {
      if (cardBody.dataset.lastState && cardBody.dataset.lastState !== 'ok') {
        cardBody.innerHTML = cardBody.dataset.originalHtml;
      }
      cardBody.dataset.lastState = 'ok';
      if (typeof window._renderHawkes_original === 'function') {
        try { window._renderHawkes_original(hk); }
        catch (e) { console.warn('[ms] _renderHawkes happy path failed:', e); }
      }
      return;
    }

    cardBody.dataset.lastState = state;

    if (state === 'no_zones') {
      cardBody.innerHTML = _emptyStateHtml({
        icon: '🔍',
        color: 'var(--amber)',
        title: 'Price Activity - No zones',
        message: 'Hawkes excitation needs demand/supply zones to score reactions at. None were detected in the current lookback window.',
      });
    } else if (state === 'insufficient_data') {
      cardBody.innerHTML = _emptyStateHtml({
        icon: '📊',
        color: 'var(--amber)',
        title: 'Price Activity - Need more candles',
        message: hk.error_reason || `Hawkes process requires at least ${hk.min_bars_required || 20} return bars; ${hk.bars_available || 0} available.`,
      });
    } else if (state === 'error') {
      cardBody.innerHTML = _emptyStateHtml({
        icon: '⚠',
        color: 'var(--red)',
        title: 'Price Activity - Error',
        message: `Hawkes fit failed: <code>${(hk.error || 'unknown')}</code>`,
      });
    } else {
      cardBody.innerHTML = _emptyStateHtml({
        icon: '❓',
        color: 'var(--muted)',
        title: 'Price Activity - Unavailable',
        message: 'No Hawkes data returned from the server.',
      });
    }
  }

  // Now also surfaces zone-detection failure modes with min-bars hint.
  function _renderBlendedZonesSection(zones) {
    _DBG('render-zones', { n: (zones || []).length });
    const tbody = document.getElementById('ms-zone-tbody');
    if (!tbody) return;

    // Use the cached zones-meta from the last response so we can show the
    // min-bars hint when the array is empty for a specific reason.
    const meta = window.__msLastZonesMeta || {};

    if (!zones || zones.length === 0) {
      const reason  = meta.state === 'no_zones'
        ? `No demand/supply zones detected. Zone detection needs at least <strong>${meta.min_bars_required || 9}</strong> bars of history (only ${meta.bars_available || 0} available). Switch to a longer timeframe.`
        : meta.state === 'error'
          ? `Zone detection failed: <code>${meta.error || 'unknown'}</code>`
          : 'No demand/supply zones detected in the current lookback window.';
      tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--muted);padding:18px 14px;line-height:1.6;">
        ${reason}
        <div style="margin-top:8px;">
          <button onclick="retryMarketStructure()"
                  style="background:rgba(88,166,255,.1);border:1px solid rgba(88,166,255,.3);
                         color:var(--blue);border-radius:6px;padding:4px 12px;font-size:11px;
                         font-weight:600;cursor:pointer;">↻ Retry</button>
        </div>
      </td></tr>`;
      return;
    }

    // Happy path: delegate to inline renderer.
    if (typeof window._renderBlendedZones_original === 'function') {
      try { window._renderBlendedZones_original(zones); }
      catch (e) {
        console.warn('[ms] _renderBlendedZones happy path failed:', e);
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--red);padding:14px;">
          Render error: <code>${(e.message || e)}</code>
        </td></tr>`;
      }
    }
  }

  function _renderMarketStructureSafe(d) {
    _DBG('response', d);
    // Cache zones meta so _renderBlendedZonesSection can show min-bars info.
    window.__msLastZonesMeta = (d && d.zones) || {};

    try {
      // Update the ticker banner (always do this from this wrapper so a crash
      // in HMM/Hawkes/zones doesn't lose the banner update).
      const bannerTicker = document.getElementById('ms-banner-ticker');
      const bannerMeta   = document.getElementById('ms-banner-meta');
      if (bannerTicker) bannerTicker.textContent = (d.ticker || '-').toUpperCase();
      if (bannerMeta)   bannerMeta.textContent   = `${d.interval || ''} · Updated ${new Date(d.updated_at).toLocaleTimeString()}`;

      // Each section runs in its own try/catch - a crash in one cannot

      try { _renderHMMSection(d.hmm); }
      catch (e) { console.error('[ms] HMM section crashed:', e); }

      try { _renderHawkesSection(d.hawkes); }
      catch (e) { console.error('[ms] Hawkes section crashed:', e); }

      try {
        if (typeof window._renderOptionsFlow === 'function') {
          window._renderOptionsFlow(d.options_flow, d.current_price);
        }
      } catch (e) { console.error('[ms] OptionsFlow section crashed:', e); }

      try {
        if (typeof window._renderVolumeProfile === 'function') {
          window._renderVolumeProfile(d.volume_profile, d.current_price);
        }
      } catch (e) { console.error('[ms] VolumeProfile section crashed:', e); }

      try { _renderBlendedZonesSection(d.blended_zones); }
      catch (e) { console.error('[ms] BlendedZones section crashed:', e); }

    } catch (e) {
      console.error('[ms] _renderMarketStructure top-level crash:', e);
    }
  }

  function retryMarketStructure() {
    _DBG('retry', { msAnalysisInProgress: window._msAnalysisInProgress });
    if (typeof window.runMarketStructure === 'function') {
      window.runMarketStructure();
    } else {
      console.warn('[ms] runMarketStructure not yet defined - cannot retry.');
    }
  }

  // We do this AFTER the inline <script> has run so window._renderHMM etc.
  // exist; the inline module sits below ours in dashboard.html but our
  // installer runs deferred so order doesn't matter.
  function _install() {
    // Cache originals (first install only) then replace.
    if (typeof window._renderHMM === 'function' && !window._renderHMM_original) {
      window._renderHMM_original         = window._renderHMM;
      window._renderHMM                  = _renderHMMSection;
    }
    if (typeof window._renderHawkes === 'function' && !window._renderHawkes_original) {
      window._renderHawkes_original      = window._renderHawkes;
      window._renderHawkes               = _renderHawkesSection;
    }
    if (typeof window._renderBlendedZones === 'function' && !window._renderBlendedZones_original) {
      window._renderBlendedZones_original = window._renderBlendedZones;
      window._renderBlendedZones          = _renderBlendedZonesSection;
    }
    if (typeof window._renderMarketStructure === 'function' && !window._renderMarketStructure_original) {
      window._renderMarketStructure_original = window._renderMarketStructure;
      window._renderMarketStructure          = _renderMarketStructureSafe;
    }
    window.retryMarketStructure = retryMarketStructure;
    _DBG('installed', {
      patched_HMM:           !!window._renderHMM_original,
      patched_Hawkes:        !!window._renderHawkes_original,
      patched_BlendedZones:  !!window._renderBlendedZones_original,
      patched_Top:           !!window._renderMarketStructure_original,
    });
  }

  // Run install once the DOM is ready AND the inline scripts have populated
  // the globals. Both conditions hold by 'DOMContentLoaded' since the inline
  // <script> blocks are parsed before that event fires.
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _install);
  } else {
    _install();
  }

  if (window.__mc_trader_modular__) {
    window.__mc_trader_modular__.extracted.push('tabs/market-structure.js');
  }
})();
