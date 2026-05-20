// ──────────────────────────────────────────────────────────────────────────
// /static/js/right-panel.js — Phase 7: right sidebar card rendering.
//
// What this file owns:
//
//  • window.renderRightPanel(d)  — single entry point called by the WS
//    message handler in dashboard.html after a full analysis broadcast.
//    Internally delegates to the per-card helpers below.
//
//  • Confidence colour scale (Fix 7-A):
//    < 30 % → var(--muted) grey
//    30–50 % → var(--amber)
//    > 50 % → var(--green)
//    Applied to both #conf-bar (fill colour) and #conf-val (text colour).
//
//  • Trade Setup confidence gate (Fix 7-B):
//    Levels are shown only when trade_setup.valid AND confidence > 40 %.
//    When valid but low-confidence the banner reads "⊘ No Edge" instead
//    of showing the level grid.
//
//  • Walk-Forward Backtest (Fix 7-C):
//    window.runBacktest() is verified working and re-exposed here so the
//    onclick="runBacktest()" button still finds it even after the inline
//    definition is eventually retired.
//
//  • Per-section try/catch error boundaries — a crash in one card never
//    takes down the whole right panel.
//
// Convention (matches Phases 3/4):
//  The large happy-path renderers (updatePriceTargets, updateSRCard,
//  _updateMicrostructureCard) still live inline in dashboard.html.
//  This module monkey-patches / overrides specific functions and adds the
//  window.renderRightPanel façade. Inline code that already works is NOT
//  moved wholesale — only touched functions are extracted.
// ──────────────────────────────────────────────────────────────────────────

(function () {
  'use strict';

  // ── Tiny helpers ──────────────────────────────────────────────────────────
  const _el  = id => document.getElementById(id);
  const _set = (id, fn) => { const el = _el(id); if (el) fn(el); };
  const _fmt = (v, dp = 2) =>
    (v != null && isFinite(v)) ? Number(v).toFixed(dp) : '—';
  const _fmtPct = (v, dp = 1) =>
    (v != null && isFinite(v))
      ? (v >= 0 ? '+' : '') + Number(v).toFixed(dp) + '%'
      : '—';
  const _fmtDollar = v =>
    (v != null && isFinite(v)) ? '$' + Number(v).toFixed(2) : '—';

  // ── Fix 7-A: confidence colour ────────────────────────────────────────────
  function _confColor(pct) {
    if (pct > 50)  return 'var(--green)';
    if (pct >= 30) return 'var(--amber)';
    return 'var(--muted)';
  }

  function _renderConfidence(confPct) {
    const bar = _el('conf-bar');
    const val = _el('conf-val');
    if (!bar || !val) return;
    bar.style.width      = confPct + '%';
    bar.style.background = _confColor(confPct);
    val.textContent      = confPct + '%';
    val.style.color      = _confColor(confPct);
  }

  // ── Fix 7-B: Trade Setup with confidence gate ────────────────────────────
  // Full replacement of the inline updateTradeSetup.
  // Signature extended: updateTradeSetup(ts, confPct)
  //   ts      — d.trade_setup from the broadcast
  //   confPct — integer 0-100 from Math.round(sig.confidence * 100)
  function _updateTradeSetup(ts, confPct) {
    if (!ts) return;

    const $   = id => document.getElementById(id);
    const set = (id, fn) => { const el = $(id); if (el) fn(el); };

    const banner = $('ts-banner');
    const pill   = $('ts-side-pill');
    const wrap   = $('ts-levels-wrap');
    if (!banner || !wrap) return;

    // Derive effective confidence — prefer ts.confidence (backend sets it),
    // fall back to the caller-supplied confPct.
    const tsConf = ts.confidence != null
      ? Math.round(ts.confidence * 100)
      : (confPct != null ? confPct : 0);
    const confOk = tsConf > 40;

    const f2    = v => v != null ? '$' + Number(v).toFixed(2) : '—';
    const fRR   = v => v != null ? Number(v).toFixed(1) + 'R' : '—';
    const fProb = v => v != null ? Math.round(v * 100) + '%' : '—';

    // Reset banner class
    banner.className = 'ts-banner';

    // ── Case 1: signal not valid (conditions not met) ──────────────────────
    if (!ts.valid) {
      banner.classList.add('none');
      set('ts-banner-label', el => el.textContent = '⊘  No Entry Right Now');
      const reason  = ts.reason || 'Conditions not met';
      const rrFail  = reason.includes('R:R too low') || reason.includes('TP targets');
      set('ts-banner-reason', el => {
        el.textContent        = reason;
        el.title              = rrFail
          ? 'The scanner shows ATR-estimated TP targets which are wider. ' +
            'This sidebar uses real MC P75/P90 from 10k paths — tighter and ' +
            'more accurate for the actual forward-candle horizon.'
          : '';
        el.style.cursor          = rrFail ? 'help' : '';
        el.style.textDecoration  = rrFail ? 'underline dotted' : '';
      });
      if (pill) { pill.textContent = '—'; pill.className = 'pill'; }
      wrap.style.display = 'none';
      return;
    }

    // ── Case 2: valid signal but confidence ≤ 40% → No Edge ───────────────
    if (!confOk) {
      banner.classList.add('none');
      set('ts-banner-label', el => el.textContent = '⊘  No Edge');
      set('ts-banner-reason', el => {
        el.textContent          = `Confidence ${tsConf}% — need >40% for a trade signal`;
        el.title                = '';
        el.style.cursor         = '';
        el.style.textDecoration = '';
      });
      if (pill) { pill.textContent = '—'; pill.className = 'pill'; }
      wrap.style.display = 'none';
      return;
    }

    // ── Case 3: valid + confident — show full level grid ──────────────────
    const isLong = ts.side === 'long';
    banner.classList.add(ts.side);
    set('ts-banner-label',  el => el.textContent =
      isLong ? '▲  LONG — Entry Confirmed' : '▼  SHORT — Entry Confirmed');
    set('ts-banner-reason', el => {
      el.textContent          = ts.reason || '';
      el.title                = '';
      el.style.cursor         = '';
      el.style.textDecoration = '';
    });

    if (pill) {
      pill.textContent      = ts.side.toUpperCase();
      pill.className        = 'pill';
      pill.style.background = isLong ? 'rgba(63,185,80,.15)' : 'rgba(248,81,73,.15)';
      pill.style.color      = isLong ? 'var(--green)' : 'var(--red)';
    }

    // Levels
    set('ts-entry',     el => el.textContent = f2(ts.entry));
    set('ts-entry-sub', el => el.textContent =
      ts.direction ? ts.direction.replace(/_/g, ' ') : '');

    const tp1Dist = ts.tp1_dist != null ? ts.tp1_dist.toFixed(1) : null;
    const tp2Dist = ts.tp2_dist != null ? ts.tp2_dist.toFixed(1) : null;
    set('ts-tp1',     el => el.textContent = f2(ts.tp1));
    set('ts-tp1-sub', el => el.textContent = tp1Dist ? `+${tp1Dist}% from entry` : '');
    set('ts-tp2',     el => el.textContent = f2(ts.tp2));
    set('ts-tp2-sub', el => el.textContent = tp2Dist ? `+${tp2Dist}% from entry` : '');

    // Recommended SL
    const recSL    = ts.sl_recommended === 'pct' ? ts.sl_pct : ts.sl_atr;
    const recDist  = ts.sl_recommended === 'pct' ? ts.sl_pct_dist : ts.sl_atr_dist;
    const recLabel = ts.sl_recommended === 'pct' ? 'Fixed %' : 'ATR';
    set('ts-sl-main', el => el.textContent = f2(recSL));
    set('ts-sl-sub',  el => el.textContent =
      recDist != null ? `${recDist.toFixed(1)}% risk · ${recLabel} stop` : '');

    // Dual SL detail
    const atrVal  = f2(ts.sl_atr) + (ts.sl_atr_dist != null ? ` (${ts.sl_atr_dist.toFixed(1)}%)` : '');
    const pctVal  = f2(ts.sl_pct) + (ts.sl_pct_dist != null ? ` (${ts.sl_pct_dist.toFixed(1)}%)` : '');
    const capNote = ts.atr_capped ? ' ⚠ capped' : '';

    set('ts-sl-atr-val', el => {
      el.textContent = atrVal + capNote;
      el.style.color = ts.sl_recommended === 'atr' ? 'var(--fg)' : 'var(--muted)';
      el.title = ts.atr_capped
        ? 'ATR SL was too wide and capped to max % for this timeframe' : '';
    });
    set('ts-sl-pct-val', el => {
      el.textContent = pctVal;
      el.style.color = ts.sl_recommended === 'pct' ? 'var(--fg)' : 'var(--muted)';
    });
    set('ts-sl-atr-label', el =>
      el.textContent = `ATR (×${ts.atr_mult?.toFixed(1) ?? '?'})${ts.sl_recommended === 'atr' ? ' ★' : ''}`);
    set('ts-sl-pct-label', el =>
      el.textContent = `Fixed ${ts.sl_pct_used ?? '?'}%${ts.sl_recommended === 'pct' ? ' ★' : ''}`);

    // R:R pills
    const setRRPill = (pillId, numId, val, isRec) => {
      set(numId,  el => el.textContent = fRR(val));
      set(pillId, el => {
        el.className   = 'ts-rr-pill ' + (val >= 2.0 ? 'good' : val >= 1.3 ? 'ok' : 'bad');
        el.style.outline = isRec ? '1px solid var(--blue)' : '';
        el.title         = isRec ? 'Recommended (tighter stop)' : '';
      });
    };
    setRRPill('ts-rr-atr-pill', 'ts-rr-atr', ts.rr_atr, ts.sl_recommended === 'atr');
    setRRPill('ts-rr-pct-pill', 'ts-rr-pct', ts.rr_pct, ts.sl_recommended === 'pct');

    // MC probabilities
    const pSL = ts.sl_recommended === 'pct' ? (ts.prob_sl_pct || 0) : (ts.prob_sl_atr || 0);
    set('ts-prob-tp1', el => {
      el.textContent = fProb(ts.prob_tp1);
      el.style.color = (ts.prob_tp1 || 0) >= 0.5 ? 'var(--green)' : 'var(--amber)';
    });
    set('ts-prob-tp2', el => {
      el.textContent = fProb(ts.prob_tp2);
      el.style.color = (ts.prob_tp2 || 0) >= 0.35 ? 'var(--green)' : 'var(--muted)';
    });
    set('ts-prob-sl', el => {
      el.textContent = fProb(pSL);
      el.style.color = pSL > 0.4 ? 'var(--red)' : pSL > 0.25 ? 'var(--amber)' : 'var(--muted)';
    });

    // Stop-hunt warning
    set('ts-stophunt-warn', el => {
      if (pSL > 0.35) {
        set('ts-stophunt-pct', p => p.textContent = Math.round(pSL * 100) + '%');
        el.style.display = 'block';
      } else {
        el.style.display = 'none';
      }
    });

    // Zone-based TP/SL panel
    const zoneWrap = $('ts-zone-wrap');
    if (zoneWrap) {
      const hasZone = ts.zone_tp1 != null && ts.zone_sl != null;
      zoneWrap.style.display = hasZone ? 'block' : 'none';
      if (hasZone) {
        const zTypeLabels = {
          demand_bounce:  '↑ Demand Bounce',
          demand_support: '↑ Demand Support',
          supply_break:   '↑ Supply Break',
          supply_bounce:  '↓ Supply Reject',
          demand_break:   '↓ Demand Break',
        };
        set('ts-zone-type-pill', el => {
          el.textContent = zTypeLabels[ts.zone_type] || ts.zone_type || '?';
          const isUp = ts.zone_type === 'demand_bounce' || ts.zone_type === 'demand_support' ||
                       ts.zone_type === 'supply_break';
          el.style.background = isUp ? 'rgba(63,185,80,.15)' : 'rgba(248,81,73,.15)';
          el.style.color      = isUp ? 'var(--green)' : 'var(--red)';
        });
        set('ts-zone-strength', el => {
          const s = ts.zone_strength != null ? Math.round(ts.zone_strength * 100) + '%' : '—';
          el.textContent = 'zone str: ' + s;
        });
        set('ts-zone-tp1',     el => el.textContent = f2(ts.zone_tp1));
        set('ts-zone-tp1-sub', el => el.textContent =
          ts.zone_tp1_dist != null ? `+${ts.zone_tp1_dist.toFixed(1)}% from entry` : '');
        if (ts.zone_tp2 != null) {
          set('ts-zone-tp2',     el => el.textContent = f2(ts.zone_tp2));
          set('ts-zone-tp2-sub', el => el.textContent =
            ts.zone_tp2_dist != null ? `+${ts.zone_tp2_dist.toFixed(1)}% from entry` : '');
        } else {
          set('ts-zone-tp2',     el => el.textContent = '—');
          set('ts-zone-tp2-sub', el => el.textContent = '');
        }
        set('ts-zone-sl',     el => el.textContent = f2(ts.zone_sl));
        set('ts-zone-sl-sub', el => el.textContent =
          ts.zone_sl_dist != null ? `${ts.zone_sl_dist.toFixed(1)}% risk` : '');
        set('ts-zone-rr', el => {
          if (ts.zone_rr != null) {
            el.textContent = ts.zone_rr.toFixed(1) + 'R';
            el.style.color = ts.zone_rr >= 2.0 ? 'var(--green)'
              : ts.zone_rr >= 1.3 ? 'var(--amber)' : 'var(--red)';
          } else {
            el.textContent = '—';
            el.style.color = 'var(--muted)';
          }
        });
        const ctxLabel = {
          at_demand: 'Price at demand zone',
          at_supply: 'Price at supply zone',
          between:   'Price between zones',
          unknown:   '',
        };
        set('ts-zone-context', el =>
          el.textContent = ctxLabel[ts.zone_context] || '');
      }
    }

    wrap.style.display = 'block';
  }

  // ── Fix 7-C: Walk-Forward Backtest (re-exposed; already correct inline) ──
  async function _runBacktest() {
    const btn    = _el('bt-run');
    const status = _el('bt-status');
    if (!btn || !status) return;
    btn.disabled     = true;
    status.textContent = 'Running walk-forward…';
    status.style.color = '';
    try {
      const res = await fetch('/api/backtest', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ history_bars: 200, n_sim: 500 }),
      });
      const j = await res.json();
      if (!res.ok || j.ok === false) {
        status.textContent = j.detail || j.error || 'Backtest failed';
        status.style.color = 'var(--red)';
      } else {
        _set('bt-hit',   el => el.textContent = j.hit_rate  == null ? '—' : j.hit_rate + '%');
        _set('bt-brier', el => el.textContent = j.brier_score != null ? Number(j.brier_score).toFixed(3) : '—');
        _set('bt-ll',    el => el.textContent = j.log_loss    != null ? Number(j.log_loss).toFixed(3)    : '—');
        _set('bt-corr',  el => el.textContent = j.expected_vs_real != null
          ? Number(j.expected_vs_real).toFixed(2) : '—');
        status.style.color = 'var(--muted)';
        const mpu = j.mean_prob_up    != null ? Number(j.mean_prob_up).toFixed(1) : '?';
        const rur = j.real_up_rate    != null ? Number(j.real_up_rate).toFixed(1) : '?';
        status.textContent = `${j.n_evaluated} evaluations · ${j.n_called} buy/sell calls` +
          ` · mean prob_up ${mpu}% · realised up ${rur}%`;
      }
    } catch (e) {
      status.textContent = 'Failed: ' + e;
      status.style.color = 'var(--red)';
    } finally {
      btn.disabled = false;
    }
  }

  // ── window.renderRightPanel(d) ─────────────────────────────────────────────
  // Single entry-point called from dashboard.html after a full analysis
  // broadcast.  Each section is wrapped in its own try/catch so a crash in
  // one card never darkens the rest.
  function _renderRightPanel(d) {
    if (!d) return;

    const sig     = d.signal      || {};
    const mc      = d.mc          || {};
    const ind     = d.indicators  || {};
    const cfg_d   = d.config      || {};
    const cur     = d.current_price;

    // ── Confidence colour (Fix 7-A) ─────────────────────────────────────
    try {
      const confPct = Math.round((sig.confidence || 0) * 100);
      _renderConfidence(confPct);
    } catch (e) { console.warn('[rp] confidence:', e); }

    // ── Trade Setup (Fix 7-B) ────────────────────────────────────────────
    try {
      const confPct = Math.round((sig.confidence || 0) * 100);
      _updateTradeSetup(d.trade_setup, confPct);
    } catch (e) { console.warn('[rp] trade-setup:', e); }

    // ── Price Targets — delegate to inline updatePriceTargets ───────────
    try {
      if (typeof window.updatePriceTargets === 'function') {
        window.updatePriceTargets(d.trade_setup, cur);
      }
    } catch (e) { console.warn('[rp] price-targets:', e); }

    // ── S/R card — delegate to inline updateSRCard ───────────────────────
    try {
      if (typeof window.updateSRCard === 'function') {
        window.updateSRCard(d.zones, d.indicators, cur);
      }
    } catch (e) { console.warn('[rp] sr-card:', e); }

    // Note: _updateMicrostructureCard is intentionally NOT called here —
    // it is already invoked by the updateUI() wrapper in dashboard.html
    // (line ~4904) to avoid double-rendering the microstructure card.
  }

  // ── Wire up on DOMContentLoaded ───────────────────────────────────────────
  // Expose everything on window so onclick="runBacktest()" and
  // inline call-sites using window.* can resolve them.
  document.addEventListener('DOMContentLoaded', function () {

    // Override inline updateTradeSetup with the Phase-7-B version.
    // The inline _updateUI now calls updateTradeSetup(d.trade_setup, confPct)
    // — but since updateTradeSetup is function-scoped inside the inline <script>
    // we can't reach it via window.  We instead expose the new version on window
    // and the companion guard in dashboard.html routes through it.
    window.updateTradeSetup = _updateTradeSetup;

    // Expose backtest runner (button onclick="runBacktest()" finds window first).
    window.runBacktest = _runBacktest;

    // Primary façade — called by the WS handler guard in dashboard.html.
    window.renderRightPanel = _renderRightPanel;

    console.debug('[right-panel] Phase 7 module loaded — renderRightPanel, ' +
      'updateTradeSetup (conf gate), runBacktest, confidence colour scale ready.');
  });

})();
