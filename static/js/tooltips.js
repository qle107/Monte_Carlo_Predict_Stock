/**
 * static/js/tooltips.js — Metric tooltip definitions + Tippy.js initialization.
 * Fix 5: Adds "?" info icons with explanatory tooltips to every metric label.
 *
 * Uses Tippy.js (MIT) loaded via CDN. Falls back silently if Tippy is not
 * available (e.g. offline), so the rest of the dashboard still works.
 *
 * To add a tooltip to a new metric:
 *   1. Add an entry to METRIC_TOOLTIPS keyed by the exact text or element ID.
 *   2. Call initTooltips() after any render that adds new metric labels.
 */

(function () {
  'use strict';

  // ── Metric tooltip definitions ────────────────────────────────────────────
  const METRIC_TOOLTIPS = {
    // ── Strip metrics (matched by .lbl text) ────────────────────────────────
    'Drift bias':       'The average directional drift per candle, net of noise. Negative = slight downward pressure in the Monte Carlo paths. Derived from log-return distribution.',
    'Confidence':       'Model self-confidence: agreement across indicators (RSI, MACD, slope, regime). <50% = low conviction — treat the signal cautiously.',
    'CVaR 5%':          'Conditional Value at Risk — average loss expected in the worst 5% of Monte Carlo scenarios. A CVaR of −31% means you could lose ~31% on average in bad tail events.',
    'Median target':    'The median price across all Monte Carlo paths at the n-forward candle horizon. Half of simulated paths end above this price.',

    // ── AI signal panel (matched by .ind-name text) ─────────────────────────
    'RSI (14)':         'Relative Strength Index (14-period). >70 = overbought (often reversal risk). <30 = oversold (often bounce potential). 30–70 = no extreme bias.',
    'Slope':            'Price momentum slope — rate of change of the closing price trend. Positive = uptrend bias in the simulation paths. Negative = downtrend bias.',
    'Momentum':         'Rate-of-change momentum oscillator. Measures how fast price is moving in a direction. High positive = strong buying pressure; high negative = strong selling.',
    'MACD hist':        'MACD Histogram — difference between MACD line and signal line. Positive = bullish momentum building. Negative = bearish momentum. Crossing zero = potential trend change.',
    'Bollinger':        'Distance from the Bollinger Band midline in standard deviations. Negative = below midline, approaching lower band. Positive = above midline, approaching upper band.',
    'ADX':              'Average Directional Index. >25 = trending market (direction matters). <20 = no clear trend (range-bound). Does NOT indicate direction — only trend strength.',
    'Volatility':       'Historical volatility (annualised %). Higher volatility = wider Monte Carlo confidence bands and higher risk/reward on trade setups.',

    // ── Options flow metrics ─────────────────────────────────────────────────
    'Max Pain':         'The option strike price where the maximum number of contracts expire worthless — often acts as a gravitational target near expiry as market makers hedge their books.',
    'Gamma Flip':       'The price level where dealer gamma exposure shifts from negative (amplifies moves) to positive (dampens moves). Below the Gamma Flip = more volatile market conditions.',
    'Call Wall':        'The strike with the highest call open interest. Acts as a resistance ceiling — dealers must sell stock to hedge as price approaches it, creating supply pressure.',
    'Put Wall':         'The strike with the highest put open interest. Acts as a support floor — dealers must buy stock to hedge as price approaches it, creating demand pressure.',
    'P/C Ratio (Vol)':  'Put/Call Volume Ratio. <0.7 = call-heavy (bullish sentiment). >1.0 = put-heavy (bearish/hedging). Extreme values often mean contrarian setups.',
    'P/C Ratio (OI)':   'Put/Call Open Interest Ratio. Reflects total outstanding contracts. Useful for spotting structural positioning beyond short-term trading activity.',

    // ── Market structure metrics ─────────────────────────────────────────────
    'POC':              'Point of Control — the single price level with the highest traded volume in the volume profile. Often acts as a magnet for price action.',
    'HVN':              'High Volume Node — price levels where the most volume traded historically. Act as strong support/resistance and consolidation magnets.',
    'LVN':              'Low Volume Node — price levels with little historical trading. Price tends to move quickly through LVNs (no support/resistance until the next HVN).',
    'P10 / P90':        'Monte Carlo percentile bands. P10 = price exceeded in only 10% of simulation paths (pessimistic scenario). P90 = price reached in only 10% of paths (optimistic).',
    'Zone Strength':    'Composite score (0–100%) for how reliable a demand/supply zone has been historically — based on number of price touches and bounce rate at that level.',
    'Hurst':            'Hurst Exponent — measures trend persistence. <0.5 = mean-reverting (price tends to bounce back). >0.5 = trending (momentum persists). 0.5 = random walk.',

    // ── Macro indicators ────────────────────────────────────────────────────
    'CPI':              'Consumer Price Index (YoY %). Measures retail inflation. Rising CPI → Fed may hike rates → bearish for equities. Falling CPI → potential rate cuts → bullish.',
    'PPI':              'Producer Price Index (YoY %). Measures wholesale/factory inflation — leads CPI by 1–2 months. Rising PPI often passes through to higher CPI.',
    'Core PCE':         'Core Personal Consumption Expenditures (YoY %). The Fed\'s preferred inflation gauge (excludes food & energy). Above 2% target → hawkish bias.',
    'Fed Rate':         'Federal Funds Rate. Higher rate = tighter credit, higher discount rates → lower equity valuations. Lower rate = looser credit → bullish for growth stocks.',
    '10Y Yield':        '10-Year Treasury Yield (%). When yields rise sharply, growth stocks and risk assets typically sell off as the "risk-free" rate becomes more attractive.',
    'GDP Growth':       'Real GDP Growth Rate (QoQ annualised %). Negative for two consecutive quarters = technical recession. Strong GDP → corporate earnings growth.',
    'Unemployment':     'Civilian Unemployment Rate (%). Low unemployment = strong economy but wage inflation risk. Sudden spike = recessionary signal.',
    'ISM Mfg PMI':      'ISM Manufacturing PMI. >50 = expansion. <50 = contraction. Watched closely as a leading indicator of economic activity and corporate earnings.',
  };

  // ── Info icon HTML ────────────────────────────────────────────────────────
  function _infoIcon(tipContent) {
    return `<span class="mc-tip-icon" data-tippy-content="${_escape(tipContent)}"
      style="display:inline-flex;align-items:center;justify-content:center;
             width:13px;height:13px;border-radius:50%;background:rgba(139,148,158,.25);
             color:#8b949e;font-size:9px;font-weight:700;margin-left:4px;cursor:help;
             vertical-align:middle;flex-shrink:0;">?</span>`;
  }

  function _escape(s) {
    return String(s).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  // ── Apply tooltips by element ID ─────────────────────────────────────────
  const ID_TOOLTIPS = {
    'm-drift':  METRIC_TOOLTIPS['Drift bias'],
    'm-conf':   METRIC_TOOLTIPS['Confidence'],
    'm-cvar':   METRIC_TOOLTIPS['CVaR 5%'],
    'm-target': METRIC_TOOLTIPS['Median target'],
    'v-rsi':    METRIC_TOOLTIPS['RSI (14)'],
    'v-slope':  METRIC_TOOLTIPS['Slope'],
    'v-mom':    METRIC_TOOLTIPS['Momentum'],
    'v-macd':   METRIC_TOOLTIPS['MACD hist'],
    'v-bb':     METRIC_TOOLTIPS['Bollinger'],
    'v-adx':    METRIC_TOOLTIPS['ADX'],
    'v-vol':    METRIC_TOOLTIPS['Volatility'],
  };

  // ── Main initializer ──────────────────────────────────────────────────────
  function initTooltips() {
    if (typeof tippy !== 'function') return;  // Tippy not loaded yet

    // 1. Metric strip (.lbl text → parent .metric)
    document.querySelectorAll('.metric .lbl').forEach(lbl => {
      const key = lbl.textContent.trim();
      const tip = METRIC_TOOLTIPS[key];
      if (tip && !lbl.querySelector('.mc-tip-icon')) {
        lbl.innerHTML = lbl.textContent + _infoIcon(tip);
      }
    });

    // 2. AI signal panel (.ind-name text)
    document.querySelectorAll('.ind-name').forEach(el => {
      const key = el.textContent.trim();
      const tip = METRIC_TOOLTIPS[key];
      if (tip && !el.querySelector('.mc-tip-icon')) {
        el.innerHTML = el.textContent + _infoIcon(tip);
      }
    });

    // 3. Specific value elements — add tooltip on their sibling label
    Object.entries(ID_TOOLTIPS).forEach(([id, tip]) => {
      const valEl = document.getElementById(id);
      if (!valEl) return;
      valEl.title = '';  // clear native title
      if (!valEl.dataset.tippyContent) {
        valEl.setAttribute('data-tippy-content', tip);
      }
    });

    // 4. Any element with data-metric-tip attribute (future extension)
    document.querySelectorAll('[data-metric-tip]').forEach(el => {
      const key = el.getAttribute('data-metric-tip');
      const tip = METRIC_TOOLTIPS[key];
      if (tip && !el.dataset.tippyContent) {
        el.setAttribute('data-tippy-content', tip);
      }
    });

    // 5. Initialize / reinitialize Tippy on all annotated elements
    tippy('[data-tippy-content]', {
      allowHTML: false,
      placement: 'top',
      theme: 'mc-dark',
      delay: [200, 0],
      maxWidth: 320,
      animation: 'fade',
    });
  }

  // ── Expose ────────────────────────────────────────────────────────────────
  window.initTooltips = initTooltips;
  window.METRIC_TOOLTIPS = METRIC_TOOLTIPS;

  // Auto-run after DOMContentLoaded (Tippy may still be loading — retry once)
  function _tryInit() {
    if (typeof tippy === 'function') {
      initTooltips();
    } else {
      // Tippy CDN loads async — wait for it
      setTimeout(() => { if (typeof tippy === 'function') initTooltips(); }, 1500);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _tryInit);
  } else {
    _tryInit();
  }

}());
