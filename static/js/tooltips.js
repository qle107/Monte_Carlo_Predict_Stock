/**
 * Tooltip copy for metric labels.
 */

(function () {
  'use strict';

  const METRIC_TOOLTIPS = {
    'Drift bias':       'Avg drift per bar from recent returns.',
    'Confidence':       'How much the indicators agree. Under 50% = weak read.',
    'CVaR 5%':          'Avg loss in the worst 5% of MC paths.',
    'Median target':    'Median price at the forward horizon.',

    'RSI (14)':         'RSI(14). Above 70 hot, below 30 cold.',
    'Slope':            'Trend slope fed into the sim.',
    'Momentum':         'ROC-style momentum.',
    'MACD hist':        'MACD minus signal line.',
    'Bollinger':        'Distance from the BB midline (in stdevs).',
    'ADX':              'Trend strength, not direction.',
    'Volatility':       'Annualized hist vol.',

    'Max Pain':         'Strike that hurts option buyers most into expiry.',
    'Gamma Flip':       'Price where dealer gamma flips sign.',
    'Call Wall':        'Highest call OI strike.',
    'Put Wall':         'Highest put OI strike.',
    'P/C Ratio (Vol)':  'Put volume / call volume.',
    'P/C Ratio (OI)':   'Put OI / call OI.',

    'POC':              'Highest-volume price in the profile.',
    'HVN':              'High-volume node.',
    'LVN':              'Low-volume node.',
    'P10 / P90':        'MC 10th / 90th percentile bands.',
    'Zone Strength':    'How often price respected this zone.',
    'Hurst':            'Below 0.5 mean-reverts; above 0.5 trends.',

    'CPI':              'CPI YoY %.',
    'PPI':              'PPI YoY %.',
    'Core PCE':         'Core PCE YoY %.',
    'Fed Rate':         'Fed funds rate.',
    '10Y Yield':        '10Y Treasury yield.',
    'GDP Growth':       'Real GDP growth (annualized).',
    'Unemployment':     'Unemployment rate.',
    'ISM Mfg PMI':      'Manufacturing PMI. 50 = flat line.',
  };

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

  function initTooltips() {
    if (typeof tippy !== 'function') return;

    document.querySelectorAll('.metric .lbl').forEach(lbl => {
      const key = lbl.textContent.trim();
      const tip = METRIC_TOOLTIPS[key];
      if (tip && !lbl.querySelector('.mc-tip-icon')) {
        lbl.innerHTML = lbl.textContent + _infoIcon(tip);
      }
    });

    document.querySelectorAll('.ind-name').forEach(el => {
      const key = el.textContent.trim();
      const tip = METRIC_TOOLTIPS[key];
      if (tip && !el.querySelector('.mc-tip-icon')) {
        el.innerHTML = el.textContent + _infoIcon(tip);
      }
    });

    Object.entries(ID_TOOLTIPS).forEach(([id, tip]) => {
      const valEl = document.getElementById(id);
      if (!valEl) return;
      valEl.title = '';
      if (!valEl.dataset.tippyContent) {
        valEl.setAttribute('data-tippy-content', tip);
      }
    });

    document.querySelectorAll('[data-metric-tip]').forEach(el => {
      const key = el.getAttribute('data-metric-tip');
      const tip = METRIC_TOOLTIPS[key];
      if (tip && !el.dataset.tippyContent) {
        el.setAttribute('data-tippy-content', tip);
      }
    });

    tippy('[data-tippy-content]', {
      allowHTML: false,
      placement: 'top',
      theme: 'mc-dark',
      delay: [200, 0],
      maxWidth: 320,
      animation: 'fade',
    });
  }

  window.initTooltips = initTooltips;
  window.METRIC_TOOLTIPS = METRIC_TOOLTIPS;

  function _tryInit() {
    if (typeof tippy === 'function') {
      initTooltips();
    } else {
      setTimeout(() => { if (typeof tippy === 'function') initTooltips(); }, 1500);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _tryInit);
  } else {
    _tryInit();
  }

}());
