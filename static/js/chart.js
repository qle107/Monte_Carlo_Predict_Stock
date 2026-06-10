// Lightweight-Charts: build, incremental update, MC overlay, measure tool.

(function () {
  'use strict';

let _lastChartData = null;
let _lwChart       = null;
let _lwCandle      = null;
let _lwVol         = null;
let _lwMcMed       = null;
let _lwMcP75       = null;
let _lwMcP25       = null;
let _lwMcP90       = null;
let _lwMcP10       = null;
let _mcOverlayCanvas = null;   // canvas drawn on top of LWC for band fills
let _mcBandData      = null;   // pre-computed {p90,p75,p25,p10,paths} data

// To keep the chart behaviour TradingView-style - new ticks update the
// forming bar in place, new bars append, zoom/scroll are preserved - we
// rebuild the chart only when the ticker or timeframe changes. Otherwise
// every incoming poll just calls series.update() on the affected bars.
let _lastBuildTicker     = null;   // ticker the current chart was built for
let _lastBuildInterval   = null;   // interval ditto
let _lastSeenCandleTime  = 0;      // unix seconds of the newest bar drawn
let _firstSeenCandleTime = 0;      // unix seconds of the oldest bar drawn

function buildCharts(d) {
  _lastChartData = d;

  // Fall back to the previous build's identity if the partial payload
  // doesn't carry ticker/interval (sometimes the case during phase 1).
  const tk = d.ticker   || _lastBuildTicker;
  const iv = d.interval || _lastBuildInterval;

  // Detect "the new payload contains older bars than what's on the chart"
  // - happens when the user expands chart_bars or lookback in Settings.
  // We can't append bars before the existing series in LWC, so we have
  // to do a full rebuild in that case.
  let newPayloadFirstTime = 0;
  if (d.candles && d.candles.length) {
    newPayloadFirstTime = _toLwcSec(d.candles[0].t);
  }
  const olderDataArrived =
        _firstSeenCandleTime > 0 &&
        newPayloadFirstTime > 0 &&
        newPayloadFirstTime < _firstSeenCandleTime;

  const needsRebuild =
        !_lwChart                      ||
        !_lwCandle                     ||
        _lastBuildTicker   == null     ||
        _lastBuildInterval == null     ||
        tk !== _lastBuildTicker        ||
        iv !== _lastBuildInterval      ||
        olderDataArrived;

  if (needsRebuild) {
    _lastBuildTicker     = tk;
    _lastBuildInterval   = iv;
    _lastSeenCandleTime  = 0;          // reset; _setLwcData will seed
    _firstSeenCandleTime = 0;
    _buildLwcChart(d);
  } else {
    _updateLwcIncremental(d);
  }
}

window.addEventListener('resize', () => {
  if (!_lwChart) return;
  const el = document.getElementById('lwc-container');
  if (el && el.offsetWidth > 0) {
    _lwChart.applyOptions({ width: el.offsetWidth, height: el.offsetHeight });
    // Double rAF: let LWC finish its internal layout before reading coordinates
    requestAnimationFrame(() => requestAnimationFrame(_drawMcOverlay));
  }
});

function _buildLwcChart(d) {
  const el = document.getElementById('lwc-container');
  if (!el) return;
  const W = el.offsetWidth || 900;
  const H = el.offsetHeight || 420;
  if (W < 10) { requestAnimationFrame(() => _buildLwcChart(d)); return; }

  // Tear down previous instance cleanly
  if (_lwChart) { _lwChart.remove(); _lwChart = null; }
  _lwCandle = _lwVol = _lwMcMed = _lwMcP75 = _lwMcP25 = _lwMcP90 = _lwMcP10 = null;

  _lwChart = LightweightCharts.createChart(el, {
    width: W, height: H,
    layout: {
      background: { type: LightweightCharts.ColorType.Solid, color: '#0e1117' },
      textColor: '#8b949e',
      fontSize: 11,
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    },
    grid: {
      vertLines: { color: 'rgba(255,255,255,0.04)' },
      horzLines: { color: 'rgba(255,255,255,0.04)' },
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
      vertLine: { color: 'rgba(88,166,255,0.45)', width: 1, style: LightweightCharts.LineStyle.Dashed, labelBackgroundColor: '#161b22' },
      horzLine: { color: 'rgba(88,166,255,0.45)', width: 1, style: LightweightCharts.LineStyle.Dashed, labelBackgroundColor: '#161b22' },
    },
    rightPriceScale: {
      borderColor: 'rgba(255,255,255,0.08)',
      scaleMargins: { top: 0.04, bottom: 0.22 },
    },
    timeScale: {
      borderColor: 'rgba(255,255,255,0.08)',
      timeVisible: true,
      secondsVisible: false,
      rightOffset: 6,
    },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true },
    handleScale:  { mouseWheel: true, pinch: true },
  });

    //    conversion (priceToCoordinate). The canvas overlay draws the actual
  //    filled bands + border lines. lastValueVisible=true keeps the P90/P10/
  //    P75/P25 labels on the right-axis price scale.
  const _bandOpts = (title) => ({
    color: 'rgba(0,0,0,0)', lineWidth: 0,
    priceLineVisible: false, lastValueVisible: true,
    crosshairMarkerVisible: false, title,
  });
  _lwMcP90 = _lwChart.addLineSeries(_bandOpts('P90'));
  _lwMcP10 = _lwChart.addLineSeries(_bandOpts('P10'));
  _lwMcP75 = _lwChart.addLineSeries(_bandOpts('P75'));
  _lwMcP25 = _lwChart.addLineSeries(_bandOpts('P25'));
  _lwMcMed = _lwChart.addLineSeries({
    color: '#d29922', lineWidth: 2,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    priceLineVisible: false, lastValueVisible: true, crosshairMarkerVisible: true,
    title: 'MC Med',
  });

  _lwCandle = _lwChart.addCandlestickSeries({
    upColor: '#3fb950', downColor: '#f85149',
    borderUpColor: '#3fb950', borderDownColor: '#f85149',
    wickUpColor:   '#3fb950', wickDownColor:   '#f85149',
    priceLineVisible: true,
    priceLineColor: 'rgba(88,166,255,0.55)',
    priceLineWidth: 1,
    priceLineStyle: LightweightCharts.LineStyle.Dashed,
  });

  _lwVol = _lwChart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    priceScaleId: 'vol',
  });
  _lwChart.priceScale('vol').applyOptions({
    scaleMargins: { top: 0.82, bottom: 0 },
    drawTicks: false, borderVisible: false,
  });

  _createMcOverlay(el);

  // Redraw overlay whenever user zooms or pans
  _lwChart.timeScale().subscribeVisibleLogicalRangeChange(() => {
    requestAnimationFrame(() => requestAnimationFrame(_drawMcOverlay));
  });

  _setLwcData(d);

  const oldTt = document.getElementById('lwc-tt');
  if (oldTt) oldTt.remove();
  const tt = document.createElement('div');
  tt.id = 'lwc-tt';
  Object.assign(tt.style, {
    position:'absolute', top:'10px', left:'12px', zIndex:'20',
    display:'none', pointerEvents:'none',
    background:'rgba(22,27,34,0.96)',
    border:'1px solid rgba(255,255,255,0.12)',
    borderRadius:'7px', padding:'9px 13px',
    fontSize:'12px', lineHeight:'1.75', color:'#e6edf3',
    fontFamily:"-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif",
    boxShadow:'0 4px 16px rgba(0,0,0,0.5)',
    minWidth:'130px',
  });
  el.appendChild(tt);

  _lwChart.subscribeCrosshairMove(param => {
    if (!param || !param.time || !param.point || param.point.x < 0 || param.point.y < 0) {
      tt.style.display = 'none'; return;
    }
    const bar = param.seriesData.get(_lwCandle);
    if (!bar) { tt.style.display = 'none'; return; }
    const up  = bar.close >= bar.open;
    const chg = bar.close - bar.open;
    const col = up ? '#3fb950' : '#f85149';
    const dt  = new Date(param.time * 1000);
    const ds  = dt.toLocaleString([], { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit' });
    tt.innerHTML = `
      <div style="font-size:10px;color:#8b949e;margin-bottom:5px;font-weight:600;letter-spacing:.3px;">${ds}</div>
      <div style="display:grid;grid-template-columns:16px 1fr;gap:0 10px;align-items:baseline;">
        <span style="color:#8b949e;font-size:11px;">O</span><span style="font-weight:700;color:${col};">$${fmt(bar.open)}</span>
        <span style="color:#8b949e;font-size:11px;">H</span><span style="font-weight:700;color:#3fb950;">$${fmt(bar.high)}</span>
        <span style="color:#8b949e;font-size:11px;">L</span><span style="font-weight:700;color:#f85149;">$${fmt(bar.low)}</span>
        <span style="color:#8b949e;font-size:11px;">C</span><span style="font-weight:700;color:${col};">$${fmt(bar.close)}</span>
        <span style="color:#8b949e;font-size:11px;">Δ</span><span style="font-weight:700;color:${col};">${up?'+':''}$${fmt(chg)} <span style="opacity:.75;font-size:10px;">(${up?'+':''}${(chg/bar.open*100).toFixed(2)}%)</span></span>
      </div>`;
    tt.style.display = 'block';
  });

  if (window._lwcRO) window._lwcRO.disconnect();
  window._lwcRO = new ResizeObserver(entries => {
    if (!_lwChart) return;
    for (const entry of entries) {
      const W = Math.round(entry.contentRect.width);
      const H = Math.round(entry.contentRect.height);
      if (W < 10 || H < 10) continue;
      _lwChart.applyOptions({ width: W, height: H });
      // Double rAF: LWC needs one full internal layout pass after applyOptions
      // before priceToCoordinate / timeToCoordinate return correct values.
      requestAnimationFrame(() => requestAnimationFrame(_drawMcOverlay));
    }
  });
  window._lwcRO.observe(el);
}

// Helper: compute the inter-bar spacing in seconds. Slim poll frames carry
// only the last bar, so cache the step from the last full payload and fall
// back to it (NOT a hardcoded 900s - that would draw the MC projection with
// 15m spacing on a 4h/1d chart after the first incremental poll).
let _lastStepSec = 900;
function _candleStepSec(candles) {
  if (!candles || candles.length < 2) return _lastStepSec;
  const n = candles.length;
  const step = _toLwcSec(candles[n - 1].t) - _toLwcSec(candles[n - 2].t);
  if (step > 0) _lastStepSec = step;
  return _lastStepSec;
}

// Build a forward-projected MC band series anchored on `lastTs`.
function _mkMcSeries(arr, lastTs, step) {
  return (arr || []).map((v, i) => ({ time: lastTs + i * step, value: v }));
}

// Apply MC bands + median path + cached band data. Shared by full-load
// and incremental-update paths; both rebuild the projection every time
// because the MC distribution shifts with each new analysis cycle.
function _applyMcProjection(mc, lastTs, step) {
  if (!mc || !mc.median_path) {
    // Partial payload (candles only) - clear any stale projection
    [_lwMcMed, _lwMcP75, _lwMcP25, _lwMcP90, _lwMcP10].forEach(s => {
      if (s) try { s.setData([]); } catch(_) {}
    });
    _mcBandData = null;
    return;
  }
  const p90 = _mkMcSeries(mc.p90_band    || mc.median_path, lastTs, step);
  const p75 = _mkMcSeries(mc.upper_band  || mc.median_path, lastTs, step);
  const p25 = _mkMcSeries(mc.lower_band  || mc.median_path, lastTs, step);
  const p10 = _mkMcSeries(mc.p10_band    || mc.median_path, lastTs, step);
  const med = _mkMcSeries(mc.median_path,                    lastTs, step);
  _lwMcMed.setData(med);
  _lwMcP75.setData(p75);
  _lwMcP25.setData(p25);
  _lwMcP90.setData(p90);
  _lwMcP10.setData(p10);
  _mcBandData = {
    p90, p75, p25, p10, med,
    paths: (mc.paths || []).slice(0, 30),
    step,
  };
}

function _setLwcData(d) {
  if (!_lwCandle || !_lwChart) return;
  const candles = d.candles;
  const mc      = d.mc;   // may be absent on type="partial" - handled below

  if (!candles || !candles.length) return;

  // Full-load: replace everything via setData. This is the initial chart
  // construction or a ticker/timeframe switch.
  const candleData = candles.map(c => ({
    time: _toLwcSec(c.t),
    open: c.o, high: c.h, low: c.l, close: c.c,
  }));
  _lwCandle.setData(candleData);

  _lwVol.setData(candles.map(c => ({
    time:  _toLwcSec(c.t),
    value: c.v,
    color: c.c >= c.o ? 'rgba(63,185,80,0.38)' : 'rgba(248,81,73,0.38)',
  })));

  // Seed the live-update watermarks so subsequent incremental polls know
  // which bars are already on the chart and whether older history just
  // expanded.
  _lastSeenCandleTime  = candleData[candleData.length - 1].time;
  _firstSeenCandleTime = candleData[0].time;

  const lastTs = _lastSeenCandleTime;
  const step   = _candleStepSec(candles);
  _applyMcProjection(mc, lastTs, step);

  // Only fit-content on full loads - incremental updates must preserve
  // the user's pan/zoom state.
  _lwChart.timeScale().fitContent();
  requestAnimationFrame(_drawMcOverlay);
}

// Called when the chart is already built for the same ticker/timeframe and
// the server just pushed a fresh poll. Strategy:
//   • For every candle in the payload whose time >= _lastSeenCandleTime
//     call series.update(). LWC semantics:
//        time === latest    -> updates the forming bar in place
//        time >  latest     -> appends as a new bar
//        time <  latest     -> would throw; we skip those
//   • The MC projection is forward-looking and small (<= n_forward+1
//     points); rebuilding it via setData is cheap and keeps it in sync
//     with the new "latest bar".
//   • We never call fitContent() here - that would jump the viewport on
//     every poll and destroy the user's zoom/scroll.
function _updateLwcIncremental(d) {
  if (!_lwCandle || !_lwChart) return;
  const candles = d.candles;
  if (!candles || !candles.length) return;

  let newMax = _lastSeenCandleTime;
  for (const c of candles) {
    const t = _toLwcSec(c.t);
    if (t < _lastSeenCandleTime) continue;          // history bar - already on chart
    _lwCandle.update({ time: t, open: c.o, high: c.h, low: c.l, close: c.c });
    _lwVol.update({
      time:  t,
      value: c.v,
      color: c.c >= c.o ? 'rgba(63,185,80,0.38)' : 'rgba(248,81,73,0.38)',
    });
    if (t > newMax) newMax = t;
  }
  _lastSeenCandleTime = newMax;

  // Refresh the MC projection so its time origin tracks the latest bar.
  const step = _candleStepSec(candles);
  _applyMcProjection(d.mc, _lastSeenCandleTime, step);

  // Redraw the canvas overlay (band fills + sample paths) in coordinates.
  requestAnimationFrame(_drawMcOverlay);
}

function _createMcOverlay(container) {
  const old = document.getElementById('lwc-mc-overlay');
  if (old) old.remove();
  const canvas = document.createElement('canvas');
  canvas.id = 'lwc-mc-overlay';
  canvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:1;';
  container.appendChild(canvas);
  _mcOverlayCanvas = canvas;
}

function _drawMcOverlay() {
  if (!_mcOverlayCanvas || !_mcBandData || !_lwChart || !_lwMcP90 || !_lwMcP10) return;

  const container = document.getElementById('lwc-container');
  if (!container) return;
  const W   = container.offsetWidth;
  const H   = container.offsetHeight;
  const dpr = window.devicePixelRatio || 1;
  const cvs = _mcOverlayCanvas;
  cvs.width  = W * dpr;
  cvs.height = H * dpr;
  cvs.style.width  = W + 'px';
  cvs.style.height = H + 'px';

  const ctx = cvs.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const ts = _lwChart.timeScale();

  // Convert one MC data point -> {x, y} in canvas pixels, or null if off-screen
  const toPx = (series, time, value) => {
    const x = ts.timeToCoordinate(time);
    const y = series.priceToCoordinate(value);
    return (x != null && y != null) ? { x, y } : null;
  };

  // Build array of {x,y} for a band, skipping off-screen nulls at edges
  const bandPts = (data, series) => {
    const pts = [];
    for (const pt of data) {
      const p = toPx(series, pt.time, pt.value);
      if (p) pts.push(p);
    }
    return pts;
  };

  // Draw filled area between topPts and botPts
  const fillBand = (topPts, botPts, color) => {
    if (topPts.length < 2 || botPts.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(topPts[0].x, topPts[0].y);
    for (let i = 1; i < topPts.length; i++) ctx.lineTo(topPts[i].x, topPts[i].y);
    for (let i = botPts.length - 1; i >= 0; i--) ctx.lineTo(botPts[i].x, botPts[i].y);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  };

  // Draw border line for a band edge
  const strokeLine = (pts, color, width, dash) => {
    if (pts.length < 2) return;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    if (dash) ctx.setLineDash(dash); else ctx.setLineDash([]);
    ctx.stroke();
    ctx.restore();
  };

  const { p90, p75, p25, p10, paths, step } = _mcBandData;

  const p90Pts = bandPts(p90, _lwMcP90);
  const p10Pts = bandPts(p10, _lwMcP10);
  const p75Pts = bandPts(p75, _lwMcP75);
  const p25Pts = bandPts(p25, _lwMcP25);

  // ① Outer fill P10-P90 (very light)
  fillBand(p90Pts, p10Pts, 'rgba(63,185,80,0.07)');

  // ② Faint individual MC simulation paths
  if (paths.length && p90.length) {
    ctx.lineWidth = 0.7;
    for (const path of paths) {
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < path.length; i++) {
        const p = toPx(_lwMcMed, p90[0].time + i * step, path[i]);
        if (!p) continue;
        if (!started) { ctx.moveTo(p.x, p.y); started = true; }
        else ctx.lineTo(p.x, p.y);
      }
      ctx.strokeStyle = 'rgba(88,166,255,0.05)';
      ctx.setLineDash([]);
      ctx.stroke();
    }
  }

  // ③ Inner fill P25-P75 (medium green)
  fillBand(p75Pts, p25Pts, 'rgba(63,185,80,0.18)');

  // ④ P75 / P25 border lines (solid green)
  strokeLine(p75Pts, 'rgba(63,185,80,0.65)', 1, []);
  strokeLine(p25Pts, 'rgba(63,185,80,0.65)', 1, []);

  // ⑤ P90 / P10 outer dashed border lines
  strokeLine(p90Pts, 'rgba(63,185,80,0.28)', 1, [5, 4]);
  strokeLine(p10Pts, 'rgba(63,185,80,0.28)', 1, [5, 4]);
}

// Click first point -> click second point -> shows Δprice, Δ%, bar count.
// Uses LightweightCharts crosshair position via the subscribeClick API.

let _measureActive  = false;
let _measurePoint1  = null;  // { time, price }
let _measureOverlay = null;  // canvas drawn on top of the chart

function toggleMeasureTool() {
  _measureActive = !_measureActive;
  _measurePoint1 = null;
  const btn     = document.getElementById('measure-btn');
  const readout = document.getElementById('measure-readout');
  if (_measureActive) {
    btn.style.background   = 'rgba(210,153,34,.15)';
    btn.style.borderColor  = 'var(--amber)';
    btn.style.color        = 'var(--amber)';
    btn.textContent        = ' Click start...';
    if (readout) { readout.textContent = 'Click a candle to set start point'; readout.style.display = 'inline'; }
    _attachMeasureListener();
  } else {
    btn.style.background  = 'var(--surface2)';
    btn.style.borderColor = 'var(--border)';
    btn.style.color       = 'var(--muted)';
    btn.textContent       = ' Measure';
    if (readout) readout.style.display = 'none';
    _clearMeasureOverlay();
    _detachMeasureListener();
  }
}

let _measureClickHandler = null;
function _attachMeasureListener() {
  if (!_lwChart) return;
  _measureClickHandler = (param) => {
    if (!param.point || !_lwCandle) return;
    const price = _lwCandle.coordinateToPrice(param.point.y);
    const time  = param.time;
    if (!price || !time) return;

    if (!_measurePoint1) {
      _measurePoint1 = { time, price };
      const btn = document.getElementById('measure-btn');
      if (btn) btn.textContent = ' Click end...';
      const r = document.getElementById('measure-readout');
      if (r) r.textContent = `Start: $${price.toFixed(2)}, Click end point`;
    } else {
      // Second click - compute measurement
      const p1 = _measurePoint1;
      const p2 = { time, price };
      const dp  = p2.price - p1.price;
      const pct = (dp / p1.price) * 100;

      // Candle count from time (approximate using series data)
      let bars = '-';
      try {
        if (_lastChartData?.ohlcv) {
          const ts = _lastChartData.ohlcv.map(c => c.time);
          const i1 = ts.findIndex(t => t >= p1.time);
          const i2 = ts.findIndex(t => t >= p2.time);
          if (i1 >= 0 && i2 >= 0) bars = Math.abs(i2 - i1);
        }
      } catch(_) {}

      const sign  = dp >= 0 ? '+' : '';
      const col   = dp >= 0 ? '#3fb950' : '#f85149';
      const r     = document.getElementById('measure-readout');
      if (r) {
        r.style.display = 'inline';
        r.innerHTML = ` ${sign}$${dp.toFixed(2)} (${sign}${pct.toFixed(2)}%), ${bars} bars`;
        r.style.color = col;
      }
      const btn = document.getElementById('measure-btn');
      if (btn) btn.textContent = ' Reset';

      _measurePoint1 = null; // ready for next measurement
    }
  };
  _lwChart.subscribeClick(_measureClickHandler);
}

function _detachMeasureListener() {
  if (_lwChart && _measureClickHandler) {
    _lwChart.unsubscribeClick(_measureClickHandler);
    _measureClickHandler = null;
  }
}

function _clearMeasureOverlay() {
  if (_measureOverlay) {
    _measureOverlay.remove();
    _measureOverlay = null;
  }
}

  window.buildCharts = buildCharts;
  window.toggleMeasureTool = toggleMeasureTool;
})();
