// ──────────────────────────────────────────────────────────────────────────
// /static/js/index.js — front-end module barrel.
//
// Loaded by templates/dashboard.html as the entry point for the dashboard's
// external JavaScript layer. Acts as a manifest of the modules that live
// under /static/js/ so a future maintainer can find every front-end file
// without grepping the HTML.
//
// PHASE 2/3/4/5 STATUS (refactor in progress)
// ────────────────────────────────────────────
//   ✅  scanner.js                 — extracted (Phase 2)
//   ✅  tabs/options.js            — extracted (Phase 3 — Unusual Activity)
//   ✅  tabs/market-structure.js   — error boundaries + retry (Phase 4)
//   ✅  tabs/sentiment.js          — live news feed + sector Cramer (Phase 5)
//   ⏳  tabs/news.js               — pending (Phase 6)
//   ⏳  tabs/zone.js               — pending (later)
//   ⏳  right-panel.js             — pending (Phase 7)
//   ⏳  ws.js / api.js / state.js  — pending (Phase 7)
//   ⏳  portfolio.js               — pending (later)
//
// The blocks not yet extracted still live as classic <script> blocks
// inside dashboard.html. As each tab is touched in Phases 3-7 we'll
// pull its functions out into the corresponding file under tabs/.
//
// Until then, this file is intentionally side-effect free; the actual
// page bootstrap continues to run from the inline <script> blocks.
// ──────────────────────────────────────────────────────────────────────────

// Marker on window so e2e tests and the browser console can see the
// refactor is partway in.
window.__mc_trader_modular__ = {
  version: 'phase-5',
  extracted: ['scanner.js', 'tabs/options.js', 'tabs/market-structure.js', 'tabs/sentiment.js'],
  inline: ['main', 'portfolio'],
};
