/* Cross-page navigation strip + contract-page opener.
 * Injected on every page that includes this script. */
(function () {
  'use strict';

  const LINKS = [
    { href: '/', label: 'Dashboard' },
    { href: '/flow', label: 'Flow' },
    { href: '/contract', label: 'Contract' },
    { href: '/portfolio', label: 'Portfolio' },
  ];

  function inject() {
    if (document.getElementById('mc-page-nav')) return;
    const here = location.pathname.replace(/\/+$/, '') || '/';
    const nav = document.createElement('div');
    nav.id = 'mc-page-nav';
    nav.style.cssText =
      'position:fixed;bottom:14px;left:14px;z-index:9999;display:flex;gap:2px;' +
      'background:rgba(13,17,23,.92);border:1px solid rgba(139,148,158,.25);' +
      'border-radius:8px;padding:3px;backdrop-filter:blur(6px);' +
      'font-family:inherit;font-size:11px;box-shadow:0 4px 16px rgba(0,0,0,.4);';
    nav.innerHTML = LINKS.map((l) => {
      const active = l.href === here;
      return '<a href="' + l.href + '" style="padding:5px 11px;border-radius:6px;text-decoration:none;' +
        'font-weight:' + (active ? '700' : '500') + ';' +
        'color:' + (active ? '#58a6ff' : '#7d8b99') + ';' +
        'background:' + (active ? 'rgba(88,166,255,.12)' : 'transparent') + ';"' +
        ' onmouseover="if(this.style.color!==\'rgb(88, 166, 255)\')this.style.color=\'#e6edf3\'"' +
        ' onmouseout="if(this.style.background===\'transparent\')this.style.color=\'#7d8b99\'">' +
        l.label + '</a>';
    }).join('');
    document.body.appendChild(nav);
  }

  /* Open the contract tracker in a NEW TAB only - never redirect the
   * current page. Note: window.open returns null in Chrome when 'noopener'
   * is used even on success, so there is deliberately no fallback redirect. */
  window.openContractPage = function (ticker, expiry, strike, optionType) {
    const p = new URLSearchParams({
      ticker: (ticker || '').toUpperCase(),
      expiry: expiry || '',
      strike: strike != null ? String(strike) : '',
      type: (optionType || 'call').toLowerCase(),
    });
    window.open('/contract?' + p.toString(), '_blank', 'noopener');
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', inject);
  } else {
    inject();
  }
})();
