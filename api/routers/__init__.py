"""Per-domain API routers."""

from . import ai, news, options, pages, portfolio, scanner, signal, store

ALL_ROUTERS = [
    pages.router,
    signal.router,
    store.router,
    portfolio.router,
    options.router,
    scanner.router,
    news.router,
    ai.router,
]
