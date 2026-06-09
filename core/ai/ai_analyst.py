"""AI Analyst: aggregate API data and call Claude for a structured trade view."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Model fallbacks if the requested name is unavailable.
_MODEL_FALLBACKS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
]

# Drop bulky keys from the prompt payload.
_DROP_KEYS = {"candles", "paths", "_mc_paths_full", "mc_paths", "articles_raw", "history"}


def _claude_max_tokens() -> int:
    try:
        return max(1024, min(int(os.getenv("CLAUDE_MAX_TOKENS", "4096")), 16384))
    except ValueError:
        return 4096


def _slim(obj: Any, max_list: int = 15, depth: int = 0) -> Any:
    """Recursively trim payloads: drop heavy keys, cap list lengths, round floats."""
    if depth > 8:
        return None
    if isinstance(obj, dict):
        return {
            k: _slim(v, max_list, depth + 1)
            for k, v in obj.items()
            if k not in _DROP_KEYS
        }
    if isinstance(obj, (list, tuple)):
        return [_slim(v, max_list, depth + 1) for v in list(obj)[:max_list]]
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


async def _best_effort(name: str, coro, timeout: float) -> tuple[str, Any]:
    """Run one data-source coroutine; return (name, data|None)."""
    try:
        return name, await asyncio.wait_for(coro, timeout=timeout)
    except Exception as e:  # data sources are best-effort
        logger.warning("[ai_analyst] %s unavailable: %s", name, e)
        return name, {"_error": str(e)[:300]}


async def gather_context(ticker: str, last_result: dict | None) -> dict:
    """Collect analysis context from API data sources."""
    loop = asyncio.get_running_loop()
    t = ticker.upper().strip()

    from core.news.fear_greed import fetch_fear_greed
    from core.news.macro import fetch_macro_indicators
    from core.news.news_aggregator import fetch_news
    from core.options.options_flow import fetch_options_flow, scan_unusual_options
    from core.sentiment import get_sentiment

    async def _gex():
        flow = await loop.run_in_executor(None, fetch_options_flow, t)
        return flow.to_dict() if hasattr(flow, "to_dict") else flow

    async def _unusual():
        return await loop.run_in_executor(
            None,
            lambda: scan_unusual_options(
                tickers=[t],
                min_volume=50,
                min_oi=25,
                vol_oi_threshold=2.0,
                otm_pct=0.05,
                max_dte=45,
                max_concurrent=2,
                top_n=15,
                min_premium=10_000.0,
            ),
        )

    async def _macro():
        return await loop.run_in_executor(None, fetch_macro_indicators, False)

    tasks = [
        _best_effort("options_gex", _gex(), 20.0),
        _best_effort("unusual_options", _unusual(), 30.0),
        _best_effort("news", fetch_news(t, 15), 15.0),
        _best_effort("sentiment", get_sentiment(t), 15.0),
        _best_effort("fear_greed", fetch_fear_greed(), 10.0),
        _best_effort("macro", _macro(), 15.0),
    ]
    results = dict(await asyncio.gather(*tasks))

    # Cached /api/signal result when it matches the requested ticker.
    technical = None
    if last_result and str(last_result.get("ticker", "")).upper() == t:
        technical = last_result

    context = {
        "ticker": t,
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "technical_analysis": technical,
        **results,
    }
    return _slim(context)


_SYSTEM_PROMPT = """You are a quantitative options analyst.
You receive a JSON payload from a stock analysis API for one ticker:
Monte Carlo simulation results, technical indicators, market regime, trade-setup gates,
supply/demand zones, expected move (IV vs RV), options gamma exposure (GEX), max pain,
call/put walls, unusual options flow (sweeps/blocks), news headlines, social sentiment,
CNN Fear & Greed, and macro indicators. Some sections may contain "_error" - ignore those.

Your job:
1. Synthesize ALL the evidence into directional probabilities for the NEXT 5 TRADING DAYS.
2. Infer what the crowd / smart money is positioning for (from unusual flow, GEX, sentiment, news).
3. Suggest concrete plays for next week (options strategies with strikes/expiries near the data
   provided, plus a simpler shares alternative), with entry, target, stop, and rough
   probability-of-profit estimates. Be specific, not generic.
4. Name the EXACT option contracts you would buy/sell right now (ticker, expiry, strike,
   call/put, approximate premium from the chain data) - the single best new opening trades.
5. Flag the biggest risks/catalysts (earnings, macro prints, max-pain pinning, IV crush).

Respond with ONLY a single JSON object, no markdown fences, matching exactly this schema:
{
  "probabilities": {"bullish_pct": 0-100, "bearish_pct": 0-100, "sideways_pct": 0-100, "rationale": "1-3 sentences"},
  "confidence_pct": 0-100,
  "market_summary": "3-6 sentence synthesis of the overall picture",
  "what_crowd_is_watching": ["bullet", ...],
  "key_levels": {"support": [numbers], "resistance": [numbers], "gamma_magnet": number or null},
  "top_contract_picks": [
    {"contract": "e.g. MRVL 2026-06-26 270C", "action": "buy|sell", "approx_price": "per-contract premium",
     "size_hint": "e.g. 1-2% of account", "why": "one sentence"}
  ],
  "suggested_plays": [
    {"name": "short label", "direction": "bullish|bearish|neutral", "instrument": "e.g. call debit spread / shares / cash-secured put",
     "details": "strikes, expiry, approx cost", "entry": "condition/price", "target": "price/level", "stop": "price/condition",
     "est_pop_pct": 0-100, "rationale": "why this play fits the data"}
  ],
  "next_week_position": "one paragraph: the single position you'd actually hold into next week and how you'd manage it",
  "risks": ["bullet", ...],
  "disclaimer": "one sentence"
}
The three probabilities must sum to 100. Base every claim on the supplied data; if data is
missing, say so in the rationale rather than inventing numbers."""


_CHAT_PROMPT_INTRO = """You are a quantitative options analyst.
Below is a JSON payload from a stock analysis API for one ticker:
Monte Carlo simulation results, technical indicators, market regime, trade-setup gates,
supply/demand zones, expected move (IV vs RV), options gamma exposure (GEX), max pain,
call/put walls, unusual options flow (sweeps/blocks), news headlines, social sentiment,
CNN Fear & Greed, and macro indicators. Sections containing "_error" were unavailable - ignore them.

Your job:
1. Synthesize ALL the evidence into directional probabilities for the NEXT 5 TRADING DAYS.
2. Infer what the crowd / smart money is positioning for (from unusual flow, GEX, sentiment, news).
3. Suggest concrete plays for next week (options strategies with strikes/expiries near the data
   provided, plus a simpler shares alternative), with entry, target, stop, and rough
   probability-of-profit estimates. Be specific, not generic.
4. Name the EXACT option contracts you would buy/sell right now (ticker, expiry, strike,
   call/put, approximate premium from the chain data) - the single best new opening trades.
5. Flag the biggest risks/catalysts (earnings, macro prints, max-pain pinning, IV crush).

Timing rules (important):
- Check every catalyst's DATE against the option expiries you pick. Only choose an expiry that
  CONTAINS the catalyst you are playing (e.g. an index-inclusion market-on-close buy is useless
  to a spread that expires the day before it).
- If the main catalyst falls OUTSIDE the next 5 trading days, give BOTH: at least one play for
  next week itself AND one dated event play whose expiry covers the catalyst, each labeled with
  its window in the "name" field.
- If a "MY CONTEXT" section appears after the data, treat it as the trader's own situation/thesis
  (positions, account size, known catalysts, bias). Tailor the plays to it, and explicitly say in
  the rationale where the data agrees or disagrees with the trader's view.
- If the trader asks follow-up questions later in the chat, ALWAYS end your reply by re-outputting
  the FULL updated ```json block reflecting the new conclusions.

Output ONLY a single ```json code block matching EXACTLY this schema - no text before or after it.
Keep every string concise (the reply must not get cut off): max 3 suggested plays, max 3 contract
picks, max 6 bullets per list.

{
  "probabilities": {"bullish_pct": 0-100, "bearish_pct": 0-100, "sideways_pct": 0-100, "rationale": "1-2 sentences"},
  "confidence_pct": 0-100,
  "market_summary": "3-5 sentence synthesis of the overall picture",
  "what_crowd_is_watching": ["short bullet", "..."],
  "key_levels": {"support": [numbers], "resistance": [numbers], "gamma_magnet": number or null},
  "top_contract_picks": [
    {"contract": "e.g. MRVL 2026-06-26 270C", "action": "buy|sell", "approx_price": "per-contract premium from chain data",
     "size_hint": "e.g. 1-2% of account", "why": "one sentence"}
  ],
  "suggested_plays": [
    {"name": "short label", "direction": "bullish|bearish|neutral", "instrument": "e.g. call debit spread / shares / cash-secured put",
     "details": "strikes, expiry, approx cost", "entry": "condition/price", "target": "price/level", "stop": "price/condition",
     "est_pop_pct": 0-100, "rationale": "1-2 sentences"}
  ],
  "next_week_position": "2-3 sentences: the single position you'd actually hold into next week and how you'd manage it",
  "risks": ["short bullet", "..."],
  "disclaimer": "one sentence"
}

The three probabilities must sum to 100. Base every claim on the supplied data; if data is
missing, say so in the rationale rather than inventing numbers. Valid JSON only: no trailing
commas, no stray braces, escape quotes inside strings.

=== ANALYSIS DATA (JSON) ===
"""


def _payload_and_sources(context: dict) -> tuple[str, list[str], list[str]]:
    """JSON-encode the context (with size cap) and list ok/failed sources."""
    sources_ok = [
        k
        for k, v in context.items()
        if k not in ("ticker", "as_of_utc") and v and not (isinstance(v, dict) and "_error" in v)
    ]
    sources_failed = [k for k, v in context.items() if isinstance(v, dict) and "_error" in v]
    payload_json = json.dumps(context, default=str, ensure_ascii=False)
    # Keep the prompt within a sane budget (~150k chars ≈ 40k tokens).
    if len(payload_json) > 150_000:
        payload_json = payload_json[:150_000] + "\n...[truncated]"
    return payload_json, sources_ok, sources_failed


async def build_prompt(ticker: str, last_result: dict | None) -> dict:
    """Build a copy-paste prompt for the Claude app."""
    t = ticker.upper().strip()
    context = await gather_context(t, last_result)
    payload_json, sources_ok, sources_failed = _payload_and_sources(context)
    prompt = _CHAT_PROMPT_INTRO + payload_json
    return {
        "ticker": t,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "sources_used": sources_ok,
        "sources_failed": sources_failed,
    }


def _drop_stray_closers(s: str) -> str:
    """Remove stray '}' / ']' that would close the root object before the end
    (e.g. the model emits '"summary": "..."},' mid-object)."""
    out: list[str] = []
    depth = 0
    in_str = False
    escaped = False
    for i, ch in enumerate(s):
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            if depth <= 1 and s[i + 1 :].strip():
                continue  # stray: would end the root early
            depth -= 1
        out.append(ch)
    return "".join(out)


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response."""
    if not text:
        return None
    # Prefer a fenced ```json block (chat responses have prose around it).
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    cleaned = text.strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_drop_stray_closers(candidate))
    except json.JSONDecodeError:
        return None


async def _call_anthropic(payload_json: str, model: str) -> tuple[str, str]:
    """Call the Messages API; returns (text, model_used). Tries fallbacks on unknown model."""
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file. "
            "Create a key at https://console.anthropic.com/settings/keys "
            "(note: a Claude.ai subscription does not include an API key - "
            "the API is billed separately)."
        )

    headers = {
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    candidates = [model] + [m for m in _MODEL_FALLBACKS if m != model]
    last_err: str = "unknown error"

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0)) as client:
        for m in candidates:
            body = {
                "model": m,
                "max_tokens": _claude_max_tokens(),
                "system": _SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Here is the aggregated analysis data as JSON. "
                            "Analyze it and respond with the JSON schema from your instructions.\n\n"
                            + payload_json
                        ),
                    }
                ],
            }
            resp = await client.post(ANTHROPIC_URL, headers=headers, json=body)
            if resp.status_code == 200:
                data = resp.json()
                text = "".join(
                    blk.get("text", "") for blk in data.get("content", []) if blk.get("type") == "text"
                )
                return text, m
            # Unknown model -> try next candidate; anything else -> raise
            try:
                err = resp.json().get("error", {})
            except Exception:
                err = {}
            msg = str(err.get("message", resp.text[:300]))
            last_err = f"{resp.status_code}: {msg}"
            if resp.status_code in (400, 404) and "model" in msg.lower():
                logger.warning("[ai_analyst] model %s unavailable (%s); trying fallback", m, msg)
                continue
            if resp.status_code == 401:
                raise RuntimeError(
                    "Anthropic API rejected the key (401). Check ANTHROPIC_API_KEY in .env."
                )
            if resp.status_code == 429:
                raise RuntimeError("Anthropic API rate limit hit (429). Wait a minute and retry.")
            raise RuntimeError(f"Anthropic API error {last_err}")

    raise RuntimeError(f"No usable Claude model found. Last error: {last_err}")


async def run_ai_analysis(ticker: str, last_result: dict | None) -> dict:
    """End-to-end: gather context -> ask Claude -> return structured result."""
    t = ticker.upper().strip()
    context = await gather_context(t, last_result)
    payload_json, sources_ok, sources_failed = _payload_and_sources(context)

    model = os.getenv("CLAUDE_MODEL", "claude-opus-4-8").strip()
    text, model_used = await _call_anthropic(payload_json, model)
    parsed = _extract_json(text)

    return {
        "ticker": t,
        "model": model_used,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis": parsed,  # structured JSON (None if parsing failed)
        "raw": text if parsed is None else None,  # raw text fallback for the UI
        "sources_used": sources_ok,
        "sources_failed": sources_failed,
    }
