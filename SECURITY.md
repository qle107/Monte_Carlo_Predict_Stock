# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

**Please do not file public GitHub issues for security vulnerabilities.**

To report a vulnerability:

1. Email the maintainer at the address on the GitHub profile, or
2. Use [GitHub's private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability) on this repo.

Please include:
- A description of the vulnerability and its potential impact
- Steps to reproduce (proof-of-concept or detailed description)
- Affected version(s)
- Any suggested mitigations

We aim to respond within **5 business days** and will work to release a fix
within **30 days** for high-severity issues.

## Scope

This project is a self-hosted local dashboard. Key security considerations:

- **API keys**: Set `API_KEY` in `.env` to gate all mutation endpoints behind
  an `X-Api-Key` header. Without it, anyone who can reach port 8000 can call
  all endpoints.
- **Host binding**: Default `HOST=127.0.0.1` binds to localhost only.
  Setting `HOST=0.0.0.0` exposes the server to the network; use a reverse
  proxy (nginx / Caddy) and set `API_KEY` before doing so.
- **Alpaca keys**: Never enter live-trading Alpaca keys in the browser-based
  portfolio tracker. Use paper-trading keys only. Keys are held in-memory
  (not `localStorage`) and are lost on page refresh.
- **FRED / Polygon / X keys**: Stored only in `.env`, never logged or
  transmitted to any third party.

## Known Limitations

- No built-in TLS — use a reverse proxy for HTTPS.
- No authentication beyond the optional `API_KEY` header.
- SQLite signal store has no access control beyond filesystem permissions.
