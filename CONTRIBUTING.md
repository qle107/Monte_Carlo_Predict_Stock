# Contributing to mc-trader

Thank you for contributing. This project is a self-hosted Monte Carlo stock research API. Bug fixes, math improvements, new models, docs, and tests are welcome.

## Quick start

```bash
# 1. Fork and clone
git clone https://github.com/your-fork/Monte_Carlo_Predict_Stock.git
cd Monte_Carlo_Predict_Stock

# 2. Create a virtual environment (Python 3.10-3.12)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install all dependencies (runtime + dev)
pip install -r requirements.txt

# 4. Copy the example env and fill in your keys
cp .env.example .env
```

## Before you submit a PR

Run the full check suite locally so CI doesn't surprise you:

```bash
# Lint
ruff check .

# Format check
ruff format --check .

# Type check (informational; failures don't block merge)
mypy core/ api/ config.py

# Tests with coverage
pytest -v
```

All four commands must pass (mypy is advisory).

## Mathematical changes

Any change to a simulation model, statistical estimator, or financial formula must include:

1. **A docstring** in the changed function that states the mathematical justification. Reference the relevant equation (e.g. "Itô lemma: E[exp(σZ)] = exp(½σ²)").
2. **A regression test** in `tests/test_montecarlo.py` or `tests/test_hurst.py` that locks in the corrected behaviour. Use `np.random.default_rng(seed)` for reproducibility.
3. **An update to `docs/math.md`** if you introduce a new model or estimator.

See `docs/math.md` for the full derivations of all existing models.

## Code style

- Line length: 110 characters (enforced by ruff).
- Use `from __future__ import annotations` at the top of every new module.
- Greek letters in docstrings and comments are intentional (α, σ, μ, κ ...); don't replace them with ASCII.
- All new public functions must have a docstring with at least one sentence of explanation.

## Pull request checklist

- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `pytest -v` passes (all existing tests still green)
- [ ] New tests added for any changed behaviour
- [ ] Docstrings updated / added
- [ ] `docs/math.md` updated if a formula changed
- [ ] No API keys or secrets in any committed file

## Reporting bugs

Use the **Bug report** issue template. Include the Python version, OS, and a minimal reproduction case.

## Proposing new models

Open a **Feature request** issue first to discuss the mathematical approach before writing code. Reference a published paper or standard textbook derivation.
