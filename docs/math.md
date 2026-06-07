# Mathematical Foundations

This document derives the key results used in the Monte Carlo simulation engine.
All notation follows the conventions in the source code.

---

## 1. Geometric Brownian Motion and the Itô / Jensen Correction

### Model

Standard GBM in continuous time:

```
dS = μ S dt + σ S dW
```

where `W` is a standard Brownian motion, `μ` is the instantaneous drift, and `σ` is the volatility.

### Discrete log-return path

For a time step `Δt = 1` (one bar), the exact solution is:

```
S_{t+1} = S_t · exp( (μ − ½σ²) + σ Z_t )       Z_t ~ N(0,1)
```

The `½σ²` term is the **Itô / Jensen correction**. Without it:

```
E[ exp(μ + σ Z) ] = exp( μ + ½σ² )              [Jensen's inequality for exp]
```

So naïvely using `log_drift = μ` (instead of `μ − ½σ²`) causes every path to drift upward by `½σ²` per step. Over `n` steps this accumulates to a multiplicative bias of `exp(n · ½σ²)`.

At `σ = 0.02`, `n = 10`:  bias `≈ exp(0.002) − 1 ≈ 0.2%` - visible on a chart.

### Implementation

```python
returns = (drift - 0.5 * sigma**2) + sigma * eps   # all non-microstructure models
log_ret = (ctx.final_drift - 0.5 * sigma_eff**2) + grav + innov   # microstructure
```

Paths are built with `exp(cumsum(log_returns))`, which is equivalent to `cumprod(exp(r_i))` but numerically stabler.

---

## 2. Merton Jump-Diffusion: Correct Log-Return Compensator

### Model

Merton (1976) adds a compound Poisson jump process to GBM:

```
dS/S = (μ − λκ) dt + σ dW + J dN
```

where:
- `N` is a Poisson process with intensity `λ` (expected jumps per bar)
- `J ~ N(μ_J, σ_J²)` is the log-jump size (so the price jump multiplier is `e^J`)
- `κ = E[e^J − 1]` is the **jump compensator** - the drift adjustment that keeps the process a martingale under the risk-neutral measure

### Computing κ correctly

Because `J ~ N(μ_J, σ_J²)`:

```
E[e^J] = exp(μ_J + ½σ_J²)        [moment-generating function of Normal]
κ = E[e^J − 1] = exp(μ_J + ½σ_J²) − 1
```

The effective log-drift per step (combining Itô and Merton) is:

```
drift_eff = drift − λ·κ
log_ret_t = (drift_eff − ½σ²) + σ·Z_t  +  Σ J_i   (for each Poisson jump)
```

### Why the old compensator was wrong

The old code used `drift_eff = drift − λ·μ_J`. This is the correct compensator for the **arithmetic** price model `S_{t+1} = S_t · (1 + r_t)` but not for the **log-return** / `exp()` path model. The error is:

```
λ·κ − λ·μ_J = λ · (exp(μ_J + ½σ_J²) − 1 − μ_J)
             ≈ λ · ½σ_J²      (for small μ_J, σ_J)
```

At `λ=0.05`, `σ_J = 5σ = 0.10`: missing bias `≈ 0.05 · 0.005 = 0.00025` per step, or `0.0025` over 10 steps - detectable without noise.

### Implementation

```python
kappa = float(np.exp(jump_mean + 0.5 * sigma_jump**2) - 1.0)
drift_eff = drift - jump_intensity * kappa
```

**Implementation note (Bernoulli approximation).** Jumps are drawn as a
Bernoulli mask (`rng.random() < λ`, at most one jump per bar) rather than a
true Poisson count. The exact Bernoulli compensator would be `log(1 + λκ)`;
the constant-λ branch uses `λκ`, which differs at `O((λκ)²)` — negligible for
the configured `λ ≤ 0.06`. (The Hawkes branch below uses the exact form.)

### Self-exciting (Hawkes) jump intensity

Constant λ implies a jump today does not change the odds of a jump tomorrow —
contradicting the well-documented clustering of large moves. When enough
history is available (≥ 40 bars, ≥ 8 large-move events), the jump intensity
is made **self-exciting** (Hawkes 1971):

```
λ_t = μ + Σ_{jumps i < t} α · exp(−β·(t − t_i))
```

Each jump kicks the intensity up by `α`; the kick decays at rate `β`.
The **branching ratio** `n = α/β` is the expected number of "aftershock"
jumps triggered per jump (stability requires `n < 1`).

`(μ, α, β)` are fitted by MLE in `core.hawkes` on large-move events
(`|return| > 1.5σ_rolling`), in bar-index time units. The fitted process is
then **re-anchored** to the engine's target jump rate:

```
λ̄ = μ / (1 − n)                       [stationary mean of a Hawkes process]
μ' = λ_target · (1 − n)               [so that λ̄' = λ_target]
```

keeping `(α, β)` — and therefore the cluster structure `n` — unchanged
(`n` is scale-free). The *current* excitation state carries over in relative
terms: if intensity now sits at `2×` its historical mean, the simulation
starts at `2× λ_target`.

**Discretisation.** The simulation runs per-bar with excitation update
`E_{t+1} = (E_t + α·1[jump])·e^{−β}`, so a jump's lifetime excitation is
`α·Σ_{k≥1} e^{−βk} = α·d/(1−d)` with `d = e^{−β}` — not the continuous
`α/β`. The anchoring therefore uses the **discrete-consistent** branching
ratio `n_disc = α·d/(1−d)` (clipped to ≤ 0.9); using `α/β` would leave the
simulated mean jump rate ≈15% off target at typical `β ≈ 0.4`.

Per step the jump probability is `p_t = 1 − exp(−λ_t)` (with `λ_t` capped at
1.0), and the drift uses the **exact per-step compensator**, per path:

```
E[e^{jump} | p_t] = (1 − p_t) + p_t·E[e^J] = 1 + p_t·κ
log_ret_t = (drift − log(1 + p_t·κ) − ½σ²) + σZ_t + 1[jump]·J_t
```

so `E[S_{t+1}/S_t] = exp(drift)` holds exactly, conditional on the intensity
path. Excitation update: `E_{t+1} = (E_t + α·1[jump]) · e^{−β}`.

---

## 3. Student-t Innovations

### Motivation

Heavy-tailed distributions are observed empirically in equity returns (excess kurtosis > 0). The Student-t with `df` degrees of freedom has excess kurtosis `6/(df−4)` for `df > 4`.

### Fitting `df` from kurtosis

Given empirical excess kurtosis `K`:

```
K = 6 / (df − 4)   →   df = 4 + 6/K
```

Clipped to `[4.5, 30]` for numerical stability. At `df=4.5`, tails are very heavy; `df=30 ≈ Normal`.

### Variance normalisation

Raw Student-t samples have variance `df/(df−2)`. To restore unit variance (so `σ` retains its interpretation):

```python
raw = rng.standard_t(df=df, size=(n_sim, n_steps))
raw *= np.sqrt((df - 2.0) / df)
```

---

## 4. GJR-GARCH(1,1) Likelihood

### Model

```
σ²_t = ω + (α + γ · 1[ε_{t−1} < 0]) · ε²_{t−1} + β · σ²_{t−1}
ε_t  = σ_t · z_t,    z_t ~ N(0,1)
```

The `γ` term is the **leverage effect** (Glosten, Jagannathan & Runkle 1993):
negative shocks raise next-period volatility by `α + γ` while positive shocks
raise it only by `α`. This asymmetry is consistently significant in equity
markets - comparative studies rank GJR (and EGARCH) above symmetric
GARCH(1,1) under both MSE and QLIKE loss. `γ = 0` recovers plain GARCH(1,1).

Under symmetric innovations `E[1[ε<0]·ε²] = σ²/2`, so:

- Stationarity requires `α + γ/2 + β < 1`
- Unconditional variance `= ω / (1 − α − γ/2 − β)`

The engine targets the long-run variance with
`ω = (1 − α − γ/2 − β) · σ²_LR`.

### Maximum likelihood estimation

The (quasi-)log-likelihood is:

```
ℓ(ω, α, γ, β) = −½ Σ_t [ log(2π σ²_t) + ε²_t / σ²_t ]
```

Optimised with Nelder-Mead (`scipy.optimize.minimize`), max 600 iterations,
with constraints `ω > 0`, `α, γ, β ≥ 0`, `α + γ/2 + β < 0.999`. Results are
cached for 5 minutes (keyed on the last 90 returns) to avoid re-running the
optimiser on every poll cycle.

---

## 5. Stationary Bootstrap (Politis & Romano, 1994)

### Why not i.i.d. resampling?

Naive i.i.d. bootstrap draws returns independently, destroying all serial autocorrelation. Volatility clustering (GARCH-like behaviour) and momentum measured by `ACF(1) > 0` both vanish → systematically underestimates tail risk under trending regimes.

### Block bootstrap

Draw contiguous blocks of returns of length `L`. Fixed block length (Künsch 1989) breaks at fixed boundaries, introducing an artificial jump in correlation at multiples of `L`.

### Stationary bootstrap

Politis & Romano (1994) randomise the block length: `L ~ Geometric(p)` with `E[L] = 1/p = b`. The starting position of each block is drawn uniformly from `{0, …, N−1}` (with wraparound). The process restarts (picks a new starting index) with probability `p` at each step.

**Optimal mean block length (Politis & White 2004, corrected 2009):**

```
b_opt = ( 2·Ĝ² / D̂_SB )^{1/3} · N^{1/3},      p = 1 / b_opt
```

with `Ĝ = Σ λ(k/2m)·|k|·R̂(k)`, `D̂_SB = 2·ĝ(0)²`,
`ĝ(0) = Σ λ(k/2m)·R̂(k)` (flat-top lag-window estimates of the spectral
quantities at frequency zero), and bandwidth `m` chosen as the first lag
after which `K_n` consecutive sample autocorrelations are insignificant
(`±2·sqrt(log10 N / N)`).

The `N^{1/3}` *rate* matches the old heuristic, but the constant adapts to
the measured dependence: white-noise returns get `b ≈ 2` (near-i.i.d.
resampling), persistent/volatility-clustered returns get much longer blocks.
Clipped to `[2, min(3√N, N/3)]`; series shorter than 50 fall back to
`b = N^{1/3}`.

**Algorithm:**

```
for each simulation s:
    idx ← Uniform{0, …, N−1}
    for each step t:
        if t > 0 and Bernoulli(p):
            idx ← Uniform{0, …, N−1}   # start new block
        else:
            idx ← (idx + 1) mod N      # continue block
        out[s, t] ← centred[idx]
```

After resampling, returns are rescaled to the target `σ` and the Itô correction `−½σ²` is applied to the drift.

---

## 6. Detrended Fluctuation Analysis (DFA-1)

### Reference

Peng, C.-K. et al. (1994). "Mosaic organization of DNA nucleotides." *Physical Review E*, 49(2), 1685-1689.

### Why DFA instead of R/S Hurst?

The classical rescaled-range (R/S) estimator of Hurst (1951) is:

1. **Biased** on short series (< 200 observations). It systematically overestimates `H` for n < 100.
2. **Sensitive to non-stationarity**: trends and seasonalities inflate the estimate.
3. **No standard error** - the regression is not standard OLS.

DFA-1 integrates then locally detrends (removing linear trends within windows), making it valid on **non-stationary** series such as log-price levels.

### Algorithm (DFA-1)

1. **Integrate the centred series:**

```
Y(i) = Σ_{k≤i} (x_k − x̄)      (cumulative deviation from mean)
```

2. **For each box size `n` (powers of 2 from `min_box` to `N//4`):**
   - Split `Y` into `⌊N/n⌋` non-overlapping windows.
   - In each window, fit a linear trend with OLS and compute the RMS residual.
   - `F(n) = sqrt( mean of squared residuals over all windows )`

3. **OLS on log-log:**

```
log F(n) = α · log n + const      →     slope = α
```

4. **Return `(α, SE_α)`** where `SE_α` is the OLS standard error of the slope.

### Interpretation

| α range | Interpretation |
|---------|----------------|
| α < 0.45 | Anti-persistent (mean-reverting) |
| α ≈ 0.50 | White noise / random walk in return space |
| 0.55 < α < 1.0 | Long-range correlated (trending) |
| α ≈ 1.00 | Random walk in price level (1/f noise) |
| α > 1.0 | Non-stationary (e.g. Brownian motion of prices) |

### Usage in this codebase

- `dfa(np.log(prices))` - regime estimation on price levels (non-stationary OK)
- `dfa(log_returns)` - microstructure model on stationary return series

### Minimum sample size

With `min_box = 4` and powers-of-2 box sizes, at least 3 box sizes (4, 8, 16)
must satisfy `n ≤ N//4`, i.e. **N ≥ 64**, otherwise `dfa()` returns its
`(0.5, 0.0)` fallback. The microstructure regime estimator therefore requires
≥ 64 returns and uses up to the last 128 (4 box sizes). Note that DFA is known
to be biased and high-variance on short series; α estimates from < 250 points
should be treated as indicative, not precise.

### Significance gating of regime switches (permutation test)

Because of that noise, the regime classifier only leaves "neutral" when the
estimate is statistically significant. The OLS slope SE is **not** a valid
yardstick here: within one realisation the `F(n)` values are nearly
collinear, so the residual-based SE is tiny (~0.03), while the true
cross-realisation SD of α at N = 128 is ~0.09 — an SE-based gate fires on
pure noise roughly a third of the time.

Instead, an **exact permutation test** is used: compute α on `K = 79` random
shuffles of the same return window. Shuffling destroys serial correlation
but preserves the marginal distribution, so the null α's share the DFA
estimator's finite-sample bias and scatter — the test is self-calibrating.
Significance uses the exact Monte Carlo p-value with the "+1" correction
(Phipson & Smyth 2010):

```
p_hi = (1 + #{α_null ≥ α}) / (K + 1)      [trending direction]
p_lo = (1 + #{α_null ≤ α}) / (K + 1)      [mean-reverting direction]
```

requiring `p ≤ 0.025` per side **and** the level threshold (α > 0.55
trending, α < 0.45 mean-reverting). Interpolated percentiles of a small null
sample are anti-conservative (the observed value beats a noisy quantile
estimate too often); the exact p-value is guaranteed ≤ level under
exchangeability. False-switch rate on uncorrelated returns: ≤ 5% two-sided.

The null simulation only runs when α is already past a level threshold, so
the common neutral case costs a single DFA evaluation. `α` and the (purely
diagnostic) OLS `SE(α)` are exposed as `ms_dfa_alpha` / `ms_dfa_se`.

---

## 7. Monte Carlo Standard Errors

### Binomial SE of probability estimates

`prob_up`, `prob_flat`, `prob_down` are sample proportions from `n_sim` Bernoulli trials. Their standard errors (in percentage points) are:

```
SE(p̂) = sqrt( p̂ · (1 − p̂) / n_sim ) × 100
```

At `n_sim = 2000`, `p̂ = 0.5`: `SE ≈ 1.1 pp`. At `n_sim = 10 000`: `SE ≈ 0.5 pp`.

### SE of CVaR (tail-mean estimator)

The 5% CVaR is the sample mean of the worst `k = max(1, round(0.05 · n_sim))` terminal returns. Its SE is the standard SE of the sample mean applied to the tail:

```
SE(CVaR) = std(tail_returns) / sqrt(k) × 100
```

### Round-to-100 fix

Independent rounding of three proportions to 1 d.p. can yield sums of 99.9 or 100.1. The fix:

1. Round each component independently: `pu_r`, `pf_r`, `pd_r`.
2. Compute `error = 100.0 − (pu_r + pf_r + pd_r)`.
3. Add `error` to the component with the largest unrounded value (least distortion).

This guarantees `prob_up + prob_flat + prob_down == 100.0` as a floating-point identity.

---

## 8. Ensemble: Mixture, Not Average

### The wrong way (former implementation)

Combining model outputs as a weighted average of *independent* log-return
draws,

```
r = w_g·r_garch + w_b·r_boot + w_j·r_jump,     w_g + w_b + w_j = 1
```

shrinks the variance: for independent components each with variance ≈ σ²,

```
Var(r) = (w_g² + w_b² + w_j²) · σ²  ≈  0.4 σ²    (typical weights)
```

so the simulated bands and CVaR were ~35-40% too narrow. The mean was also
biased: each component already subtracts its own Itô term `½σ²`, so the
average has log-mean `drift − ½σ²` but variance only `≈ 0.4σ²`, giving

```
E[exp(r)] = exp(drift − ½σ² + ½·0.4σ²) = exp(drift − 0.3σ²)  <  exp(drift)
```

### The right way (current implementation)

Draw the number of paths per model from `Multinomial(n_sim; w_g, w_b, w_j)`
and simulate each path entirely under its assigned model. The result is the
intended **mixture distribution**:

```
F(x) = w_g·F_garch(x) + w_b·F_boot(x) + w_j·F_jump(x)
```

Each component preserves `E[exp(r)] = exp(drift)` on its own, so the mixture
does too, and the cross-sectional variance is the full mixture variance
(weighted mean of component variances plus between-component spread) instead
of the shrunken average-of-draws variance.

Weights are set empirically per call: vol-of-vol → GARCH weight (0.30-0.55),
excess kurtosis → jump weight (0.15-0.35), FHS takes the remainder
(0 when fewer than 30 returns of history are available).

---

## 9. Filtered Historical Simulation (FHS)

### Reference

Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999). VaR without
correlations for portfolios of derivative securities. *Journal of Futures
Markets*, 19(5), 583-602.

### Motivation

Two classic approaches each capture half the problem:

- **Historical simulation** keeps the real (skewed, fat-tailed) shock
  distribution but ignores volatility dynamics - yesterday's calm returns are
  replayed even in today's storm.
- **GARCH with Normal innovations** captures volatility dynamics but forces
  Gaussian shocks - tails are too thin.

FHS combines both. Comparative VaR studies find GARCH+FHS well calibrated
where plain HS and GARCH-Normal are badly miscalibrated.

### Algorithm

1. **Fit** GJR-GARCH(1,1) on the recent return window (cached MLE, §4).
2. **Filter** - run the fitted recursion in-sample and extract standardised
   residuals:

```
z_t = ε_t / σ_t
```

   After filtering, the `z_t` are approximately i.i.d. (the serial dependence
   lives in σ_t, not in z_t), so plain i.i.d. resampling is valid - no block
   bootstrap needed. The pool is re-centred and re-scaled to exactly unit
   variance so σ keeps its interpretation.

   Note that the filter *legitimately absorbs* much of the raw-return
   kurtosis into the σ_t dynamics (fat tails ≈ volatility clustering ×
   thinner-tailed innovations), so the residual pool is often much closer to
   Gaussian than the raw returns. Whatever shape remains - skew, residual
   kurtosis, asymmetry GARCH can't explain - is preserved exactly by the
   resampling.

3. **Rescale** the fitted process to the engine's blended volatility estimate.
   The anchor is the **seed**, not the fitted unconditional variance — MLE on
   a 90-bar window often pushes persistence toward the 0.999 boundary, which
   inflates `ω/(1−α−γ/2−β)` arbitrarily and would collapse the scale factor:

```
k        = σ_target² / σ²_last           (first simulated bar gets σ_target²)
LR       = clip(k · ω/(1−α−γ/2−β),  σ_target²/9,  9·σ_target²)
ω'       = LR · (1 − α − γ/2 − β)        (bounded long-run reversion target)
ε_last   → √k·ε_last,    σ²_last → σ_target²
```

4. **Simulate forward** - GJR variance recursion fed with residuals resampled
   uniformly from the pool:

```
σ²_t = ω + (α + γ·1[ε_{t−1}<0])·ε²_{t−1} + β·σ²_{t−1}
r_t  = (drift − ½σ²_t) + σ_t · z*        z* ~ Uniform(pool)
```

The Itô correction is exact for Gaussian shocks and approximate (to third-
moment order) for the empirical pool; the residual bias is `O(skew·σ³)` per
step - negligible at per-bar σ ≤ 0.1.

### Usage

- Standalone model: `model="fhs"` (falls back to Gaussian below 30 returns).
- Inside the ensemble (§8): FHS replaces the raw stationary bootstrap as the
  empirical component. The stationary bootstrap (§5) remains available as a
  standalone model.

---

## 10. Adaptive Conformal Calibration of the Outer Band

### Reference

Gibbs, I. & Candès, E. (2021). Adaptive conformal inference under
distribution shift. *NeurIPS 34*.

### Why

The Monte Carlo SEs of §7 quantify *sampling* noise only - they assume the
model is right. Model misspecification (wrong σ, wrong tails, regime breaks)
dominates in practice and makes nominal P10-P90 bands cover less (or more)
than 80% empirically.

### Method (ACI)

Every issued outer band is stored (`core/conformal.py`). When its horizon
elapses, the realised price is scored against it and the miscoverage level is
updated online:

```
err_t     = 1[realised outside band]
α_{t+1}   = α_t + γ·(α* − err_t),     α* = 0.20,  γ = 0.02
α_t       clipped to [0.02, 0.45]
```

The MC engine then extracts the outer band at percentiles
`[α_t/2, 1 − α_t/2]` instead of fixed `[0.10, 0.90]`. If bands have been
missing more than 20% of the time, α_t falls → wider band (toward P1-P99);
too-conservative bands tighten. The long-run average miscoverage converges
to α* regardless of distribution shift or serial dependence - the guarantee
comes from the online update, not from exchangeability (which time series
violate).

### Bookkeeping

- One α per `(ticker, interval, horizon)` triple, persisted in SQLite
  alongside the signal store.
- Nominal α = 0.20 is used until ≥ 10 bands have settled.
- **Warm start**: on first load of a (ticker, interval, horizon) the state is
  seeded by replaying reduced-size MC forecasts over the fetched history
  (pseudo-out-of-sample: each origin uses only bars ≤ t, scored against the
  close at t+horizon; origins spaced ≥ horizon/2 apart). Calibration numbers
  therefore appear immediately instead of after hours of live polling. Live
  settles then take over; a warm start never overwrites live state.
- Maturity is wall-clock (`horizon × interval`); settlement uses the first
  poll after maturity. Market closures add timing slack, which self-corrects.
- `MCResult.band_alpha` reports the level actually used; the
  `band_calibration` payload exposes empirical vs target coverage.

---

## References

- Black, F. & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637-654.
- Merton, R.C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1-2), 125-144.
- Hawkes, A.G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika*, 58(1), 83-90.
- Peng, C.-K., Buldyrev, S.V., Havlin, S., Simon, M., Stanley, H.E., & Goldberger, A.L. (1994). Mosaic organization of DNA nucleotides. *Physical Review E*, 49(2), 1685-1689.
- Politis, D.N. & Romano, J.P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313.
- Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *Journal of Futures Markets*, 19(5), 583-602.
- Politis, D.N. & White, H. (2004). Automatic block-length selection for the dependent bootstrap. *Econometric Reviews*, 23(1), 53-70. [Correction: Patton, Politis & White (2009), *Econometric Reviews*, 28(4), 372-375.]
- Engle, R.F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007.
- Glosten, L.R., Jagannathan, R. & Runkle, D.E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779-1801.
- Gibbs, I. & Candès, E. (2021). Adaptive conformal inference under distribution shift. *Advances in Neural Information Processing Systems*, 34.
- Phipson, B. & Smyth, G.K. (2010). Permutation p-values should never be zero. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.
- Bollerslev, T. (1986). Generalised autoregressive conditional heteroscedasticity. *Journal of Econometrics*, 31(3), 307-327.
