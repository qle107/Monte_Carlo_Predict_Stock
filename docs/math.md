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

At `σ = 0.02`, `n = 10`:  bias `≈ exp(0.002) − 1 ≈ 0.2%` — visible on a chart.

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
- `κ = E[e^J − 1]` is the **jump compensator** — the drift adjustment that keeps the process a martingale under the risk-neutral measure

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

At `λ=0.05`, `σ_J = 5σ = 0.10`: missing bias `≈ 0.05 · 0.005 = 0.00025` per step, or `0.0025` over 10 steps — detectable without noise.

### Implementation

```python
kappa = float(np.exp(jump_mean + 0.5 * sigma_jump**2) - 1.0)
drift_eff = drift - jump_intensity * kappa
```

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

## 4. GARCH(1,1) Likelihood

### Model

```
σ²_t = ω + α · ε²_{t−1} + β · σ²_{t−1}
ε_t  = σ_t · z_t,    z_t ~ N(0,1)
```

Stationarity requires `α + β < 1`; unconditional variance `= ω / (1 − α − β)`.

### Maximum likelihood estimation

The (quasi-)log-likelihood is:

```
ℓ(ω, α, β) = −½ Σ_t [ log(2π σ²_t) + ε²_t / σ²_t ]
```

Optimised with Nelder-Mead (`scipy.optimize.minimize`), max 400 iterations. Results are cached for 5 minutes (keyed on the last 90 returns) to avoid re-running the optimiser on every poll cycle.

---

## 5. Stationary Bootstrap (Politis & Romano, 1994)

### Why not i.i.d. resampling?

Naive i.i.d. bootstrap draws returns independently, destroying all serial autocorrelation. Volatility clustering (GARCH-like behaviour) and momentum measured by `ACF(1) > 0` both vanish → systematically underestimates tail risk under trending regimes.

### Block bootstrap

Draw contiguous blocks of returns of length `L`. Fixed block length (Künsch 1989) breaks at fixed boundaries, introducing an artificial jump in correlation at multiples of `L`.

### Stationary bootstrap

Politis & Romano (1994) randomise the block length: `L ~ Geometric(p)` with `E[L] = 1/p = b`. The starting position of each block is drawn uniformly from `{0, …, N−1}` (with wraparound). The process restarts (picks a new starting index) with probability `p` at each step.

**Optimal mean block length:**

```
b = N^{1/3}         (optimal MSE rate for general stationary processes)
p = 1 / b
```

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

Peng, C.-K. et al. (1994). "Mosaic organization of DNA nucleotides." *Physical Review E*, 49(2), 1685–1689.

### Why DFA instead of R/S Hurst?

The classical rescaled-range (R/S) estimator of Hurst (1951) is:

1. **Biased** on short series (< 200 observations). It systematically overestimates `H` for n < 100.
2. **Sensitive to non-stationarity**: trends and seasonalities inflate the estimate.
3. **No standard error** — the regression is not standard OLS.

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

- `dfa(np.log(prices))` — regime estimation on price levels (non-stationary OK)
- `dfa(log_returns)` — microstructure model on stationary return series

---

## 7. Monte Carlo Standard Errors

### Binomial SE of probability estimates

`prob_up`, `prob_flat`, `prob_down` are sample proportions from `n_sim` Bernoulli trials. Their standard errors (in percentage points) are:

```
SE(p̂) = sqrt( p̂ · (1 − p̂) / n_sim ) × 100
```

At `n_sim = 2000`, `p̂ = 0.5`: `SE ≈ 1.1 pp`. At `n_sim = 10 000`: `SE ≈ 0.5 pp`.

### SE of CVaR (tail-mean estimator)

The 5% CVaR is the sample mean of the worst `k = ⌈0.05 · n_sim⌉` terminal returns. Its SE is the standard SE of the sample mean applied to the tail:

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

## References

- Black, F. & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637–654.
- Merton, R.C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1–2), 125–144.
- Peng, C.-K., Buldyrev, S.V., Havlin, S., Simon, M., Stanley, H.E., & Goldberger, A.L. (1994). Mosaic organization of DNA nucleotides. *Physical Review E*, 49(2), 1685–1689.
- Politis, D.N. & Romano, J.P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.
- Engle, R.F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007.
- Bollerslev, T. (1986). Generalised autoregressive conditional heteroscedasticity. *Journal of Econometrics*, 31(3), 307–327.
