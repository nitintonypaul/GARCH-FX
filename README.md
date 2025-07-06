# GARCH-FX

A stochastic, regime-aware extension of traditional GARCH volatility forecasting.

---

## Overview

GARCH-FX is an experimental & novel augmentation of the classic **GARCH(1,1)** model, designed to inject stochastic behavior and dynamic regime-switching into volatility forecasts. Traditional GARCH models, while robust, tend to smooth out or "flatten" during long-horizon forecasts. GARCH-FX introduces structural enhancements to better reflect the jagged, uncertain nature of real-world volatility.

This model is not a replacement for GARCH — it's a framework-level extension that respects GARCH mechanics but adds a stochastic engine and regime context for more flexible, adaptive forecasting.

---

## Key Features

**Stochastic Variance Injection:**
Forecasted variance is sampled from a Gamma distribution whose shape dynamically evolves with past volatility.

**Regime Awareness:**
A simple Markov Chain mechanism simulates market regimes (e.g., calm, normal, chaotic). Each regime alters volatility scaling via a Δ (delta) multiplier.

---

## Parameters

* **OMEGA, ALPHA, BETA**: Fitted GARCH(1,1) parameters
* **DELTA**: Market regime multiplier (e.g., 1 for normal, >1 for high-vol, <1 for decaying vol)
* **THETA**: Gamma scale — Proportional to volatility of volatility

---

## Why GARCH-FX?

Realized volatility often exhibits jagged, noisy behavior that classic GARCH fails to capture over longer horizons. GARCH-FX fills this gap by mimicking market market disorder with controlled randomness and regime shifts — particularly useful in stress-testing, Monte Carlo simulations, or synthetic market generation.

---

## Note

GARCH-FX is experimental and should not be used for high-stakes production trading. Instead, it serves as a research playground for modeling volatility in a more expressive and intuitive manner.
