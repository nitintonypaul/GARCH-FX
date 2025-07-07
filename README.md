# GARCH-FX

A stochastic, regime-aware extension of traditional GARCH volatility forecasting.

---

## Overview 

**GARCH-FX**, or GARCH Forecasting eXtension, is an experimental and novel augmentation of the classic **GARCH(1,1)** model. While traditional GARCH models are robust, they often struggle to accurately capture the jagged, noisy behavior of realized volatility over longer horizons. This is because GARCH forecasts for long-term volatility tend to flatline, driven by parameters $$(\alpha, \beta, \omega)$$ that are trained on a specific historical window. Even though traditional GARCH models excel at precisely modeling the long-term mean variance (or average volatility) within that training period, their predictive power for extended future periods can be limited by this inherent smoothing.

GARCH-FX fills this gap by injecting **stochastic behavior** and dynamic **regime-switching** into volatility forecasts. It introduces structural enhancements to better reflect the uncertain nature of real-world volatility, mimicking market disorder with controlled randomness and regime shifts. This model isn't a replacement for GARCH; rather, it's a framework-level extension that respects GARCH mechanics but adds a stochastic engine and regime context for more flexible, adaptive forecasting. By leveraging the precisely trained parameters from GARCH, GARCH-FX forecasts volatility stochastically and in a regime-aware manner. This makes it particularly useful in applications requiring more dynamic and realistic long-term volatility projections, such as stress-testing, Monte Carlo simulations, or synthetic market generation.

---

## Note

GARCH-FX is experimental and should not be used for high-stakes production trading. Instead, it serves as a research playground for modeling volatility in a more expressive and intuitive manner.
