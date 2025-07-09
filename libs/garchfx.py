# Copyright (c) 2025 Nitin Tony Paul
# All rights reserved.

# Importing dependencies
import numpy as np
import sys

# Global seed for comparison
GLOBAL_SEED = 100

# Regime switching function
def regimeswitcher(delta, regimeStates, regimes):

    # Basic static Markov chain of probabilities per regime
    if regimeStates == None:
        
        # 3 stage regime probabilities
        # Default
        regimeStates = np.array([[0.97, 0.029, 0.001], [0.015, 0.95, 0.035], [0.00, 0.04, 0.96]])
    
    if regimes == None:
        # Regime multiplier values
        regimes = [0.5, 1.0, 1.5]
    
    if len(regimeStates) != len(regimes):
        print("Invalid regime input")
        sys.exit(0)
    currentRegime = regimes.index(delta)

    # Generating random number for a chance to switch regimes
    r = np.random.rand()
    probs = regimeStates[currentRegime]
    new_regime = regimes[np.searchsorted(np.cumsum(probs), r)]

    # returning the next regime
    return new_regime

# GARCH-FX forecasting
# Takes final conditional volatility, forecast horizon, GARCH parameters, delta and theta
# delta can be tweaked for regime shifting function
def fxforecast(volatility, nahead, params, theta, reg=False, regimeStates=None, regimes=None):

    # Regime switcher
    delta = 1 

    # Assigning variables and forecast list
    forecasts = []
    forecasts.append(volatility)
    previousVariance = volatility ** 2
    ALPHA, BETA, OMEGA = params[0], params[1], params[2]
    np.random.seed(GLOBAL_SEED)

    for i in range(nahead-1):
        
        # Modelling shape to fit variance as the mode
        SHAPE = (previousVariance / theta) + 1

        # Regime switching if regime switching is enabled
        # enabled via `reg` argument
        if reg:
            delta = regimeswitcher(delta, regimeStates, regimes)
        
        # Sampling from gamma distribution
        stochasticVariance = np.random.gamma(shape=SHAPE, scale=theta, size=1)[0]

        # GARCH-FX equation
        forecastedVariance = (OMEGA * delta) + (ALPHA + BETA) * stochasticVariance
        previousVariance = forecastedVariance

        # Appending GARCH-FX volatility
        forecasts.append(np.sqrt(previousVariance))

    return np.array(forecasts)
