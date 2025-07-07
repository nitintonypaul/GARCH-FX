# Copyright (c) 2025 Nitin Tony Paul
# All rights reserved.

# Importing dependencies
import numpy as np
import sys

GLOBAL_SEED = None

# Regime switching function
def regimeswitcher(delta, regimeStates, regimes):
    
    # Basic static Markov chain of probabilities per regime
    if regimeStates == None:
        
        # 3 stage regime probabilities
        # regimeStates = np.array([[0.97, 0.029, 0.001], [0.015, 0.95, 0.035], [0.00, 0.04, 0.96]])

        # 9 stage regime probabilities (DEFAULT)
        regimeStates = np.array([
            [0.985, 0.015, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000], 
            [0.015, 0.965, 0.020, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000], 
            [0.000, 0.015, 0.960, 0.025, 0.000, 0.000, 0.000, 0.000, 0.000], 
            [0.000, 0.000, 0.020, 0.950, 0.030, 0.000, 0.000, 0.000, 0.000], 
            [0.000, 0.000, 0.000, 0.020, 0.945, 0.035, 0.000, 0.000, 0.000], 
            [0.000, 0.000, 0.000, 0.000, 0.025, 0.935, 0.040, 0.000, 0.000], 
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.030, 0.925, 0.045, 0.000], 
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.035, 0.950, 0.015], 
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.020, 0.980]
        ])
    
    if regimes == None:
        # Regime multiplier values
        regimes = [0.3, 0.6, 0.7, 0.8, 1.00, 1.2, 1.3, 1.4, 1.7]
    
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
    forecasts.append(volatility / 100)
    previousVariance = volatility ** 2
    OMEGA, ALPHA, BETA = params[0], params[1], params[2]
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
        forecasts.append(np.sqrt(previousVariance) / 100)

    return np.array(forecasts)
