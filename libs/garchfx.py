import numpy as np
import sys

GLOBAL_SEED = None

# Regime switching function
def regimeswitcher(delta, regimeStates, regimes):
    
    # Basic static Markov chain of probabilities per regime
    if regimeStates == None:
        regimeStates = np.array([[0.88, 0.10, 0.02], [0.06, 0.89, 0.05], [0.02, 0.33, 0.65]])
    if regimes == None:
        regimes = [0.7, 1, 1.3]
    
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
def fxforecast(volatility, nahead, params, delta, theta, reg=False, regimeStates=None, regimes=None):

    
    forecasts = []
    forecasts.append(volatility / 100)
    previousVariance = volatility ** 2
    OMEGA, ALPHA, BETA = params[0], params[1], params[2]
    np.random.seed(GLOBAL_SEED)

    for i in range(nahead):
        
        # Modelling shape to fit variance as the mode
        SHAPE = (previousVariance / theta) + 1

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
