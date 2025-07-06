import numpy as np
from arch import arch_model
import sys

GLOBAL_SEED = None

# GARCH-FX forecasting
# Takes final conditional volatility, forecast horizon, GARCH parameters, delta and theta
# delta can be tweaked for regime shifting function
def fxforecast(volatility, nahead, params, delta, theta, reg=False, regimeStates=None, regimes=None):

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
    
        # returning a new regime (might be same)
        return new_regime

    
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

# Normal GARCH forecasting
# Takes final conditional volatility, forecast horizon and GARCH parameters
def garchforecast(volatility, nahead, params):
    
    GARCHforecasts = []
    GARCHforecasts.append(volatility / 100)
    previousVariance = volatility ** 2

    OMEGA, ALPHA, BETA = params[0], params[1], params[2]
    for i in range(nahead):

        # GARCH equation
        newVariance = OMEGA + (ALPHA + BETA) * previousVariance
        previousVariance = newVariance

        # Appending GARCH volatility
        GARCHforecasts.append(np.sqrt(previousVariance) / 100)
    
    return np.array(GARCHforecasts)

# Function to obtain final conditional volatility and parameters
def getGARCHdata(returns):

    model = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
    results = model.fit(disp='off')

    # Obtaining parameters and final volatility by using residuals to prepare manual forecasting
    OMEGA, ALPHA, BETA = results.params["omega"], results.params["alpha[1]"], results.params["beta[1]"]
    VOLATILITY = np.sqrt(results.forecast(horizon=1).variance.values[-1, :][0])

    return VOLATILITY, [ALPHA, BETA, OMEGA]

# Function to get realized volatility
def realVol(returns, steps):

    # Assigning a small np array
    volarr = np.zeros(1)

    # Find each volatility
    for i in range(1000, 1000+steps):
        partial = returns[:i][-180:]
        volarr = np.append(volarr, np.std(partial))
    
    # Deleting the first element (0) and returning
    return np.delete(volarr, 0)

# Heston model function to model volatility
def heston(kappa, theta, v0):
    
    # Heston parameters
    sigma = 0.6
    T = 2.75          
    N = 1001       
    dt = T / N

    # Simulate
    np.random.seed(GLOBAL_SEED)
    vt = np.zeros(N)
    vt[0] = v0

    for t in range(1, N):
        z = np.random.normal()
        vt[t] = vt[t-1] + kappa * (theta - vt[t-1]) * dt + sigma * np.sqrt(max(vt[t-1], 0)) * np.sqrt(dt) * z
        vt[t] = max(vt[t], 0)

    return np.sqrt(vt)
