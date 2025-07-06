import numpy as np
from arch import arch_model

# GARCH-FX forecasting
# Takes final conditional volatility, forecast horizon, GARCH parameters, delta and theta
# delta can be tweaked for regime shifting function
def fxforecast(volatility, nahead, params, delta, theta):
    
    forecasts = []
    forecasts.append(volatility / 100)
    previousVariance = volatility ** 2
    OMEGA, ALPHA, BETA = params[0], params[1], params[2]

    for i in range(nahead):

        # Modelling shape to fit variance as the mode
        SHAPE = (previousVariance / theta) + 1

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

def getGARCHdata(returns):
    # Using the normal GARCH model from arch package
    # GARCH(1,1) is used to compare
    model = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
    results = model.fit(disp='off')

    # Obtaining parameters and final volatility by using residuals to prepare manual forecasting
    OMEGA, ALPHA, BETA = results.params["omega"], results.params["alpha[1]"], results.params["beta[1]"]
    VOLATILITY = np.sqrt(results.forecast(horizon=1).variance.values[-1, :][0])

    return VOLATILITY, [ALPHA, BETA, OMEGA]