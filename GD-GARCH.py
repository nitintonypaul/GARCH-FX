import numpy as np
from arch import arch_model
import yfinance as yf
import matplotlib.pyplot as plt

# Constants
ANNUALIZER = np.sqrt(252)
FORECASTS = 1000
ticker = 'AAPL'

# Defining variables of this GARCH implementation
# SCALE controls the volatility wiggliness, i.e. volatility of volatility
# SCALE → 0 => Volatility models closer to GARCH volatility, OR the mean
# SCALE → 1 (or infinity) => Heavy volatility spikes are observed
SCALE = 0.001  

# DELTA is the regime shift variable. 
# D = 1 normal market mean. 
# D > 1 => highly volatile regime. (1.001)
# D < 1 => unusually calm regime. (0.999)
DELTA = 1

# Fetching data using yfinance
# AAPLreturns are scaled by 100 for better optimizer performance. It is scaled back by 100 whenever necessary
# Can be tweaked for backtesting
AAPL = yf.Ticker(ticker).history(period="180d")["Close"]
AAPLreturns = np.log(AAPL / AAPL.shift(1)).dropna()
AAPLreturns *= 100

# Using the normal GARCH model from arch package
# GARCH(1,1) is used to compare
model = arch_model(AAPLreturns, vol='Garch', p=1, q=1, dist='normal')
results = model.fit(disp='off')

# Obtaining parameters and final volatility by using residuals to prepare manual forecasting
OMEGA, ALPHA, BETA = results.params["omega"], results.params["alpha[1]"], results.params["beta[1]"]
VOLATILITY = np.sqrt(results.forecast(horizon=1).variance.values[-1, :][0])

# using GARCH to forecast
GARCHforecast = []
GARCHforecast.append(VOLATILITY / 100)
previousVariance = VOLATILITY ** 2
for i in range(FORECASTS):

    # GARCH equation
    newVariance = OMEGA + (ALPHA + BETA) * previousVariance
    previousVariance = newVariance

    # Appending GARCH volatility
    GARCHforecast.append(np.sqrt(previousVariance) / 100)


# using this new implementation of GARCH to forecast stochastically
GARCHsampled = []
previousSampledVariance = VOLATILITY ** 2
GARCHsampled.append(VOLATILITY / 100)
for i in range(FORECASTS):

    # Modelling shape to fit (DELTA * Variance) as the mode
    SHAPE = (DELTA * previousSampledVariance) / SCALE + 1
    
    # Sampling from gamma distribution
    stochasticVariance = np.random.gamma(shape=SHAPE, scale=SCALE, size=1)[0]

    # vanilla GARCH forecast equation
    newSampledVariance = OMEGA + (ALPHA + BETA) * stochasticVariance
    previousSampledVariance = newSampledVariance

    # Appending stochastic volatility
    GARCHsampled.append(np.sqrt(previousSampledVariance) / 100)

# Plotting both values
GARCHsampled = 100 * np.array(GARCHsampled)
GARCHforecast = 100 * np.array(GARCHforecast)

plt.plot(GARCHsampled, label=f"S-GARCH @ {SCALE}", alpha=0.8)
plt.plot(GARCHforecast, label="GARCH")
plt.xlabel("Forecasted Days")
plt.ylabel("Daily Volatility (%)")
plt.legend()
plt.grid(True)
plt.show()