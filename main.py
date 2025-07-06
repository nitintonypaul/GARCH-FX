import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from garchfx import fxforecast, garchforecast, getGARCHdata

# Constants
FORECASTS = 1000
ticker = 'AAPL'

# Defining variables of this GARCH implementation
# SCALE controls the volatility wiggliness, i.e. volatility of volatility
# SCALE → 0 => Volatility models closer to GARCH volatility, OR the long term variance 
# SCALE → 1 (or infinity) => Heavy volatility spikes are observed
SCALE = 1e-3

# DELTA is the regime shift variable. 
# D = 1 normal market mean. 
# D > 1 => highly volatile regime. (1.2)
# D < 1 => unusually calm regime. (0.8)
DELTA = 1

# Fetching data using yfinance
# logReturns are scaled by 100 for better optimizer performance. It is scaled back by 100 whenever necessary
data = yf.Ticker(ticker).history(period="1000d")["Close"]
logReturns = np.log(data / data.shift(1)).dropna()
logReturns *= 100

# Obtaining GARCH conditional volatility and parameters
VOLATILITY, PARAMS = getGARCHdata(logReturns)

# manual GARCH to forecast in percentages
GARCHforecast = 100 * garchforecast(VOLATILITY, nahead=FORECASTS, params=PARAMS)

# GARCH-FX stochastic forecasting extension in percentages
bigFX = 100 * fxforecast(VOLATILITY, nahead=FORECASTS, params=PARAMS, delta=DELTA, theta=1e-2)
FXforecast = 100 * fxforecast(VOLATILITY, nahead=FORECASTS, params=PARAMS, delta=DELTA, theta=SCALE)
smallFX = 100 * fxforecast(VOLATILITY, nahead=FORECASTS, params=PARAMS, delta=DELTA, theta=1e-4)

# Plotting Values
plt.plot(GARCHforecast, label="GARCH", linewidth=3)
plt.plot(bigFX, label=f"S-GARCH @ 0.01", alpha=0.4)
plt.plot(FXforecast, label=f"S-GARCH @ {SCALE}", alpha=0.6)
plt.plot(smallFX, label="S-GARCH @ 0.0001", alpha=0.8)
plt.title("AAPL - GARCH forecast vs GARCH-FX")
plt.xlabel("Forecasted Days")
plt.ylabel("Daily Volatility (%)")
plt.legend()
plt.grid(True)
plt.show()