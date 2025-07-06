import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import garchfx as gfx
from garchfx import fxforecast, garchforecast, getGARCHdata

# Constants
FORECASTS = 1000
TICKER = 'GLD'

# Defining variables of this GARCH implementation
# SCALE controls the volatility wiggliness, i.e. volatility of volatility
# SCALE → 0 => Volatility models closer to GARCH volatility, OR the long term variance 
# SCALE → 1 (or infinity) => Heavy volatility spikes are observed
# SCALE = 1e-3 seems ideal
SCALE = 1e-3

# DELTA is the regime shift variable. 
# D = 1 normal market mean. 
# D > 1 => highly volatile regime. (1.2)
# D < 1 => unusually calm regime. (0.8)
DELTA = 1

# Fetching data using yfinance
# logReturns are scaled by 100 for better optimizer performance. It is scaled back by 100 whenever necessary
data = yf.Ticker(TICKER).history(period="2000d")["Close"]
realizedVolatility = 100 * gfx.realVol(np.log(data / data.shift(1)).dropna(), FORECASTS)
volatilityData = data[-1000:]
logReturns = np.log(volatilityData / volatilityData.shift(1)).dropna()
logReturns *= 100

# Obtaining GARCH conditional volatility and parameters
VOLATILITY, PARAMS = getGARCHdata(logReturns)

# Heston forecast
speed = 1 - PARAMS[1] - PARAMS[2]
variance = PARAMS[0]/speed
heston = gfx.heston(speed, variance, VOLATILITY**2)

# manual GARCH to forecast in percentages
GARCHforecast = 100 * garchforecast(VOLATILITY, nahead=FORECASTS, params=PARAMS)

# GARCH-FX stochastic forecasting extension in percentages
FXforecast = 100 * fxforecast(VOLATILITY, nahead=FORECASTS, params=PARAMS, delta=DELTA, theta=SCALE, reg=True)

# Plotting Values
plt.plot(realizedVolatility, label="Realized Vol.")
plt.plot(GARCHforecast, label="GARCH", linewidth=3)
plt.plot(FXforecast, label=f"GARCH-FX @ {SCALE}", alpha=0.8)
plt.plot(heston, label="Heston", alpha=0.8)
plt.title(f"{TICKER} - GARCH forecast vs GARCH-FX")
plt.xlabel("Forecasted Days")
plt.ylabel("Daily Volatility (%)")
plt.legend()
plt.grid(True)
plt.show()